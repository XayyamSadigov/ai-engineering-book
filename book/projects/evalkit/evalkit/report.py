# path: book/projects/evalkit/evalkit/report.py
"""Markdown evaluation report: lineage, metrics with CIs, slices, comparison, per-case deltas.

The report is the artifact a reviewer reads before approving a release, and the artifact you
attach to the release so it stays auditable. It leads with lineage because a number without
versions is not evidence.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

from .gate import GateResult
from .runner import Run
from .stats import ALL, bootstrap_ci, compare_runs, compare_slices, per_case_deltas, slice_breakdown


def _f(x: float, digits: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def _lineage(run: Run) -> list[str]:
    v = run.versions
    rows = [
        ("run", run.run_id),
        ("created", run.created_at.isoformat(timespec="seconds")),
        ("dataset", v.dataset),
        ("target", v.target),
        ("prompt", v.prompt or "-"),
        ("model", v.model or "-"),
        ("evaluators", ", ".join(f"{k}@{val}" for k, val in sorted(v.evaluators.items())) or "-"),
        *((k, val) for k, val in sorted(v.extra.items())),
    ]
    return ["| field | value |", "|---|---|", *[f"| {k} | `{val}` |" for k, val in rows]]


def render_report(
    run: Run,
    *,
    baseline: Run | None = None,
    metrics: Sequence[str] | None = None,
    gate: GateResult | None = None,
    title: str | None = None,
    slice_min_n: int = 3,
    max_case_deltas: int = 15,
    n_resamples: int = 2000,
    seed: int = 0,
) -> str:
    metrics = list(metrics or run.metric_names())
    out: list[str] = [f"# {title or 'Evaluation report'}: {run.versions.target}", ""]
    if gate is not None:
        out += [gate.to_markdown(), ""]

    out += ["## Lineage", "", *_lineage(run), ""]
    if baseline is not None:
        out += [f"Baseline run `{baseline.run_id}`: target `{baseline.versions.target}`, "
                f"prompt `{baseline.versions.prompt or '-'}`, model `{baseline.versions.model or '-'}`.", ""]

    n_cases = len(run.by_case())
    out += [
        "## Operations",
        "",
        "| cases | calls | errors | p50 ms | p95 ms | total cost usd | cost/case usd | wall s |",
        "|---|---|---|---|---|---|---|---|",
        f"| {n_cases} | {len(run.results)} | {len(run.errors)} | {_f(run.latency_percentile(50), 0)} | "
        f"{_f(run.latency_percentile(95), 0)} | {run.total_cost_usd:.4f} | {run.cost_per_case_usd:.5f} | {run.wall_time_s:.1f} |",
        "",
    ]
    if run.errors:
        kinds: dict[str, int] = {}
        for r in run.errors:
            kinds[r.error_type or "error"] = kinds.get(r.error_type or "error", 0) + 1
        out += ["Errors by type: " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())), ""]

    out += ["## Metrics", ""]
    if baseline is None:
        out += ["| metric | mean [95% CI] | pass rate | n |", "|---|---|---|---|"]
        for m in metrics:
            ci = bootstrap_ci(list(run.case_scores(m).values()), n_resamples=n_resamples, seed=seed)
            out.append(f"| {m} | {_f(ci.estimate)} [{_f(ci.low)}, {_f(ci.high)}] | {_f(run.pass_rate(m))} | {ci.n} |")
    else:
        out += ["| metric | baseline | candidate | delta [95% CI] | p | W/L/T |", "|---|---|---|---|---|---|"]
        for m in metrics:
            try:
                pd = compare_runs(baseline, run, m, n_resamples=n_resamples, seed=seed)
            except ValueError:
                out.append(f"| {m} | n/a | {_f(run.mean(m))} | n/a | n/a | n/a |")
                continue
            out.append(
                f"| {m} | {_f(pd.baseline_mean)} | {_f(pd.candidate_mean)} | {pd.delta:+.3f} "
                f"[{pd.low:+.3f}, {pd.high:+.3f}] | {pd.p_value:.3f} | {pd.wins}/{pd.losses}/{pd.ties} |"
            )
    if run.repeats > 1:
        out += ["", "Flaky cases (verdict differs across repeats): "
                + "; ".join(f"{m}: {len(run.flaky_cases(m))}" for m in metrics)]
    out.append("")

    out += ["## Slices", ""]
    for m in metrics:
        if baseline is None:
            stats = [s for s in slice_breakdown(run, m, min_n=slice_min_n, n_resamples=n_resamples, seed=seed)]
            out += [f"### {m}", "", "| slice | n | mean [95% CI] |", "|---|---|---|"]
            out += [f"| {'all' if s.slice == ALL else s.slice} | {s.n} | {_f(s.mean)} [{_f(s.low)}, {_f(s.high)}] |" for s in stats]
        else:
            deltas = compare_slices(baseline, run, m, min_n=slice_min_n, n_resamples=n_resamples, seed=seed)
            out += [f"### {m}", "", "| slice | n | baseline | candidate | delta [95% CI] |", "|---|---|---|---|---|"]
            out += [
                f"| {'all' if s.slice == ALL else s.slice} | {s.n} | {_f(s.baseline_mean)} | {_f(s.candidate_mean)} | "
                f"{s.delta:+.3f} [{s.low:+.3f}, {s.high:+.3f}] |"
                for s in deltas
            ]
        out.append("")

    if baseline is not None:
        out += ["## Per-case changes", ""]
        for m in metrics:
            deltas = per_case_deltas(baseline, run, m)
            if not deltas:
                continue
            regressions = [d for d in deltas if d.delta < 0]
            fixes = [d for d in deltas if d.delta > 0]
            out += [f"### {m}: {len(regressions)} regressed, {len(fixes)} improved", "",
                    "| case | baseline | candidate | delta | tags |", "|---|---|---|---|---|"]
            shown = regressions[:max_case_deltas] + list(reversed(fixes))[: max(0, max_case_deltas - len(regressions))]
            out += [f"| {d.case_id} | {_f(d.baseline)} | {_f(d.candidate)} | {d.delta:+.3f} | {', '.join(d.tags)} |" for d in shown]
            out.append("")
    else:
        failing = {m: run.failing_cases(m) for m in metrics}
        if any(failing.values()):
            out += ["## Failing cases", ""]
            for m, ids in failing.items():
                if ids:
                    out.append(f"- **{m}** ({len(ids)}): " + ", ".join(ids[:max_case_deltas]) + (" ..." if len(ids) > max_case_deltas else ""))
            out.append("")
    return "\n".join(out).rstrip() + "\n"


__all__ = ["render_report"]
