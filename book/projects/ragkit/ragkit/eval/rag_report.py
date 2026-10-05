# path: book/projects/ragkit/ragkit/eval/rag_report.py
"""Markdown report for a RAG evaluation run, optionally compared with a baseline.

Order of sections follows the order a reviewer should read them in:

1. Gate verdict.
2. Permission leaks, separately and first. Never averaged, never traded against quality.
3. Retrieval metrics with bootstrap CIs (or paired deltas against the baseline).
4. Answer metrics, including the abstention confusion table.
5. Stage isolation: where the failing cases lost their evidence, and how that moved.
6. Slices by tag for the headline metrics.
7. Per-case regressions.
Lineage comes from evalkit's run record so the report can be audited later.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from evalkit import Dataset, GateResult, Run, bootstrap_ci, compare_runs, compare_slices, per_case_deltas, slice_breakdown
from evalkit.stats import ALL

from .rag_dataset import RagOutput, expectation
from .rag_metrics import abstention_cell
from .stage_isolation import OWNER, FailureStage, StageDiagnosis, stage_counts, stage_shift

HEADLINE_RETRIEVAL = ["hit@1", "hit@5", "recall@5", "recall@10", "precision@5", "mrr", "ndcg@10", "context_relevance", "evidence_packed"]
HEADLINE_ANSWER = ["abstention_correct", "citation_precision", "citation_recall", "citations_valid",
                   "faithfulness", "contradiction_free", "rubric_coverage", "relevance"]


def _f(x: float | None, d: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def _metric_table(run: Run, metrics: Sequence[str], baseline: Run | None, n_resamples: int, seed: int) -> list[str]:
    present = [m for m in metrics if m in run.metric_names()]
    if not present:
        return ["_no metrics of this kind in the run_", ""]
    if baseline is None:
        rows = ["| metric | mean [95% CI] | n |", "|---|---|---|"]
        for m in present:
            ci = bootstrap_ci(list(run.case_scores(m).values()), n_resamples=n_resamples, seed=seed)
            rows.append(f"| {m} | {_f(ci.estimate)} [{_f(ci.low)}, {_f(ci.high)}] | {ci.n} |")
    else:
        rows = ["| metric | baseline | candidate | delta [95% CI] | p | W/L/T | n |", "|---|---|---|---|---|---|---|"]
        for m in present:
            try:
                pd = compare_runs(baseline, run, m, n_resamples=n_resamples, seed=seed)
            except ValueError:
                rows.append(f"| {m} | n/a | {_f(run.mean(m))} | n/a | n/a | n/a | n/a |")
                continue
            flag = " **sig**" if pd.significant else ""
            rows.append(
                f"| {m} | {_f(pd.baseline_mean)} | {_f(pd.candidate_mean)} | {pd.delta:+.3f} "
                f"[{pd.low:+.3f}, {pd.high:+.3f}]{flag} | {pd.p_value:.3f} | {pd.wins}/{pd.losses}/{pd.ties} | {pd.n} |"
            )
    return rows + [""]


def _acl_dropped_warning(run: Run) -> list[str]:
    """Chapter 12 retrievers re-check ACLs on what their store returned and count the drops.
    Nothing leaked, but a non-zero count means a store or filter returned forbidden rows."""
    cases = []
    for r in run.results:
        trace = (r.output or {}).get("retrieval", {}) if isinstance(r.output, dict) else {}
        dropped = (trace or {}).get("trace", {}).get("acl_dropped", 0) if trace else 0
        if isinstance(dropped, (int, float)) and dropped > 0:
            cases.append(f"{r.case_id} ({int(dropped)})")
    if not cases:
        return []
    return [f"Warning, not a blocker: retrievers' own ACL re-check dropped forbidden rows in {len(cases)} case(s): "
            + ", ".join(cases[:15]) + ". Investigate the store filter.", ""]


def _leak_section(run: Run) -> list[str]:
    """Leaks first, and cases whose leak status is unknown right after them.

    A target that crashed was never checked for leaks. evalkit records its detail as the string
    "target_error" (and its score as the run's error score, which may be None). Such a case is not
    "no leak": it is listed as an evaluation error that blocks the release like a leak would, and the
    "none" line is printed only when every case was actually checked.
    """
    def unchecked(r) -> bool:  # type: ignore[no-untyped-def]
        return r.error is not None or not isinstance(r.details.get("no_permission_leak", {}), dict)

    errored = [r for r in run.results if unchecked(r)]
    leaks = [r for r in run.results if not unchecked(r) and r.scores.get("no_permission_leak") == 0.0]
    out = ["## Permission leaks", ""]
    if not leaks and not errored:
        return out + [f"None across {len(run.by_case())} cases. Required: zero.", ""] + _acl_dropped_warning(run)
    if leaks:
        out += [f"**{len(leaks)} case(s) leaked restricted documents. This blocks the release regardless of quality.**",
                "", "| case | tags | leaked docs |", "|---|---|---|"]
        for r in leaks:
            detail = r.details.get("no_permission_leak") or {}
            docs = sorted({*detail.get("forbidden_retrieved", []), *detail.get("forbidden_packed", []),
                           *detail.get("forbidden_cited", []),
                           *(c.rsplit(":", 1)[0] for c in detail.get("acl_violations", []))})
            out.append(f"| {r.case_id} | {', '.join(r.tags)} | {', '.join(docs)} |")
        out.append("")
    else:
        out += ["No leak observed in the cases that ran, but not every case ran.", ""]
    if errored:
        out += [f"**{len(errored)} case(s) could not be checked for leaks because the system under test failed. "
                "Their leak status is unknown, which blocks the release like a leak.**", "",
                "| case | tags | evaluation error |", "|---|---|---|"]
        for r in errored:
            raw = r.details.get("no_permission_leak")
            reason = f"{r.error_type or 'Error'}: {r.error}" if r.error is not None else str(raw)
            out.append(f"| {r.case_id} | {', '.join(r.tags)} | {reason.replace('|', '/')[:160]} |")
        out.append("")
    return out + _acl_dropped_warning(run)


def _abstention_table(run: Run, dataset: Dataset) -> list[str]:
    cells: Counter[str] = Counter()
    for r in run.results:
        if r.repeat or r.output is None or r.case_id not in dataset:
            continue
        out = RagOutput.coerce(r.output)
        cells[abstention_cell(out.abstained, expectation(dataset.get(r.case_id)).expect_abstain)] += 1
    return [
        "| | system answered | system abstained |", "|---|---|---|",
        f"| abstention expected | false-answer: {cells['false-answer']} | correct-abstain: {cells['correct-abstain']} |",
        f"| answer expected | answered: {cells['answered']} | false-abstain: {cells['false-abstain']} |", "",
    ]


def _stage_section(diagnoses: Sequence[StageDiagnosis], baseline_diagnoses: Sequence[StageDiagnosis] | None) -> list[str]:
    counts = stage_counts(diagnoses)
    total = len(diagnoses)
    before = stage_counts(baseline_diagnoses) if baseline_diagnoses else None
    head = "| stage | cases | share | baseline | where to look | examples |" if before is not None else \
        "| stage | cases | share | where to look | examples |"
    out = ["## Stage isolation", "", "Each case is labelled with the earliest stage that lost its evidence.", "",
           head, "|---|---|---|---|---|" + ("---|" if before is not None else "")]
    order = [s for s in FailureStage if s != FailureStage.OK] + [FailureStage.OK]
    for s in order:
        n = counts.get(s.value, 0)
        b = before.get(s.value, 0) if before is not None else None
        if n == 0 and not b:
            continue
        ex = ", ".join(d.case_id for d in diagnoses if d.stage == s)[:60]
        base_col = f" {b} |" if before is not None else ""
        out.append(f"| {s.value} | {n} | {n / total:.0%} |{base_col} {OWNER.get(s, '-')} | {ex} |")
    out.append("")
    if baseline_diagnoses:
        shifts = stage_shift(baseline_diagnoses, diagnoses)
        if shifts:
            out += ["Label changes against the baseline:", "", "| case | baseline | candidate |", "|---|---|---|"]
            out += [f"| {c} | {a} | {b} |" for c, a, b in shifts]
            out.append("")
    return out


def _slices(run: Run, metric: str, baseline: Run | None, min_n: int, n_resamples: int, seed: int) -> list[str]:
    if metric not in run.metric_names():
        return []
    if baseline is None:
        stats = slice_breakdown(run, metric, min_n=min_n, n_resamples=n_resamples, seed=seed)
        rows = [f"### {metric}", "", "| slice | n | mean [95% CI] |", "|---|---|---|"]
        rows += [f"| {'all' if s.slice == ALL else s.slice} | {s.n} | {_f(s.mean)} [{_f(s.low)}, {_f(s.high)}] |" for s in stats]
    else:
        deltas = compare_slices(baseline, run, metric, min_n=min_n, n_resamples=n_resamples, seed=seed)
        rows = [f"### {metric}", "", "| slice | n | baseline | candidate | delta [95% CI] |", "|---|---|---|---|---|"]
        rows += [f"| {'all' if s.slice == ALL else s.slice} | {s.n} | {_f(s.baseline_mean)} | {_f(s.candidate_mean)} | "
                 f"{s.delta:+.3f} [{s.low:+.3f}, {s.high:+.3f}] |" for s in deltas]
    return rows + [""]


def render_rag_report(
    run: Run,
    dataset: Dataset,
    *,
    diagnoses: Sequence[StageDiagnosis] | None = None,
    baseline: Run | None = None,
    baseline_diagnoses: Sequence[StageDiagnosis] | None = None,
    gate: GateResult | None = None,
    title: str = "RAG evaluation",
    slice_metrics: Sequence[str] = ("recall@5", "faithfulness", "abstention_correct"),
    slice_min_n: int = 2,
    n_resamples: int = 2000,
    seed: int = 0,
) -> str:
    v = run.versions
    out = [f"# {title}: {v.target}", ""]
    if gate is not None:
        out += [gate.to_markdown(), ""]
    out += ["## Lineage", "", "| field | value |", "|---|---|",
            f"| run | `{run.run_id}` |", f"| dataset | `{v.dataset}` |", f"| target | `{v.target}` |",
            f"| model | `{v.model or '-'}` |",
            f"| evaluators | `{', '.join(f'{k}@{x}' for k, x in sorted(v.evaluators.items()))}` |",
            *(f"| {k} | `{x}` |" for k, x in sorted(v.extra.items()))]
    if baseline is not None:
        out.append(f"| baseline | `{baseline.run_id}` target `{baseline.versions.target}` |")
    out += ["", f"Cases: {len(run.by_case())}. Target errors: {len(run.errors)}. Evaluator errors: {run.evaluator_error_count}. "
            f"p95 latency: {_f(run.latency_percentile(95), 0)} ms.", ""]
    out += _leak_section(run)
    out += ["## Retrieval", "", "Computed on answerable cases only; forbidden-doc and abstain cases are scored by leaks and abstention.", ""]
    out += _metric_table(run, HEADLINE_RETRIEVAL, baseline, n_resamples, seed)
    out += ["## Answers", ""]
    out += _metric_table(run, HEADLINE_ANSWER, baseline, n_resamples, seed)
    out += ["Abstention outcomes:", ""] + _abstention_table(run, dataset)
    if diagnoses is not None:
        out += _stage_section(diagnoses, baseline_diagnoses)
    out += ["## Slices", ""]
    for m in slice_metrics:
        out += _slices(run, m, baseline, slice_min_n, n_resamples, seed)
    if baseline is not None:
        out += ["## Per-case regressions", ""]
        any_reg = False
        for m in ("recall@5", "evidence_packed", "abstention_correct", "faithfulness", "rubric_coverage"):
            if m not in run.metric_names():
                continue
            regs = [d for d in per_case_deltas(baseline, run, m) if d.delta < 0]
            if regs:
                any_reg = True
                out.append(f"- **{m}**: " + ", ".join(f"{d.case_id} ({d.delta:+.2f})" for d in regs[:15]))
        out += (["- none"] if not any_reg else []) + [""]
    return "\n".join(out).rstrip() + "\n"


__all__ = ["render_rag_report", "HEADLINE_RETRIEVAL", "HEADLINE_ANSWER"]
