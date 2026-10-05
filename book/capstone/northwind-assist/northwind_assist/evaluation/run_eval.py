# path: book/capstone/northwind-assist/northwind_assist/evaluation/run_eval.py
"""Offline evaluation and release gate: `northwind-assist-eval` (exit 0 pass, 1 fail, 2 error).

    northwind-assist-eval --out eval/out                         # candidate config: gate passes
    northwind-assist-eval --out eval/out-broken --set rag_enforce_acl=false   # leaks: gate fails

Steps: build the system from settings (plus `--set key=value` overrides, which is how CI tests a
candidate configuration), run the three suites through the real orchestrator, save each evalkit
Run as JSON, then hand the runs to Chapter 25's release gate script, which applies eval/gates.toml
and writes summary.md and gate.json. The capstone adds report.md: the Chapter 14 RAG report with
stage isolation, latency and cost tables, and the judge calibration.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from evalkit import RunVersions, run_target
from ragkit.eval.rag_report import render_rag_report
from ragkit.eval.stage_isolation import diagnose_run

from .. import _paths
from ..config import Settings
from ..container import build_container
from ..observability.tracing import build_tracer
from ..rag.service import RAG_PROMPT
from .datasets import rag_dataset, security_dataset, tool_dataset, write_attack_docs
from .suites import judge_calibration, rag_evaluators, rag_target, security_target, tool_evaluators, tool_target

DEFAULT_GATES = _paths.CAPSTONE_ROOT / "eval" / "gates.toml"


def _overrides(pairs: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        try:
            out[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            out[key.strip()] = raw
    return out


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def run_suites(settings: Settings, out: Path, suites: list[str]) -> dict[str, Any]:
    runs_dir = out / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"config": settings.behavior_config()}
    tracer = build_tracer({"TRACE_SINK": "memory"})
    c = build_container(settings, tracer=tracer)
    versions = RunVersions(target="northwind-assist", prompt=RAG_PROMPT, model="router:nw-general",
                           extra={k: str(v) for k, v in c.base_manifest.flatten().items() if not k.startswith("git")})
    report_parts: list[str] = []
    if "rag" in suites:
        ds = rag_dataset(c.kb.doc_visible)
        run = run_target(rag_target(c), ds, versions=versions, evaluators=rag_evaluators(), concurrency=1)
        run.save_json(runs_dir / "rag.json")
        diagnoses = diagnose_run(run, ds, doc_visible=lambda d, p: c.kb.doc_visible(d, p.tenant, p.groups))
        report_parts.append(render_rag_report(run, ds, diagnoses=diagnoses, title="RAG suite (candidate)"))
        lat = [cr.latency_ms for cr in run.results]
        summary["rag"] = {"cases": len(ds), **{m: round(run.mean(m), 3) for m in (
            "no_permission_leak", "recall@5", "hit@1", "mrr", "abstention_correct", "citation_precision",
            "citations_valid", "faithfulness_lexical")},
            "p50_ms": round(_percentile(lat, 50), 1), "p95_ms": round(_percentile(lat, 95), 1),
            "cost_per_case_usd": round(run.cost_per_case_usd, 6)}
    if "tools" in suites:
        tc = build_container(settings, tracer=build_tracer({"TRACE_SINK": "memory"}))
        ds = tool_dataset()
        run = run_target(tool_target(tc), ds, versions=versions, evaluators=tool_evaluators(tc), concurrency=1)
        run.save_json(runs_dir / "tools.json")
        summary["tools"] = {"cases": len(ds), **{m: round(run.mean(m), 3) for m in (
            "traj_safe", "traj_success", "traj_tool_args", "traj_efficiency", "world_safe")},
            "cost_per_case_usd": round(run.cost_per_case_usd, 6)}
    if "security" in suites:
        with tempfile.TemporaryDirectory() as tmp:
            write_attack_docs(Path(tmp))
            sec_settings = settings.model_copy(update={"extra_docs_dir": tmp})
            sc = build_container(sec_settings, tracer=build_tracer({"TRACE_SINK": "memory"}))
            target, evaluator = security_target(sc)
            ds = security_dataset()
            run = run_target(target, ds, versions=versions, evaluators=[evaluator], concurrency=1)
        run.save_json(runs_dir / "security.json")
        failing = [cr.case_id for cr in run.results if cr.passed.get("effect_prevented") is False]
        summary["security"] = {"cases": len(ds), "effect_prevented": round(run.mean("effect_prevented"), 3),
                               "attack_detected": round(run.mean("attack_detected"), 3), "failing": failing}
    summary["judge_calibration"] = judge_calibration()
    summary["cost_report"] = c.ledger.daily_report()
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "report.md").write_text("\n\n".join(report_parts) + "\n\n## Judge calibration\n\n```json\n" +
                                   json.dumps(summary["judge_calibration"], indent=2) + "\n```\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("eval/out"))
    ap.add_argument("--gates", type=Path, default=DEFAULT_GATES)
    ap.add_argument("--baselines", type=Path, default=None)
    ap.add_argument("--suites", default="rag,tools,security")
    ap.add_argument("--set", dest="sets", action="append", default=[], help="settings override key=value")
    args = ap.parse_args(argv)
    try:
        settings = Settings(environment="test", **_overrides(args.sets))
        summary = run_suites(settings, args.out, [s.strip() for s in args.suites.split(",") if s.strip()])
    except Exception as exc:  # setup failure: the gate could not be evaluated
        print(f"eval setup error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({k: v for k, v in summary.items() if k != "cost_report"}, indent=2, default=str))
    # Chapter 25's ci/release_gate.py, on sys.path through northwind_assist._paths
    import release_gate  # type: ignore[import-not-found]

    gate_args = ["--config", str(args.gates), "--runs", str(args.out / "runs"), "--out", str(args.out)]
    if args.baselines:
        gate_args += ["--baselines", str(args.baselines)]
    suites = {s.strip() for s in args.suites.split(",")}
    if suites != {"rag", "tools", "security"}:   # gate only the suites that ran
        import tomllib  # noqa: PLC0415

        cfg = tomllib.loads(args.gates.read_text(encoding="utf-8"))
        cfg["suites"] = [s for s in cfg["suites"] if s["name"] in suites]
        gate_args[1] = str(_toml_from(cfg, args.out / "gates.partial.toml"))
    return int(release_gate.main(gate_args))


def _toml_from(cfg: dict[str, Any], path: Path) -> Path:
    """Write a minimal TOML for the subset (values are str, int, float, bool)."""
    def val(v: Any) -> str:
        return json.dumps(v) if not isinstance(v, bool) else ("true" if v else "false")

    lines = [f"name = {val(cfg['name'])}"]
    for s in cfg["suites"]:
        lines += ["", "[[suites]]", f"name = {val(s['name'])}", "", "[suites.gate]"]
        gate = s.get("gate", {})
        lines += [f"{k} = {val(v)}" for k, v in gate.items() if not isinstance(v, list)]
        for key in ("metrics", "critical", "slices"):
            for rule in gate.get(key, []):
                lines += ["", f"[[suites.gate.{key}]]", *[f"{k} = {val(v)}" for k, v in rule.items()]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    sys.exit(main())
