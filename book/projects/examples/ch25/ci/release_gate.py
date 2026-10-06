# path: book/projects/examples/ch25/ci/release_gate.py
"""Release gate for CI: evalkit Run JSON + gate config -> verdict, Markdown summary, exit code.

    python ci/release_gate.py --config ci/gates.toml --runs eval-out/runs \
        --baselines ci/baselines --out eval-out

Exit codes: 0 every suite passed; 1 at least one gate check failed; 2 the gate could not be
evaluated (missing or unreadable run, bad config). Both non-zero codes must block a merge;
the distinction tells the on-call person whether to read the report or fix the pipeline.

Outputs in --out: summary.md (also appended to $GITHUB_STEP_SUMMARY when set), gate.json
(machine-readable), reports/<suite>.md (full evalkit report per suite).

The config is evalkit's GateConfig per suite plus `aggregates`: run-level rules (macro-F1,
per-class recall, calibration) that per-case metric rules cannot express.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalkit import GateCheck, GateConfig, Run, evaluate_gate, render_report  # noqa: E402

from taskevals.classification import classification_aggregates  # noqa: E402
from taskevals.suites import LABELS  # noqa: E402

EXIT_PASS, EXIT_FAIL, EXIT_ERROR = 0, 1, 2

# Run-level metric providers. Each returns a lookup `metric name -> value`.
AGGREGATORS: dict[str, Callable[[Run], Callable[[str], float]]] = {
    "classification": lambda run: classification_aggregates(run, LABELS).metric,
}


class AggregateRule(BaseModel):
    metric: str
    min: float | None = None
    max: float | None = None


class SuiteGate(BaseModel):
    name: str
    run: str | None = None  # file name under --runs; default <name>.json
    aggregator: str | None = None
    gate: GateConfig = Field(default_factory=GateConfig)
    aggregates: list[AggregateRule] = Field(default_factory=list)


class GateFile(BaseModel):
    name: str = "release"
    suites: list[SuiteGate]


class SuiteVerdict(BaseModel):
    suite: str
    status: str  # pass | fail | error
    checks: list[GateCheck] = Field(default_factory=list)
    message: str = ""


def aggregate_checks(rules: list[AggregateRule], lookup: Callable[[str], float]) -> list[GateCheck]:
    checks = []
    for r in rules:
        value = lookup(r.metric)
        ok = not math.isnan(value) and (r.min is None or value >= r.min) and (r.max is None or value <= r.max)
        bound = " and ".join(x for x in [f">= {r.min}" if r.min is not None else "",
                                          f"<= {r.max}" if r.max is not None else ""] if x)
        checks.append(GateCheck(name=f"aggregate {r.metric}", passed=ok, observed=f"{value:.3f}", threshold=bound))
    return checks


def evaluate_suite(sg: SuiteGate, runs: Path, baselines: Path | None, reports: Path) -> SuiteVerdict:
    path = runs / (sg.run or f"{sg.name}.json")
    try:
        candidate = Run.load_json(path)
    except (OSError, ValueError, ValidationError) as exc:
        return SuiteVerdict(suite=sg.name, status="error", message=f"cannot read run {path}: {type(exc).__name__}")
    baseline = None
    if baselines is not None and (baselines / path.name).exists():
        try:
            baseline = Run.load_json(baselines / path.name)
        except (OSError, ValueError, ValidationError) as exc:   # an unreadable baseline is a pipeline error
            return SuiteVerdict(suite=sg.name, status="error",
                                message=f"cannot read baseline {baselines / path.name}: {type(exc).__name__}")
    result = evaluate_gate(sg.gate.model_copy(update={"name": sg.name}), candidate, baseline)
    checks = list(result.checks)
    if sg.aggregates:
        if sg.aggregator not in AGGREGATORS:
            return SuiteVerdict(suite=sg.name, status="error", message=f"unknown aggregator {sg.aggregator!r}")
        try:
            checks += aggregate_checks(sg.aggregates, AGGREGATORS[sg.aggregator](candidate))
        except (ValueError, AttributeError, KeyError) as exc:
            return SuiteVerdict(suite=sg.name, status="error", message=f"aggregates failed: {type(exc).__name__}: {exc}")
    result = result.model_copy(update={"checks": checks, "passed": all(c.passed for c in checks)})
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"{sg.name}.md").write_text(
        render_report(candidate, baseline=baseline, gate=result, title=f"Suite {sg.name}"), encoding="utf-8")
    return SuiteVerdict(suite=sg.name, status="pass" if result.passed else "fail", checks=checks)


def render_summary(name: str, verdicts: list[SuiteVerdict], context: dict[str, str]) -> str:
    overall = "PASS" if all(v.status == "pass" for v in verdicts) else (
        "ERROR" if any(v.status == "error" for v in verdicts) else "FAIL")
    lines = [f"## Release gate `{name}`: {overall}", ""]
    if context:
        lines += [" ".join(f"{k}=`{v}`" for k, v in context.items()), ""]
    lines += ["| suite | verdict | failed checks |", "|---|---|---|"]
    for v in verdicts:
        failed = [c for c in v.checks if not c.passed]
        cell = v.message or (", ".join(c.name for c in failed) if failed else "-")
        lines.append(f"| {v.suite} | {v.status.upper()} | {cell} |")
    failing = [(v.suite, c) for v in verdicts for c in v.checks if not c.passed]
    if failing:
        lines += ["", "### Failing checks", "", "| suite | check | observed | threshold | detail |", "|---|---|---|---|---|"]
        lines += [f"| {s} | {c.name} | {c.observed} | {c.threshold} | {c.detail} |" for s, c in failing]
    lines += ["", "Full per-suite reports are in the `reports/` folder of the eval artifact."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate the release gate over stored evalkit runs.")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--baselines", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("eval-out"))
    args = ap.parse_args(argv)
    try:
        with args.config.open("rb") as f:
            cfg = GateFile.model_validate(tomllib.load(f))
    except (OSError, tomllib.TOMLDecodeError, ValidationError) as exc:
        print(f"release gate: bad config {args.config}: {exc}", file=sys.stderr)
        return EXIT_ERROR

    verdicts = [evaluate_suite(sg, args.runs, args.baselines, args.out / "reports") for sg in cfg.suites]
    context = {k: v for k, v in {"commit": os.environ.get("CI_COMMIT_SHA") or os.environ.get("GITHUB_SHA", ""),
                                  "pipeline": os.environ.get("CI_PIPELINE_ID") or os.environ.get("GITHUB_RUN_ID", "")}.items() if v}
    summary = render_summary(cfg.name, verdicts, context)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.md").write_text(summary, encoding="utf-8")
    (args.out / "gate.json").write_text(json.dumps([v.model_dump() for v in verdicts], indent=2), encoding="utf-8")
    if step_summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(step_summary, "a", encoding="utf-8") as f:
            f.write(summary)
    print(summary)
    if any(v.status == "error" for v in verdicts):
        return EXIT_ERROR
    return EXIT_PASS if all(v.status == "pass" for v in verdicts) else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
