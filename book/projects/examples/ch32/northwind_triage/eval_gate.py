# path: book/projects/examples/ch32/northwind_triage/eval_gate.py
"""Offline evaluation gate for CI: run the golden set, compare with the stored baseline,
exit non-zero on regression.

    python -m northwind_triage.eval_gate --prompt-version 1.1.0 --report eval/report.json
    python -m northwind_triage.eval_gate --prompt-version 1.1.0 --write-baseline

Exit codes: 0 pass, 1 regression, 2 baseline not comparable (different dataset).
This is a package module with an entry point, not a file in scripts/: it imports the same
composition root as production, so the gate evaluates the code that ships. Chapter 25 owns
the general evaluation-gate design; this is the triage-specific instance.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aie_core.llm.client import LLMClient
from aie_core.observability import InMemoryTracer

from .composition import build_service
from .config import load_settings
from .domain import Category, Priority, Ticket
from .version_manifest import VersionManifest, content_hash, versioned

HERE = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = HERE / "eval" / "golden.jsonl"
DEFAULT_BASELINE = HERE / "eval" / "baseline.json"
EVALUATOR_VERSION = "1.0.0"  # bump when metric definitions below change


@dataclass(frozen=True)
class GoldenCase:
    ticket: Ticket
    category: Category
    priority: Priority


def load_dataset(path: Path) -> tuple[list[GoldenCase], str]:
    raw = path.read_bytes()
    cases = []
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cases.append(GoldenCase(
            ticket=Ticket(id=row["id"], tenant=row["tenant"], subject=row["subject"], body=row["body"],
                          requester_role=row.get("requester_role", "employee")),
            category=Category(row["category"]),
            priority=Priority(row["priority"]),
        ))
    return cases, versioned(f"{path.stem}-{len(cases)}", raw)


def evaluate(cases: list[GoldenCase], prompt_version: str, llm_client: LLMClient | None = None,
             dataset_version: str = "unknown") -> tuple[dict[str, float], VersionManifest]:
    # Isolated from the environment: a flag file or treatment version in the shell must not split
    # the golden set across prompt variants.
    settings = load_settings(environment="test", prompt_control_version=prompt_version,
                             flags_path=None, prompt_treatment_version=None, model_candidate=None,
                             _env_file=None, dataset_version=dataset_version,
                             evaluator_version=versioned(EVALUATOR_VERSION, Path(__file__).read_bytes()))
    service = build_service(settings, llm_client=llm_client, tracer=InMemoryTracer())
    correct_cat = correct_pri = errors = 0
    per_cat_hits: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    p1_hits = p1_total = 0
    manifest: VersionManifest | None = None
    for case in cases:
        result = service.triage(case.ticket, unit_id=f"eval:{case.ticket.id}")
        if manifest is not None and result.manifest.fingerprint() != manifest.fingerprint():
            raise RuntimeError("eval cases ran under different version manifests; results are not attributable")
        manifest = manifest or result.manifest
        got = result.triage
        if result.outcome != "ok":
            errors += 1
        hit = got.category == case.category
        correct_cat += hit
        correct_pri += got.priority == case.priority
        per_cat_hits[case.category.value][0] += hit
        per_cat_hits[case.category.value][1] += 1
        if case.priority == Priority.P1:
            p1_total += 1
            p1_hits += got.priority == Priority.P1
    n = len(cases)
    metrics: dict[str, float] = {
        "n": float(n),
        "category_accuracy": correct_cat / n,
        "priority_accuracy": correct_pri / n,
        "error_rate": errors / n,
        "p1_recall": p1_hits / p1_total if p1_total else 1.0,
    }
    for cat, (hits, total) in sorted(per_cat_hits.items()):
        metrics[f"recall.{cat}"] = hits / total
    assert manifest is not None
    return metrics, manifest


@dataclass(frozen=True)
class GateRules:
    min_category_accuracy: float = 0.70
    max_regression: float = 0.02          # absolute drop allowed on aggregate metrics
    critical_slices: tuple[str, ...] = ("recall.security_report", "p1_recall")
    max_error_rate: float = 0.02


def gate(current: dict[str, float], baseline: dict[str, float] | None, rules: GateRules) -> list[str]:
    failures: list[str] = []
    if current["category_accuracy"] < rules.min_category_accuracy:
        failures.append(f"category_accuracy {current['category_accuracy']:.3f} < floor {rules.min_category_accuracy}")
    if current["error_rate"] > rules.max_error_rate:
        failures.append(f"error_rate {current['error_rate']:.3f} > {rules.max_error_rate}")
    if baseline:
        for metric in ("category_accuracy", "priority_accuracy"):
            if current[metric] < baseline[metric] - rules.max_regression:
                failures.append(f"{metric} regressed {baseline[metric]:.3f} -> {current[metric]:.3f}")
        # Critical slices may not regress at all: an average can hide a security miss.
        for metric in rules.critical_slices:
            if metric in baseline and current.get(metric, 0.0) < baseline[metric]:
                failures.append(f"critical slice {metric} regressed {baseline[metric]:.3f} -> "
                                f"{current.get(metric, 0.0):.3f}")
    return failures


def run(argv: list[str] | None = None, llm_client: LLMClient | None = None) -> int:
    parser = argparse.ArgumentParser(prog="northwind_triage.eval_gate")
    parser.add_argument("--prompt-version", default="1.0.0")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--baseline", type=Path, default=None,
                        help="default: eval/baseline.json for the fake provider, eval/baseline-<provider>.json otherwise")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--strict-attribution", action="store_true",
                        help="fail when more than one component changed versus the baseline")
    args = parser.parse_args(argv)

    cases, dataset_version = load_dataset(args.dataset)
    metrics, manifest = evaluate(cases, args.prompt_version, llm_client, dataset_version)
    provider = manifest.models.get("provider", "fake")
    if args.baseline is None:   # one baseline per provider: a real model is never judged against the fake one
        args.baseline = DEFAULT_BASELINE if provider == "fake" else DEFAULT_BASELINE.with_name(f"baseline-{provider}.json")
    baseline_doc: dict[str, Any] | None = (json.loads(args.baseline.read_text())
                                           if args.baseline.exists() else None)
    report: dict[str, Any] = {"metrics": metrics, "manifest": manifest.model_dump(mode="json"),
                              "fingerprint": manifest.fingerprint(), "failures": [], "warnings": []}

    if args.write_baseline:
        args.baseline.write_text(json.dumps({"dataset": dataset_version, "metrics": metrics,
                                             "manifest": manifest.model_dump(mode="json")},
                                            indent=2, sort_keys=True) + "\n")
        print(f"baseline written: {args.baseline}")
        return 0

    exit_code = 0
    base_provider = (baseline_doc or {}).get("manifest", {}).get("models", {}).get("provider")
    if baseline_doc is None:
        # Fail closed: without a baseline the regression and critical-slice checks would be skipped.
        report["failures"].append(f"no baseline at {args.baseline}; create one with --write-baseline in a reviewed change")
        exit_code = 2
    elif base_provider is not None and base_provider != provider:
        report["failures"].append(f"baseline made with provider {base_provider!r}, current is {provider!r}; "
                                  f"keep one baseline per provider")
        exit_code = 2
    elif baseline_doc["dataset"] != dataset_version:
        report["failures"].append(f"baseline dataset {baseline_doc['dataset']} != {dataset_version}; "
                                  f"re-baseline on the new dataset before comparing")
        exit_code = 2
    else:
        baseline_metrics = baseline_doc["metrics"] if baseline_doc else None
        report["failures"] = gate(metrics, baseline_metrics, GateRules())
        if baseline_doc:
            changed = manifest.changed_components(VersionManifest(**baseline_doc["manifest"]))
            changed.discard("config")  # flag snapshot hash etc.
            if len(changed) > 1:
                msg = f"{len(changed)} components changed at once ({sorted(changed)}); attribution is lost"
                (report["failures"] if args.strict_attribution else report["warnings"]).append(msg)
        exit_code = 1 if report["failures"] else 0

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for key in ("category_accuracy", "priority_accuracy", "p1_recall", "recall.security_report", "error_rate"):
        print(f"{key:28s} {metrics.get(key, float('nan')):.3f}")
    for w in report["warnings"]:
        print(f"WARNING: {w}")
    for f in report["failures"]:
        print(f"FAIL: {f}")
    print("GATE:", "PASS" if exit_code == 0 else "FAIL")
    return exit_code


def main() -> None:  # console-script entry point
    sys.exit(run())


if __name__ == "__main__":  # pragma: no cover
    main()
