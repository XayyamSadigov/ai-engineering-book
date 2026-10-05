# path: book/projects/p3-rag-assistant/rag_assistant/eval/run_eval.py
"""CI evaluation gate for the assistant: the shared-data gold set against the real service code.

    rag-assistant-eval --out out/eval                     # offline (fake LLM, vocabulary embeddings)
    rag-assistant-eval --baseline-run main/candidate_run.json   # also apply regression rules
    LLM_PROVIDER=openai ... rag-assistant-eval            # same gate against a real provider

It builds a container exactly as the API does, ingests the docs folder through the queue and
the worker, then evaluates `AnswerService.ask` with ragkit.eval (Chapter 14): `evaluate_system`
over a `(question, principal) -> RagOutput` target, stage isolation, and `render_rag_report`.
Caches are disabled so the run measures the pipeline, not a warm cache.

Exit codes: 0 gate passed, 1 gate failed (any permission leak always fails), 2 setup error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from evalkit import GateConfig, Run, evaluate_gate
from ragkit.eval.rag_dataset import RagOutput, from_grounded_qa, load_gold_dataset
from ragkit.eval.rag_report import render_rag_report
from ragkit.eval.run_rag_eval import DEFAULT_GATE, evaluate_system, load_corpus
from ragkit.retrieval import Principal

from ..config import DEFAULT_GOLD_PATH, AssistantSettings
from ..wiring import Container, build_container

# Absolute floors, set from the offline reference run minus a margin (illustrative). Leaks and
# invalid citations are never averaged: one case fails the gate.
P3_GATE = GateConfig.from_dict({
    "name": "p3-release",
    "metrics": [
        {"metric": "no_permission_leak", "must_pass_all": True},
        {"metric": "citations_valid", "must_pass_all": True},
        {"metric": "recall@5", "min_mean": 0.85},
        {"metric": "hit@1", "min_mean": 0.70},
        {"metric": "evidence_packed", "min_mean": 0.80},
        {"metric": "abstention_correct", "min_mean": 0.80},
    ],
    "critical": [{"tag": "forbidden-doc", "metric": "no_permission_leak"},
                 {"tag": "conflicting-versions", "metric": "hit@1"}],
    "max_error_rate": 0.0,
    "max_evaluator_errors": 0,
})


def service_target(container: Container):  # type: ignore[no-untyped-def]
    """Adapt AnswerService to the ragkit.eval contract."""

    def system(question: str, principal: Principal) -> RagOutput:
        outcome = container.answers.ask(question, principal)
        r = outcome.response
        meta: dict[str, Any] = {"mode": r.mode, "degraded": r.degraded, "security": outcome.security_events}
        if outcome.qa is not None and outcome.retrieval is not None:
            out = from_grounded_qa(outcome.qa, outcome.retrieval)
            out.metadata.update(meta)
            return out
        # sources-only, blocked or unavailable: no generated answer, so it counts as an abstention
        return RagOutput(answer="", abstained=True, retrieval=outcome.retrieval, metadata=meta)

    return system


def build_eval_container(settings: AssistantSettings | None = None, **overrides: Any) -> Container:
    s = (settings or AssistantSettings()).model_copy(update={"answer_cache": False, "retrieval_cache": False})
    container = build_container(s, **overrides)
    sync = container.ingestion.sync("folder")
    if sync["errors"]:
        raise RuntimeError(f"ingestion errors: {sync['errors']}")
    container.drain()
    if container.status().dead_letters:
        raise RuntimeError("ingestion jobs dead-lettered; see the queue's dead letters")
    return container


def run(out_dir: Path, *, gold: Path = DEFAULT_GOLD_PATH, baseline_run: Path | None = None,
        container: Container | None = None, gate: GateConfig = P3_GATE) -> tuple[bool, dict[str, Any]]:
    container = container or build_eval_container()
    dataset = load_gold_dataset(gold)
    corpus = load_corpus(container.settings.docs_dir)
    outcome = evaluate_system(service_target(container), dataset, name="p3-rag-assistant", corpus=corpus,
                              extra_versions={"index": container.index_set.active_version,
                                              "reranker": container.settings.reranker,
                                              "tenancy": container.settings.tenancy_mode}, concurrency=1)
    result = evaluate_gate(gate, outcome.run)
    regression = None
    baseline = Run.load_json(baseline_run) if baseline_run else None
    if baseline is not None:
        regression = evaluate_gate(DEFAULT_GATE, outcome.run, baseline)
    passed = result.passed and (regression is None or regression.passed)
    report = render_rag_report(outcome.run, dataset, diagnoses=outcome.diagnoses, baseline=baseline, gate=result,
                               title="Project 3 release evaluation")
    if regression is not None:
        report += "\n\n## Regression gate vs baseline\n\n" + regression.to_markdown()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    outcome.run.save_json(out_dir / "candidate_run.json")
    summary = {
        "passed": passed,
        "metrics": {m: round(outcome.run.mean(m), 4) for m in
                    ("no_permission_leak", "recall@5", "hit@1", "evidence_packed", "abstention_correct",
                     "citations_valid") if m in outcome.run.metric_names()},
        "failures": [f"{f.name}: observed {f.observed}, threshold {f.threshold}" for f in result.failures]
                    + ([f"regression {f.name}: {f.observed} vs {f.threshold}" for f in regression.failures]
                       if regression else []),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return passed, summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out/eval")
    ap.add_argument("--gold", default=str(DEFAULT_GOLD_PATH))
    ap.add_argument("--baseline-run", default=None)
    args = ap.parse_args(argv)
    try:
        passed, summary = run(Path(args.out), gold=Path(args.gold),
                              baseline_run=Path(args.baseline_run) if args.baseline_run else None)
    except Exception as exc:  # setup failures are distinct from gate failures
        print(f"evaluation setup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2))
    print(f"gate {'PASS' if passed else 'FAIL'}; report: {Path(args.out) / 'report.md'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["P3_GATE", "build_eval_container", "main", "run", "service_target"]
