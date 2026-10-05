# path: book/projects/p1-extraction-api/extraction_api/eval/run_eval.py
"""Run the full pipeline over gold invoices and report field metrics, routing quality, cost.

    python -m extraction_api.eval.run_eval                  # replay model, shared-data gold
    LLM_PROVIDER=openai LLM_MODEL=... python -m extraction_api.eval.run_eval --out report.json

Routing is judged against two kinds of truth: was the extraction right (field gold), and
was the document itself consistent (the gold `validation` block). An inconsistent invoice
routed to review is a *correct* review, even if every field was extracted perfectly.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from typing import Any

from pydantic import BaseModel, computed_field

from ..application import DocumentIn, ExtractionService
from .calibration import ThresholdChoice, choose_threshold, expected_calibration_error
from .dataset import GoldInvoice, default_gold_path, load_invoice_gold
from .metrics import InvoiceEvalReport, score_invoices


class RoutingReport(BaseModel):
    total: int
    accepted: int
    reviewed: int
    errors: int
    false_accepts: int          # accepted although a critical field was wrong or the document inconsistent
    unnecessary_reviews: int    # reviewed although everything was right and the document consistent
    llm_calls: int
    input_tokens: int
    output_tokens: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def false_accept_rate(self) -> float:
        return self.false_accepts / self.accepted if self.accepted else 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def review_rate(self) -> float:
        return self.reviewed / self.total if self.total else 0.0


class EvalRun(BaseModel):
    gold_path: str
    report: InvoiceEvalReport
    routing: RoutingReport
    by_format: dict[str, float]
    threshold: ThresholdChoice | None
    ece: float
    routes: dict[str, str]


async def evaluate(service: ExtractionService, gold: list[GoldInvoice], target_precision: float = 0.98,
                   gold_path: str = "") -> EvalRun:
    docs = [DocumentIn(document_id=g.id, text=g.text) for g in gold]  # no type hint: classification is measured too
    items = await service.extract_batch(docs, request_id="eval")
    pairs: list[tuple[str, dict[str, Any] | None, dict[str, Any]]] = []
    for g, item in zip(gold, items):
        pairs.append((g.id, item.result.data if item.result else None, g.expected))
    report = score_invoices(pairs)

    accepted = reviewed = errors = false_acc = unnecessary = calls = tin = tout = 0
    scores: list[float] = []
    correct: list[bool] = []
    by_format: dict[str, list[bool]] = defaultdict(list)
    routes: dict[str, str] = {}
    for g, item, doc in zip(gold, items, report.docs):
        by_format[g.format].append(doc.critical_correct)
        if item.result is None:
            errors += 1
            routes[g.id] = "error"
            continue
        r = item.result
        routes[g.id] = r.route.value
        calls += r.llm_calls
        tin += r.usage.input_tokens
        tout += r.usage.output_tokens
        truly_ok = doc.critical_correct and g.document_is_consistent
        if not any(v.severity.value == "error" for v in r.violations):
            # the threshold only ever decides rule-clean documents, so calibrate on those
            scores.append(r.score)
            correct.append(truly_ok)
        if r.route.value == "accept":
            accepted += 1
            false_acc += not truly_ok
        else:
            reviewed += 1
            unnecessary += truly_ok

    routing = RoutingReport(total=len(gold), accepted=accepted, reviewed=reviewed, errors=errors,
                            false_accepts=false_acc, unnecessary_reviews=unnecessary,
                            llm_calls=calls, input_tokens=tin, output_tokens=tout)
    return EvalRun(
        gold_path=gold_path, report=report, routing=routing,
        by_format={k: sum(v) / len(v) for k, v in sorted(by_format.items())},
        threshold=choose_threshold(scores, correct, target_precision) if scores else None,
        ece=expected_calibration_error(scores, correct) if scores else 0.0,
        routes=routes,
    )


def format_run(run: EvalRun) -> str:
    lines = [f"gold: {run.gold_path}", "",
             f"{'field':<16}{'P':>7}{'R':>7}{'F1':>7}{'exact':>8}"]
    for fs in [*run.report.fields.values(), run.report.line_items]:
        lines.append(f"{fs.field:<16}{fs.precision:>7.3f}{fs.recall:>7.3f}{fs.f1:>7.3f}{fs.exact_match:>8.3f}")
    rt = run.routing
    lines += [
        "",
        f"critical-field accuracy  {run.report.critical_accuracy:.3f}   weighted {run.report.mean_weighted:.3f}",
        f"by format                {', '.join(f'{k}={v:.2f}' for k, v in run.by_format.items())}",
        f"routes                   accept={rt.accepted} review={rt.reviewed} error={rt.errors}",
        f"false accepts            {rt.false_accepts} ({rt.false_accept_rate:.1%} of accepted)",
        f"unnecessary reviews      {rt.unnecessary_reviews}",
        f"cost                     {rt.llm_calls} calls, {rt.input_tokens} in / {rt.output_tokens} out tokens",
        f"calibration              ECE={run.ece:.3f}  threshold={run.threshold.model_dump() if run.threshold else None}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Project 1 extraction against gold invoices")
    parser.add_argument("--gold", default=None, help="JSONL gold file (default: shared-data or fixture)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--target-precision", type=float, default=0.98)
    parser.add_argument("--out", default=None, help="write the full report as JSON")
    args = parser.parse_args(argv)

    from ..wiring import build_service  # imported here so tests can import evaluate() cheaply

    gold_path = str(args.gold or default_gold_path())
    gold = load_invoice_gold(gold_path, args.limit)
    run = asyncio.run(evaluate(build_service(), gold, args.target_precision, gold_path))
    print(format_run(run))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(run.model_dump(mode="json"), f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
