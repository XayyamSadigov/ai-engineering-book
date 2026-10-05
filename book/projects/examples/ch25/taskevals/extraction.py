# path: book/projects/examples/ch25/taskevals/extraction.py
"""Extraction evaluators: field-level P/R with critical-field weighting, line items, evidence.

The extractor's output contract (Chapter 6 owns the pipeline):

    {"fields": {"vendor": "...", "total": 3327.48, ...},
     "line_items": [{"description": "...", "quantity": 2400, "unit_price": 0.62, "amount": 1488.0}],
     "evidence": {"total": "TOTAL DUE     $3,327.48", ...}}

`evidence` maps a field to a verbatim quote from the document. An evidence location is correct
when the quote is found in the document AND the quote contains the gold value in one of its
usual renderings. Evidence is what lets a reviewer verify a field in seconds; evidence that
points at the wrong line is worse than none, because it makes a wrong value look checked.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from evalkit import EvalCase, Score
from evalkit.metrics import field_prf, numeric_close, prf_from_counts

# Silent errors in these fields reach finance systems; weights are illustrative and owned by the business.
FIELD_WEIGHTS: dict[str, float] = {
    "invoice_number": 3.0,
    "total": 3.0,
    "vendor": 2.0,
    "currency": 2.0,
    "due_date": 2.0,
    "po_number": 1.0,
    "invoice_date": 1.0,
    "subtotal": 1.0,
    "tax_amount": 1.0,
}
CRITICAL_FIELDS: tuple[str, ...] = ("invoice_number", "total", "currency", "vendor")
NUMERIC_TOL = 0.005


def weighted_field_prf(per_field: Mapping[str, str], weights: Mapping[str, float]) -> tuple[float, float, float]:
    """Weighted P/R/F1 from evalkit's per-field outcomes (tp, fp, fn, fp+fn, tn)."""
    tp = fp = fn = 0.0
    for name, outcome in per_field.items():
        w = weights.get(name, 1.0)
        if outcome == "tp":
            tp += w
        elif outcome == "fp":
            fp += w
        elif outcome == "fn":
            fn += w
        elif outcome == "fp+fn":
            fp += w
            fn += w
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def value_renderings(value: Any) -> list[str]:
    """Ways a gold value is commonly written in a document, lowercased."""
    if value is None:
        return []
    if isinstance(value, date):
        value = value.isoformat()
    if isinstance(value, bool):
        return [str(value).lower()]
    if isinstance(value, (int, float)):
        v = float(value)
        out = {f"{v:,.2f}", f"{v:.2f}"}
        if v.is_integer():
            out |= {f"{int(v):,}", str(int(v))}
        return sorted(out)
    s = str(value).strip().lower()
    out = {s}
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        d = date.fromisoformat(s)
        out |= {d.strftime("%B %-d, %Y").lower(), d.strftime("%-d %B %Y").lower(), d.strftime("%d/%m/%Y")}
    return sorted(out)


def locate_value(text: str, value: Any) -> str | None:
    """The first document line that contains a rendering of `value` (used by stand-in extractors)."""
    renders = value_renderings(value)
    for line in text.splitlines():
        low = line.lower()
        if any(r and r in low for r in renders):
            return line.strip()
    return None


def evidence_status(document: str, quote: str | None, gold_value: Any) -> str:
    """correct | wrong_location (quote exists, does not support the value) | not_found | missing."""
    if not quote:
        return "missing"
    q = _norm_text(quote)
    if q not in _norm_text(document):
        return "not_found"  # fabricated or paraphrased quote: cannot be verified
    return "correct" if any(r in q for r in value_renderings(gold_value)) else "wrong_location"


def line_item_prf(predicted: Sequence[Mapping[str, Any]], gold: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    """Greedy one-to-one match on amount (within tolerance) and normalized description prefix."""
    unmatched = list(gold)
    tp = 0
    for p in predicted:
        for g in unmatched:
            same_amount = numeric_close(p.get("amount"), g.get("amount"), abs_tol=NUMERIC_TOL)
            same_desc = _norm_text(str(p.get("description", "")))[:20] == _norm_text(str(g.get("description", "")))[:20]
            if same_amount and same_desc:
                tp += 1
                unmatched.remove(g)
                break
    prf = prf_from_counts(tp, len(predicted) - tp, len(gold) - tp)
    return prf.precision, prf.recall


EXTRACTION_METRICS = ["field_precision", "field_recall", "field_f1_weighted", "critical_fields_exact",
                      "line_item_recall", "evidence_correct"]


class ExtractionEvaluator:
    """evalkit evaluator for invoice extraction. `case.input["text"]` is the document,
    `case.expected` is the gold record from shared-data/invoices.jsonl."""

    name = "extraction"
    version = "1"
    metric_names = EXTRACTION_METRICS

    def __init__(self, weights: Mapping[str, float] = FIELD_WEIGHTS,
                 critical: Sequence[str] = CRITICAL_FIELDS, evidence_threshold: float = 0.9) -> None:
        self.weights = dict(weights)
        self.critical = tuple(critical)
        self.evidence_threshold = evidence_threshold

    def __call__(self, case: EvalCase, output: Any) -> list[Score]:
        gold = {k: case.expected.get(k) for k in self.weights}
        pred_fields = output.get("fields", {})
        pred = {k: pred_fields.get(k) for k in self.weights}
        fs = field_prf(pred, gold, fields=list(self.weights), numeric_tol=NUMERIC_TOL)
        wp, wr, wf = weighted_field_prf(fs.per_field, self.weights)
        bad_critical = [f for f in self.critical if fs.per_field.get(f) not in ("tp", "tn")]
        _, li_recall = line_item_prf(output.get("line_items", []), case.expected.get("line_items", []))

        statuses = {
            f: evidence_status(case.input["text"], output.get("evidence", {}).get(f), gold[f])
            for f in self.weights if gold[f] not in (None, "") and fs.per_field.get(f) == "tp"
        }
        ev_rate = sum(1 for s in statuses.values() if s == "correct") / len(statuses) if statuses else 1.0
        ev_bad = {f: s for f, s in statuses.items() if s != "correct"}
        return [
            Score(name="field_precision", value=wp, passed=None),
            Score(name="field_recall", value=wr, passed=None),
            Score(name="field_f1_weighted", value=wf, passed=wf >= 0.9,
                  detail={f: o for f, o in fs.per_field.items() if o not in ("tp", "tn")} or None),
            Score(name="critical_fields_exact", value=0.0 if bad_critical else 1.0, passed=not bad_critical,
                  detail=bad_critical or None),
            Score(name="line_item_recall", value=li_recall, passed=li_recall >= 1.0),
            Score(name="evidence_correct", value=ev_rate, passed=ev_rate >= self.evidence_threshold,
                  detail=ev_bad or None),
        ]


def per_field_report(run_details: Sequence[Mapping[str, str]], fields: Sequence[str]) -> dict[str, dict[str, int]]:
    """Aggregate per-field outcome counts across cases (input: the field_f1_weighted details)."""
    out = {f: {"fp": 0, "fn": 0, "fp+fn": 0} for f in fields}
    for detail in run_details:
        for f, outcome in (detail or {}).items():
            if f in out and outcome in out[f]:
                out[f][outcome] += 1
    return out


__all__ = ["FIELD_WEIGHTS", "CRITICAL_FIELDS", "weighted_field_prf", "value_renderings", "locate_value",
           "evidence_status", "line_item_prf", "ExtractionEvaluator", "EXTRACTION_METRICS", "per_field_report"]
