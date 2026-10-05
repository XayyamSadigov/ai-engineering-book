# path: book/projects/p1-extraction-api/extraction_api/eval/metrics.py
"""Field-level evaluation for invoice extraction.

Document-level "success" hides which field is failing. Each field gets its own counts:
  TP  gold has a value and the prediction matches it
  FP  the prediction has a value that is wrong, or gold has no value at all
  FN  gold has a value and the prediction missed it or got it wrong
A wrong value is both an FP and an FN: it asserted something false and missed the truth.
Exact match counts agreement including both-null, which is what the downstream system sees.
"""
from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import BaseModel, Field, computed_field

FIELD_WEIGHTS: dict[str, float] = {
    "total": 3.0, "invoice_number": 2.0, "vendor": 2.0, "invoice_date": 2.0, "currency": 2.0,
    "due_date": 1.0, "po_number": 1.0, "subtotal": 1.0, "tax_rate": 1.0, "tax_amount": 1.0,
}
CRITICAL = ("vendor", "invoice_number", "invoice_date", "currency", "total")


def _text(v: Any) -> str:
    return " ".join(str(v).split()).casefold().strip(" .,")


def _ident(v: Any) -> str:
    return "".join(str(v).split()).casefold()


def _num_eq(tol: str) -> Callable[[Any, Any], bool]:
    def eq(a: Any, b: Any) -> bool:
        try:
            return abs(Decimal(str(a)) - Decimal(str(b))) <= Decimal(tol)
        except InvalidOperation:
            return False
    return eq


COMPARATORS: dict[str, Callable[[Any, Any], bool]] = {
    "vendor": lambda a, b: _text(a) == _text(b),
    "invoice_number": lambda a, b: _ident(a) == _ident(b),
    "po_number": lambda a, b: _ident(a) == _ident(b),
    "invoice_date": lambda a, b: str(a)[:10] == str(b)[:10],
    "due_date": lambda a, b: str(a)[:10] == str(b)[:10],
    "currency": lambda a, b: str(a).upper() == str(b).upper(),
    "subtotal": _num_eq("0.005"),
    "tax_amount": _num_eq("0.005"),
    "total": _num_eq("0.005"),
    "tax_rate": _num_eq("0.0005"),
}


class FieldScore(BaseModel):
    field: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    correct: int = 0
    n: int = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 1.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def exact_match(self) -> float:
        return self.correct / self.n if self.n else 0.0


def field_correct(field: str, pred: Any, gold: Any) -> bool:
    if gold is None or pred is None:
        return gold is None and pred is None
    return COMPARATORS[field](pred, gold)


def score_field(fs: FieldScore, pred: Any, gold: Any) -> bool:
    fs.n += 1
    ok = field_correct(fs.field, pred, gold)
    if ok:
        fs.correct += 1
        if gold is not None:
            fs.tp += 1
        return True
    if pred is not None:
        fs.fp += 1
    if gold is not None:
        fs.fn += 1
    return False


def score_line_items(fs: FieldScore, pred: list[dict[str, Any]], gold: list[dict[str, Any]]) -> None:
    """Greedy match on amount; a matched line counts only if the quantity also agrees."""
    amount_eq, qty_eq = _num_eq("0.005"), _num_eq("0.0001")
    remaining = list(gold)
    fs.n += 1
    matched = 0
    for p in pred:
        for g in remaining:
            if amount_eq(p.get("amount"), g.get("amount")) and (
                p.get("quantity") is None or g.get("quantity") is None or qty_eq(p["quantity"], g["quantity"])
            ):
                remaining.remove(g)
                matched += 1
                break
    fs.tp += matched
    fs.fp += len(pred) - matched
    fs.fn += len(gold) - matched
    fs.correct += int(matched == len(pred) == len(gold))


class DocScore(BaseModel):
    id: str
    weighted: float
    critical_correct: bool
    wrong_fields: list[str] = Field(default_factory=list)


class InvoiceEvalReport(BaseModel):
    fields: dict[str, FieldScore]
    line_items: FieldScore
    docs: list[DocScore]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mean_weighted(self) -> float:
        return sum(d.weighted for d in self.docs) / len(self.docs) if self.docs else 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def critical_accuracy(self) -> float:
        return sum(d.critical_correct for d in self.docs) / len(self.docs) if self.docs else 0.0


def score_invoices(pairs: list[tuple[str, dict[str, Any] | None, dict[str, Any]]]) -> InvoiceEvalReport:
    """`pairs` are (doc id, predicted data or None, gold expected)."""
    fields = {f: FieldScore(field=f) for f in FIELD_WEIGHTS}
    lines = FieldScore(field="line_items")
    docs: list[DocScore] = []
    for doc_id, pred, gold in pairs:
        pred = pred or {}
        wrong: list[str] = []
        got = 0.0
        for f, w in FIELD_WEIGHTS.items():
            if score_field(fields[f], pred.get(f), gold.get(f)):
                got += w
            else:
                wrong.append(f)
        score_line_items(lines, pred.get("line_items") or [], gold.get("line_items") or [])
        docs.append(DocScore(id=doc_id, weighted=got / sum(FIELD_WEIGHTS.values()),
                             critical_correct=not any(f in CRITICAL for f in wrong), wrong_fields=wrong))
    return InvoiceEvalReport(fields=fields, line_items=lines, docs=docs)


__all__ = ["FieldScore", "DocScore", "InvoiceEvalReport", "score_invoices", "field_correct", "FIELD_WEIGHTS", "CRITICAL"]
