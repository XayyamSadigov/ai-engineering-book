# path: book/projects/p1-extraction-api/extraction_api/domain/invoice.py
"""Invoice schemas: the *wire* schema the model fills, and the *domain* record code trusts.

The two are deliberately different. The wire schema is lenient where models drift (amounts
may arrive as "1,488.00", dates are copied as written) and strict where a closed set helps
the model (evidence field names). The domain record is strict everywhere: Decimal money,
real dates, an ISO currency enum. `build_invoice` is the only bridge between them.
"""
from __future__ import annotations

import re
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .common import EvidenceSpan, RuleViolation
from .normalize import (
    NormalizationError, clean_identifier, clean_text, locate, parse_currency, parse_date,
    parse_money, parse_quantity, parse_rate, parse_unit_price,
)


class Currency(str, Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    CAD = "CAD"


InvoiceFieldName = Literal[
    "vendor", "invoice_number", "invoice_date", "due_date", "po_number", "currency",
    "subtotal", "tax_rate", "tax_amount", "total",
]
CRITICAL_INVOICE_FIELDS: tuple[str, ...] = ("vendor", "invoice_number", "invoice_date", "currency", "total")
Number = float | str  # models emit 1488.0 or "1,488.00"; normalization handles both


# ------------------------------------------------------------------ wire schema
class LineItemDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str = Field(description="Line description exactly as written.")
    quantity: Number | None = Field(description="Quantity as a number, null if not stated.")
    unit_price: Number | None = Field(description="Unit price as a number without currency symbol, null if not stated.")
    amount: Number = Field(description="Line amount as a number without currency symbol.")


class InvoiceEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: InvoiceFieldName
    quote: str = Field(max_length=300, description="Shortest verbatim excerpt of the document that contains the value.")
    confidence: float = Field(ge=0.0, le=1.0, description="How sure you are that the value is correct, 0 to 1.")


class InvoiceDraft(BaseModel):
    """Every field is required but nullable: the model must consider each one and say null
    explicitly rather than silently omitting it. This shape also satisfies the strict
    structured-output modes that reject optional properties."""

    model_config = ConfigDict(extra="forbid")
    vendor: str | None = Field(description="Legal name of the issuing company.")
    invoice_number: str | None = Field(description="Invoice or statement number exactly as printed.")
    invoice_date: str | None = Field(description="Issue date copied exactly as written. Do not reformat.")
    due_date: str | None = Field(description="Payment due date copied exactly as written, null if absent.")
    po_number: str | None = Field(description="Purchase order number, null if absent or marked not provided.")
    currency: str | None = Field(description="Currency code or symbol as shown, for example USD or EUR.")
    line_items: list[LineItemDraft] = Field(description="Every billed line, in document order.")
    subtotal: Number | None = Field(description="Subtotal before tax as stated, null if not stated.")
    tax_rate: Number | None = Field(description="Tax rate as stated, for example 8% or 0.08, null if not stated.")
    tax_amount: Number | None = Field(description="Tax amount as stated, null if not stated.")
    total: Number | None = Field(description="Total amount due as stated on the document.")
    evidence: list[InvoiceEvidence] = Field(description="One entry for each non-null scalar field above.")


# ---------------------------------------------------------------- domain record
class LineItem(BaseModel):
    description: str = Field(min_length=1)
    quantity: Decimal | None = None
    unit_price: Decimal | None = None
    amount: Decimal


class Invoice(BaseModel):
    """What downstream finance code receives. If this object exists, its types are real."""

    vendor: str = Field(min_length=1)
    invoice_number: str = Field(min_length=1)
    invoice_date: date
    due_date: date | None = None
    po_number: str | None = None
    currency: Currency
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: Decimal | None = None
    tax_rate: Decimal | None = None
    tax_amount: Decimal | None = None
    total: Decimal
    evidence: list[EvidenceSpan] = Field(default_factory=list)


# ------------------------------------------------------------------- the bridge
_NUMBER_IN_QUOTE = re.compile(r"\d[\d,.]*\d|\d")
_MONEY_FIELDS = ("subtotal", "tax_amount", "total")
_TEXT_FIELDS = ("vendor", "invoice_number", "po_number", "invoice_date", "due_date")


def _squash(s: str) -> str:
    return " ".join(s.split()).casefold()


def value_supported_by_quote(field: str, raw: Any, quote: str) -> bool:
    """A quote that exists in the document but does not contain the value proves nothing.
    Money is compared numerically (the quote may print $3,327.48 for 3327.48); text and
    as-written dates must appear inside the quote."""
    if raw is None:
        return True
    if field in _MONEY_FIELDS:
        try:
            target = parse_money(raw)
        except NormalizationError:
            return False
        for token in _NUMBER_IN_QUOTE.findall(quote):
            try:
                if parse_money(token) == target:
                    return True
            except NormalizationError:
                continue
        return False
    if field in _TEXT_FIELDS:
        return _squash(str(raw)) in _squash(quote)
    return True  # currency and tax_rate: symbols and percent forms vary too much to check this way


def _norm(field: str, fn: Any, raw: Any, violations: list[RuleViolation], **kw: Any) -> Any:
    try:
        return fn(raw, **kw)
    except NormalizationError as exc:
        violations.append(RuleViolation(
            code=exc.code, field=field, detail=exc.detail,
            # an ambiguous date is a property of the document; re-asking cannot fix it
            repairable=exc.code != "AMBIGUOUS_DATE",
        ))
        return None


def build_invoice(
    draft: InvoiceDraft, text: str, *, dollar_means: str = "USD"
) -> tuple[dict[str, Any], Invoice | None, list[RuleViolation]]:
    """Normalize a draft against the source text.

    Returns the normalized data as a plain dict (always, so a reviewer sees partial work),
    the typed Invoice if every required field survived, and the violations found on the way.
    """
    v: list[RuleViolation] = []
    data: dict[str, Any] = {
        "vendor": clean_text(draft.vendor),
        "invoice_number": clean_identifier(draft.invoice_number),
        "invoice_date": _norm("invoice_date", parse_date, draft.invoice_date, v),
        "due_date": _norm("due_date", parse_date, draft.due_date, v),
        "po_number": clean_identifier(draft.po_number),
        "currency": _norm("currency", parse_currency, draft.currency, v, dollar_means=dollar_means),
        "subtotal": _norm("subtotal", parse_money, draft.subtotal, v),
        "tax_rate": _norm("tax_rate", parse_rate, draft.tax_rate, v),
        "tax_amount": _norm("tax_amount", parse_money, draft.tax_amount, v),
        "total": _norm("total", parse_money, draft.total, v),
        "line_items": [],
    }
    if data["currency"] is not None and data["currency"] not in Currency.__members__:
        v.append(RuleViolation(code="UNSUPPORTED_CURRENCY", field="currency",
                               detail=f"{data['currency']} is not an accepted currency", repairable=False))
        data["currency"] = None
    for i, item in enumerate(draft.line_items):
        data["line_items"].append({
            "description": clean_text(item.description) or "",
            "quantity": _norm(f"line_items.{i}.quantity", parse_quantity, item.quantity, v),
            "unit_price": _norm(f"line_items.{i}.unit_price", parse_unit_price, item.unit_price, v),
            "amount": _norm(f"line_items.{i}.amount", parse_money, item.amount, v),
        })

    spans: list[EvidenceSpan] = []
    for ev in draft.evidence:
        if not value_supported_by_quote(ev.field, getattr(draft, ev.field), ev.quote):
            v.append(RuleViolation(code="EVIDENCE_VALUE_MISMATCH", field=ev.field, repairable=True,
                                   detail=f"value {getattr(draft, ev.field)!r} does not appear in its quote {ev.quote[:80]!r}"))
        pos = locate(ev.quote, text)
        spans.append(EvidenceSpan(
            field=ev.field, quote=ev.quote, model_confidence=ev.confidence,
            start=pos[0] if pos else None, end=pos[1] if pos else None,
        ))
    data["evidence"] = [s.model_dump() for s in spans]

    for name in CRITICAL_INVOICE_FIELDS:
        if data.get(name) is None and not any(x.field == name for x in v):
            v.append(RuleViolation(code="MISSING_FIELD", field=name,
                                   detail=f"required field {name} is missing", repairable=True))
    try:
        invoice = Invoice.model_validate(data)
    except ValidationError as exc:
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"])
            if not any(x.field == loc for x in v):
                v.append(RuleViolation(code="INVALID_FIELD", field=loc, detail=err["msg"], repairable=True))
        invoice = None
    return data, invoice, v


__all__ = [
    "Currency", "InvoiceFieldName", "CRITICAL_INVOICE_FIELDS", "LineItemDraft", "InvoiceEvidence",
    "InvoiceDraft", "LineItem", "Invoice", "build_invoice", "value_supported_by_quote",
]
