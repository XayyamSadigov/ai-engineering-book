# path: book/projects/p1-extraction-api/extraction_api/domain/rules.py
"""Business rules: the checks a schema cannot express.

A schema says "total is a decimal". A rule says "total equals subtotal plus tax", "the
due date is not before the issue date", "Northwind pays nothing without a PO". Each rule is
a pure function of the record, so every one has a unit test and none needs a model.
"""
from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from .common import EvidenceSpan, RuleViolation, Severity
from .invoice import Invoice
from .ticket import Priority, SupportTicket, TicketCategory

MONEY_TOLERANCE = Decimal("0.01")


def _close(a: Decimal, b: Decimal, tol: Decimal = MONEY_TOLERANCE) -> bool:
    return abs(a - b) <= tol


def check_invoice(
    inv: Invoice,
    *,
    today: date,
    require_po: bool = True,
    max_age_days: int = 730,
) -> list[RuleViolation]:
    out: list[RuleViolation] = []

    for i, li in enumerate(inv.line_items):
        if li.quantity is not None and li.unit_price is not None:
            # a unit price printed with d decimals may be off by half a unit in the last
            # place, and that error is multiplied by the quantity
            expected = li.quantity * li.unit_price
            exponent = li.unit_price.as_tuple().exponent
            half_ulp = Decimal(5).scaleb(exponent - 1) if isinstance(exponent, int) else Decimal("0.005")
            tol = max(MONEY_TOLERANCE, abs(li.quantity) * half_ulp)
            if not _close(expected, li.amount, tol):
                out.append(RuleViolation(
                    code="LINE_AMOUNT_MISMATCH", field=f"line_items.{i}.amount", repairable=True,
                    detail=f"{li.quantity} x {li.unit_price} = {expected:.2f}, line says {li.amount}",
                ))

    line_sum = sum((li.amount for li in inv.line_items), Decimal("0"))
    if inv.line_items and inv.subtotal is not None and not _close(line_sum, inv.subtotal):
        out.append(RuleViolation(code="LINE_SUM_MISMATCH", field="subtotal", repairable=True,
                                 detail=f"sum of lines {line_sum:.2f} != subtotal {inv.subtotal}"))

    base = inv.subtotal if inv.subtotal is not None else (line_sum if inv.line_items else None)
    if base is not None:
        tax = inv.tax_amount or Decimal("0")
        if not _close(base + tax, inv.total):
            out.append(RuleViolation(code="TOTAL_MISMATCH", field="total", repairable=True,
                                     detail=f"subtotal + tax = {base + tax:.2f}, total says {inv.total}"))
        if inv.tax_rate is not None and inv.tax_amount is not None:
            if not _close(base * inv.tax_rate, inv.tax_amount, Decimal("0.05")):
                out.append(RuleViolation(code="TAX_RATE_MISMATCH", field="tax_amount", severity=Severity.WARNING,
                                         detail=f"{base} x {inv.tax_rate} != {inv.tax_amount}"))

    if inv.due_date is not None and inv.due_date < inv.invoice_date:
        out.append(RuleViolation(code="DUE_BEFORE_ISSUE", field="due_date", repairable=True,
                                 detail=f"due {inv.due_date} is before issue {inv.invoice_date}"))
    if inv.invoice_date > today + timedelta(days=1):
        out.append(RuleViolation(code="DATE_IN_FUTURE", field="invoice_date", repairable=True,
                                 detail=f"issue date {inv.invoice_date} is after today {today}"))
    if inv.invoice_date < today - timedelta(days=max_age_days):
        out.append(RuleViolation(code="DATE_TOO_OLD", field="invoice_date", severity=Severity.WARNING,
                                 detail=f"issue date {inv.invoice_date} is older than {max_age_days} days"))
    if inv.total <= 0:
        out.append(RuleViolation(code="NON_POSITIVE_TOTAL", field="total",
                                 detail="credit notes are handled by a different flow"))
    if require_po and not inv.po_number:
        out.append(RuleViolation(code="MISSING_PO", field="po_number", repairable=True,
                                 detail="Northwind requires a purchase order number"))
    return out


def check_ticket(t: SupportTicket) -> list[RuleViolation]:
    out: list[RuleViolation] = []
    if t.category is TicketCategory.OTHER:
        out.append(RuleViolation(code="CATEGORY_ABSTAINED", field="category",
                                 detail="model chose other; a person assigns the queue"))
    if t.priority is Priority.P1 and not any(e.field == "priority" and e.found for e in t.evidence):
        out.append(RuleViolation(code="P1_WITHOUT_EVIDENCE", field="priority", repairable=True,
                                 detail="P1 pages on-call staff; it needs a quoted reason"))
    return out


def check_evidence(
    spans: list[EvidenceSpan], present_fields: set[str], required: tuple[str, ...]
) -> list[RuleViolation]:
    """Every present critical field needs a quote, and the quote must exist in the text."""
    out: list[RuleViolation] = []
    by_field = {s.field: s for s in spans}
    for name in required:
        if name not in present_fields:
            continue
        span = by_field.get(name)
        if span is None:
            out.append(RuleViolation(code="MISSING_EVIDENCE", field=name, repairable=True,
                                     detail=f"no quote supports {name}"))
        elif not span.found:
            out.append(RuleViolation(code="EVIDENCE_NOT_FOUND", field=name, repairable=True,
                                     detail=f"quote for {name} does not occur in the document: {span.quote[:80]!r}"))
    return out


__all__ = ["check_invoice", "check_ticket", "check_evidence", "MONEY_TOLERANCE"]
