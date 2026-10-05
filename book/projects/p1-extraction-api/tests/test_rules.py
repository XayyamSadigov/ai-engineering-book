# path: book/projects/p1-extraction-api/tests/test_rules.py
from datetime import date
from decimal import Decimal

from conftest import INV1_TEXT, TODAY, draft

from extraction_api.domain import (
    InvoiceDraft, Route, RoutingPolicy, RuleViolation, Severity, SupportTicketDraft, build_invoice,
    build_ticket, check_invoice, check_ticket, decide,
)
from extraction_api.domain.common import EvidenceSpan
from extraction_api.domain.rules import check_evidence


def _invoice(**changes):
    data, inv, violations = build_invoice(InvoiceDraft.model_validate(draft(**changes)), INV1_TEXT)
    return data, inv, violations


def codes(vs):
    return {v.code for v in vs}


def test_consistent_invoice_passes_every_rule():
    _, inv, violations = _invoice()
    assert violations == []
    assert inv is not None and inv.total == Decimal("3327.48") and inv.tax_rate == Decimal("0.08")
    assert check_invoice(inv, today=TODAY) == []
    assert all(span.found for span in inv.evidence)


def test_line_sum_and_total_mismatch():
    _, inv, _ = _invoice(subtotal="3,181.00")
    found = codes(check_invoice(inv, today=TODAY))
    assert {"LINE_SUM_MISMATCH", "TOTAL_MISMATCH"} <= found


def test_line_amount_mismatch_respects_printed_precision():
    lines = draft()["line_items"]
    # "7.90" printed with two decimals tolerates 120 x 0.005 = 0.60 of rounding; 950 is off by 2.00.
    # As a float, 7.9 would look like one decimal and tolerate 6.00: keep printed precision.
    lines[1] = {**lines[1], "unit_price": "7.90", "amount": 950.00}
    _, inv, _ = _invoice(line_items=lines)
    assert "LINE_AMOUNT_MISMATCH" in codes(check_invoice(inv, today=TODAY))


def test_tiny_unit_price_with_huge_quantity_is_exact():
    lines = [{"description": "Gateway fees", "quantity": "4.82e+07", "unit_price": "0.0045", "amount": "216900.00"}]
    _, inv, _ = _invoice(line_items=lines, subtotal="216900.00", tax_amount=0, tax_rate=0, total="216900.00")
    assert "LINE_AMOUNT_MISMATCH" not in codes(check_invoice(inv, today=TODAY))


def test_date_rules():
    _, inv, _ = _invoice(due_date="2026-01-01")
    assert "DUE_BEFORE_ISSUE" in codes(check_invoice(inv, today=TODAY))
    _, inv, _ = _invoice(invoice_date="2027-01-14", due_date=None)
    assert "DATE_IN_FUTURE" in codes(check_invoice(inv, today=TODAY))
    old = check_invoice(inv.model_copy(update={"invoice_date": date(2020, 1, 1)}), today=TODAY)
    assert any(v.code == "DATE_TOO_OLD" and v.severity is Severity.WARNING for v in old)


def test_missing_po_is_a_business_rule_not_a_schema_rule():
    evidence = [e for e in draft()["evidence"] if e["field"] != "po_number"]
    _, inv, violations = _invoice(po_number="(not provided)", evidence=evidence)
    assert violations == [] and inv.po_number is None   # schema-valid
    assert "MISSING_PO" in codes(check_invoice(inv, today=TODAY))
    assert "MISSING_PO" not in codes(check_invoice(inv, today=TODAY, require_po=False))


def test_missing_critical_field_and_bad_currency_do_not_build_a_record():
    data, inv, violations = _invoice(total=None, currency="Doubloons")
    assert inv is None
    assert {"MISSING_FIELD", "UNKNOWN_CURRENCY"} <= codes(violations)
    assert data["vendor"] == "Brightline Supply Co."   # partial work is kept for the reviewer


def test_ambiguous_date_is_not_repairable():
    _, inv, violations = _invoice(invoice_date="03/04/2026")
    v = next(v for v in violations if v.code == "AMBIGUOUS_DATE")
    assert v.repairable is False and inv is None


def test_quote_must_contain_the_value():
    # misread total; the quote "TOTAL DUE $3,327.48" is real text, but it does not support 3,237.48
    _, _, violations = _invoice(total="3,237.48")
    assert "EVIDENCE_VALUE_MISMATCH" in codes(violations)


def test_evidence_must_exist_in_document():
    spans = [EvidenceSpan(field="total", quote="TOTAL DUE $9,999.00", model_confidence=0.99)]
    found = check_evidence(spans, {"total", "vendor"}, ("vendor", "total"))
    assert codes(found) == {"EVIDENCE_NOT_FOUND", "MISSING_EVIDENCE"}


def test_ticket_entities_grounding_and_pii_flag():
    text = "Please send Jordan's number. Mine is +1-555-0142, store 0412 register 3 shows SH-305."
    d = SupportTicketDraft.model_validate({
        "category": "account_access", "priority": "P3", "summary": "Requests a colleague's phone number",
        "affected_system": None, "contains_personal_data": False,
        "entities": [
            {"type": "person", "value": "Jordan", "quote": "Jordan"},
            {"type": "client_account", "value": "7731", "quote": "account 7731"},  # not in text
        ],
        "evidence": [{"field": "category", "quote": "send Jordan's number", "confidence": 0.8},
                     {"field": "priority", "quote": "Please send", "confidence": 0.7}],
    })
    data, ticket, violations = build_ticket(d, text)
    types = {(e.type, e.value, e.source) for e in ticket.entities}
    assert ("person", "Jordan", "model") in types
    assert ("phone", "+1-555-0142", "pattern") in types
    assert ("error_code", "SH-305", "pattern") in types
    assert not any(e.type == "client_account" for e in ticket.entities)
    assert ticket.contains_personal_data is True
    assert {"ENTITY_NOT_GROUNDED", "PII_FLAG_CORRECTED"} <= codes(violations)
    assert all(v.severity is Severity.WARNING for v in violations)


def test_ticket_rules_abstention_and_p1_evidence():
    d = SupportTicketDraft.model_validate({
        "category": "other", "priority": "P1", "summary": "Unclear", "affected_system": None,
        "contains_personal_data": False, "entities": [], "evidence": [],
    })
    _, ticket, _ = build_ticket(d, "something odd happened")
    assert codes(check_ticket(ticket)) == {"CATEGORY_ABSTAINED", "P1_WITHOUT_EVIDENCE"}


def test_routing_decisions():
    policy = RoutingPolicy(accept_threshold=0.8, max_rule_repairs=1)
    repairable = RuleViolation(code="TOTAL_MISMATCH", detail="x", repairable=True)
    fixed = RuleViolation(code="AMBIGUOUS_DATE", detail="x", repairable=False)
    warning = RuleViolation(code="DATE_TOO_OLD", detail="x", severity=Severity.WARNING)
    assert decide([repairable], 0.9, 0, policy).route is Route.REPAIR
    assert decide([repairable], 0.9, 1, policy).route is Route.HUMAN_REVIEW
    assert decide([fixed], 0.9, 0, policy).route is Route.HUMAN_REVIEW
    assert decide([warning], 0.9, 0, policy).route is Route.ACCEPT
    low = decide([], 0.5, 0, policy)
    assert low.route is Route.HUMAN_REVIEW and low.reasons == ["LOW_CONFIDENCE:0.50"]
