# path: book/projects/examples/ch25/tests/test_extraction.py
from __future__ import annotations

import pytest

from taskevals.extraction import (
    ExtractionEvaluator,
    evidence_status,
    line_item_prf,
    locate_value,
    value_renderings,
    weighted_field_prf,
)
from taskevals.suites import build_extraction_dataset, run_suite


@pytest.fixture(scope="module")
def ds():
    return build_extraction_dataset()


def _perfect(case) -> dict:
    gold = case.expected
    fields = {k: gold[k] for k in ["vendor", "invoice_number", "invoice_date", "due_date", "po_number",
                                   "currency", "subtotal", "tax_amount", "total"]}
    ev = {k: locate_value(case.input["text"], v) for k, v in fields.items() if v is not None}
    return {"fields": fields, "line_items": gold["line_items"], "evidence": ev}


def test_weighting_makes_a_critical_miss_cost_more_than_a_minor_one() -> None:
    w = {"total": 3.0, "po_number": 1.0, "vendor": 1.0}
    _, r_total, _ = weighted_field_prf({"total": "fn", "po_number": "tp", "vendor": "tp"}, w)
    _, r_po, _ = weighted_field_prf({"total": "tp", "po_number": "fn", "vendor": "tp"}, w)
    assert r_total == pytest.approx(2 / 5) and r_po == pytest.approx(4 / 5)


def test_number_renderings_cover_document_formats() -> None:
    assert "3,327.48" in value_renderings(3327.48)
    assert "23368.00" in value_renderings(23368.0)
    assert "1,728" in value_renderings(1728.0)


def test_evidence_status_distinguishes_wrong_line_from_fabricated_quote(ds) -> None:
    text = ds.get("INV-001").input["text"]
    assert evidence_status(text, "TOTAL DUE     $3,327.48", 3327.48) == "correct"
    assert evidence_status(text, "Subtotal     $3,081.00", 3327.48) == "wrong_location"
    assert evidence_status(text, "Total: $3,327.48", 3327.48) == "not_found"
    assert evidence_status(text, None, 3327.48) == "missing"


def test_perfect_extraction_scores_one_everywhere(ds) -> None:
    ev = ExtractionEvaluator()
    for case in ds:
        scores = {s.name: s.value for s in ev(case, _perfect(case))}
        assert all(v == 1.0 for v in scores.values()), (case.id, scores)


def test_wrong_total_fails_critical_and_counts_as_fp_and_fn(ds) -> None:
    case = ds.get("INV-006")
    out = _perfect(case)
    out["fields"]["total"] = case.expected["subtotal"]
    scores = {s.name: s for s in ExtractionEvaluator()(case, out)}
    assert scores["critical_fields_exact"].passed is False and scores["critical_fields_exact"].detail == ["total"]
    assert scores["field_f1_weighted"].detail == {"total": "fp+fn"}
    assert scores["field_precision"].value < 1.0 and scores["field_recall"].value < 1.0


def test_invented_po_number_is_a_false_positive(ds) -> None:
    case = ds.get("INV-009")  # gold has no PO number
    out = _perfect(case)
    out["fields"]["po_number"] = "PO-NW-2026-99999"
    scores = {s.name: s for s in ExtractionEvaluator()(case, out)}
    assert scores["field_recall"].value == 1.0 and scores["field_precision"].value < 1.0


def test_line_items_match_on_amount_and_description() -> None:
    gold = [{"description": "Thermal label roll 100x150", "amount": 948.0},
            {"description": "Tote lid, stackable", "amount": 645.0}]
    p, r = line_item_prf([{"description": "Thermal label roll 100x150", "amount": 948.0}], gold)
    assert (p, r) == (1.0, 0.5)


def test_suite_shows_baseline_po_miss_and_candidate_fix() -> None:
    base, cand = run_suite("extraction", "baseline"), run_suite("extraction", "candidate")
    letters = {"INV-002", "INV-005", "INV-010", "INV-013", "INV-018"}
    assert {c for c, v in base.case_scores("field_recall").items() if v < 1.0} == letters
    assert cand.mean("field_f1_weighted") == 1.0 and cand.mean("evidence_correct") == 1.0
