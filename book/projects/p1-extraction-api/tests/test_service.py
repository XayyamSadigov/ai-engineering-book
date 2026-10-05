# path: book/projects/p1-extraction-api/tests/test_service.py
import asyncio

import pytest
from aie_core import CompletionRequest, RateLimitError, count_message_tokens
from conftest import CLASSIFY_INVOICE, GOLD, INV1_TEXT, draft

from extraction_api.application import DocumentClassifier, DocumentIn, DocumentTooLarge
from extraction_api.domain import DocumentType, Route


def test_happy_path_accepts_and_normalizes(make_service, queue, tracer):
    svc, llm = make_service([CLASSIFY_INVOICE, draft()])
    res = svc.extract(DocumentIn(document_id="INV-001", text=INV1_TEXT), request_id="req-1")

    assert res.route is Route.ACCEPT and res.review_id is None
    assert res.doc_type is DocumentType.INVOICE and res.classification_confidence == pytest.approx(0.96)
    assert res.data["total"] == "3327.48" and res.data["line_items"][0]["quantity"] == "2400"
    assert res.score == pytest.approx(0.95) and res.llm_calls == 2
    assert queue.list() == []
    # the schema went to the provider natively and the request was tagged for tracing
    assert llm.requests[1].response_schema is not None
    assert llm.requests[1].metadata["task"] == "extract_invoice"
    doc_span = tracer.find("extract.document")[0]
    assert doc_span.attributes["route"] == "accept" and doc_span.attributes["request_id"] == "req-1"
    assert [s.name for s in tracer.spans].count("extract.llm") == 1


def test_schema_repair_loop_recovers_and_bills_every_call(make_service):
    broken = draft()
    del broken["total"]                      # required key missing: pydantic rejects it
    svc, llm = make_service([CLASSIFY_INVOICE, "Sure! Here is the JSON: {not json", broken, draft()])
    res = svc.extract(DocumentIn(text=INV1_TEXT))

    assert res.route is Route.ACCEPT
    assert res.llm_calls == 4                # classify + 3 extraction attempts
    final_req = llm.requests[-1]
    assert "did not match the required schema" in final_req.messages[-1].text
    # usage is the sum over all four calls, not just the final completion
    assert res.usage.input_tokens == sum(count_message_tokens(r.messages) for r in llm.requests)
    assert res.usage.input_tokens > 3 * count_message_tokens(llm.requests[1].messages)


def test_schema_failure_after_budget_routes_to_review(make_service, queue):
    svc, _ = make_service([CLASSIFY_INVOICE, "nope", "still nope", "no"], max_schema_repairs=2)
    res = svc.extract(DocumentIn(text=INV1_TEXT))
    assert res.route is Route.HUMAN_REVIEW and res.reasons == ["SCHEMA_FAILURE"]
    assert queue.get(res.review_id).reasons == ["SCHEMA_FAILURE"]


def test_rule_repair_fixes_a_misread_value(make_service):
    misread = draft(total="3,237.48")       # digits swapped: total no longer adds up
    misread["evidence"][4] = {"field": "total", "quote": "$3,327.48", "confidence": 0.9}
    svc, llm = make_service([CLASSIFY_INVOICE, misread, draft()])
    res = svc.extract(DocumentIn(text=INV1_TEXT))

    assert res.route is Route.ACCEPT and res.rule_repairs == 1
    repair_req = llm.requests[-1]
    assert repair_req.metadata["task"] == "repair_invoice"
    assert "TOTAL_MISMATCH" in repair_req.messages[-1].text
    assert "do not change numbers to make totals agree" in repair_req.messages[-1].text


def test_document_inconsistency_survives_repair_and_goes_to_review(make_service, queue):
    gold = GOLD["INV-007"]                   # the vendor printed a total that does not add up
    e = gold["expected"]
    stated = {
        "vendor": e["vendor"], "invoice_number": e["invoice_number"], "invoice_date": e["invoice_date"],
        "due_date": e["due_date"], "po_number": e["po_number"], "currency": "USD",
        "line_items": e["line_items"], "subtotal": "60000.00", "tax_rate": "0.00", "tax_amount": "0.00",
        "total": "59000.00",
        "evidence": [
            {"field": "vendor", "quote": "Issuer: Vantage Software Ltd", "confidence": 0.95},
            {"field": "invoice_number", "quote": "VS-INV-104877", "confidence": 0.95},
            {"field": "invoice_date", "quote": "Issue date: 2026-02-15", "confidence": 0.95},
            {"field": "currency", "quote": "Currency: USD", "confidence": 0.95},
            {"field": "total", "quote": "total=59000.00", "confidence": 0.95},
        ],
    }
    svc, _ = make_service([stated, stated])  # same answer before and after the repair prompt
    res = svc.extract(DocumentIn(document_id="INV-007", text=gold["text"], doc_type=DocumentType.INVOICE))

    assert res.route is Route.HUMAN_REVIEW and res.reasons == ["TOTAL_MISMATCH"]
    assert res.rule_repairs == 1 and res.llm_calls == 2   # hint skipped classification
    item = queue.get(res.review_id)
    assert item.document_id == "INV-007" and item.data["total"] == "59000.00"
    assert item.status == "pending"


def test_fabricated_evidence_routes_to_review(make_service):
    fake = draft()
    fake["evidence"][4] = {"field": "total", "quote": "Amount payable: $3,327.48", "confidence": 0.99}
    svc, _ = make_service([CLASSIFY_INVOICE, fake, fake])
    res = svc.extract(DocumentIn(text=INV1_TEXT))
    assert res.route is Route.HUMAN_REVIEW and "EVIDENCE_NOT_FOUND" in res.reasons
    assert res.field_scores["total"] == 0.0


def test_low_confidence_routes_to_review(make_service):
    unsure = draft()
    for ev in unsure["evidence"]:
        ev["confidence"] = 0.55
    svc, _ = make_service([CLASSIFY_INVOICE, unsure])
    res = svc.extract(DocumentIn(text=INV1_TEXT))
    assert res.route is Route.HUMAN_REVIEW and res.reasons == ["LOW_CONFIDENCE:0.55"]


def test_classifier_abstains_on_unknown_documents(make_service, queue):
    svc, _ = make_service([{"doc_type": "other", "confidence": 0.4, "reason": "newsletter"}])
    res = svc.extract(DocumentIn(text="Vendor newsletter: our spring catalog is here!"))
    assert res.route is Route.HUMAN_REVIEW and res.reasons == ["UNCLASSIFIED"] and res.llm_calls == 1
    assert len(queue.list()) == 1


def test_self_consistency_uses_agreement_as_confidence(make_service):
    votes = [CLASSIFY_INVOICE, CLASSIFY_INVOICE, {"doc_type": "support_ticket", "confidence": 0.99, "reason": "x"}]
    svc, llm = make_service(votes, classifier=DocumentClassifier(threshold=0.7, samples=3))
    res = svc.classifier.classify(llm, INV1_TEXT)
    assert res.doc_type is DocumentType.INVOICE and res.confidence == pytest.approx(2 / 3)
    assert res.abstained is True                       # 0.67 < 0.70: disagreement means abstain
    assert all(r.temperature > 0 for r in llm.requests)


def test_ticket_extraction(make_service):
    text = "Subject: Register 3 declines every card\n\nStore 0412 here. Since we opened at 8 all cards are declined on register 3."
    ticket = {
        "category": "pos_payments", "priority": "P1", "summary": "Register declines all cards at one store",
        "affected_system": "PayBridge", "contains_personal_data": False,
        "entities": [{"type": "store", "value": "0412", "quote": "Store 0412"}],
        "evidence": [{"field": "category", "quote": "declines every card", "confidence": 0.93},
                     {"field": "priority", "quote": "all cards are declined", "confidence": 0.9}],
    }
    svc, _ = make_service([{"doc_type": "support_ticket", "confidence": 0.95, "reason": "store issue"}, ticket])
    res = svc.extract(DocumentIn(text=text))
    assert res.route is Route.ACCEPT
    assert res.data["category"] == "pos_payments" and res.data["priority"] == "P1"
    assert [e["value"] for e in res.data["entities"]] == ["0412"]   # pattern and model agree; deduplicated


def test_oversized_document_is_rejected_before_any_model_call(make_service):
    svc, llm = make_service([], max_document_chars=100)
    with pytest.raises(DocumentTooLarge):
        svc.extract(DocumentIn(text="x" * 101))
    assert llm.requests == []


async def test_batch_preserves_order_isolates_failures_and_bounds_concurrency(make_service):
    in_flight = {"now": 0, "max": 0}

    def handler(req: CompletionRequest):
        if "EXPLODE" in req.messages[-1].text:
            raise RateLimitError("provider says slow down", retry_after_s=1.0)
        in_flight["now"] += 1
        in_flight["max"] = max(in_flight["max"], in_flight["now"])
        try:
            return CLASSIFY_INVOICE if req.metadata["task"] == "classify" else draft()
        finally:
            in_flight["now"] -= 1

    svc, _ = make_service(handler=handler, batch_concurrency=2)
    docs = [DocumentIn(document_id=f"d{i}", text=INV1_TEXT) for i in range(5)]
    docs.insert(2, DocumentIn(document_id="boom", text="EXPLODE"))
    items = await svc.extract_batch(docs, request_id="batch-1")

    assert [i.index for i in items] == list(range(6))
    assert items[2].result is None and "RateLimitError" in items[2].error
    assert all(i.result.route is Route.ACCEPT for k, i in enumerate(items) if k != 2)
    assert items[3].result.request_id == "batch-1-3"
    assert in_flight["max"] <= 2


def test_batch_runs_from_sync_code(make_service):
    svc, _ = make_service(handler=lambda req: CLASSIFY_INVOICE if req.metadata["task"] == "classify" else draft())
    items = asyncio.run(svc.extract_batch([DocumentIn(text=INV1_TEXT)] * 3))
    assert len(items) == 3 and all(i.result.route is Route.ACCEPT for i in items)
