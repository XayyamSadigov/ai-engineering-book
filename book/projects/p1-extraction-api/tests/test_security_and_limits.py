# path: book/projects/p1-extraction-api/tests/test_security_and_limits.py
"""Authentication, tenant scoping, request limits, error mapping, and the per-document deadline."""
import asyncio

import pytest
from aie_core import InvalidRequestError, RateLimitError
from conftest import CLASSIFY_INVOICE, INV1_TEXT, draft
from fastapi.testclient import TestClient

from extraction_api.adapters import SQLiteReviewQueue
from extraction_api.api import create_app
from extraction_api.application import DocumentIn
from extraction_api.application.ports import ReviewItem
from extraction_api.config import ApiKey, AppSettings
from extraction_api.domain import DocumentType, Route

NO_PO = INV1_TEXT.replace("PO Number: PO-NW-2026-10412\n", "")
KEYS = [
    ApiKey(key="k-retail-submit", principal="retail-ingest", role="submitter", tenant="retail"),
    ApiKey(key="k-retail-review", principal="ap-clerk-7", role="reviewer", tenant="retail"),
    ApiKey(key="k-logi-review", principal="logi-clerk-2", role="reviewer", tenant="logistics"),
]


def _handler(req):
    if req.metadata["task"] == "classify":
        return CLASSIFY_INVOICE
    if "PO-NW-2026-10412" in req.messages[1].text:
        return draft()
    return draft(po_number=None, evidence=[e for e in draft()["evidence"] if e["field"] != "po_number"])


def bearer(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture
def secured(make_service):
    svc, _ = make_service(handler=_handler)
    return TestClient(create_app(svc, AppSettings(api_keys=KEYS, max_request_bytes=20_000)))


def test_missing_or_wrong_token_is_401(secured):
    assert secured.post("/extract", json={"text": INV1_TEXT}).status_code == 401
    assert secured.post("/extract", json={"text": INV1_TEXT}, headers=bearer("nope")).status_code == 401
    assert secured.get("/healthz").status_code == 200   # liveness stays open for the orchestrator


def test_roles_are_enforced(secured):
    assert secured.post("/extract", json={"text": INV1_TEXT}, headers=bearer("k-retail-review")).status_code == 403
    assert secured.get("/review", headers=bearer("k-retail-submit")).status_code == 403


def test_tenant_bound_key_stamps_its_tenant_and_cannot_claim_another(secured, queue):
    res = secured.post("/extract", json={"text": NO_PO}, headers=bearer("k-retail-submit")).json()
    assert res["route"] == "human_review"
    assert queue.get(res["review_id"]).tenant == "retail"
    other = secured.post("/extract", json={"text": NO_PO, "tenant": "logistics"}, headers=bearer("k-retail-submit"))
    assert other.status_code == 403


def test_review_items_are_invisible_across_tenants(secured):
    rid = secured.post("/extract", json={"text": NO_PO}, headers=bearer("k-retail-submit")).json()["review_id"]
    assert [i["review_id"] for i in secured.get("/review", headers=bearer("k-retail-review")).json()] == [rid]
    assert secured.get("/review", headers=bearer("k-logi-review")).json() == []
    assert secured.get(f"/review/{rid}", headers=bearer("k-logi-review")).status_code == 404
    r = secured.post(f"/review/{rid}/resolve", json={"decision": "approve", "reviewer": "x"},
                     headers=bearer("k-logi-review"))
    assert r.status_code == 404


def test_reviewer_identity_comes_from_credentials_not_the_body(secured):
    rid = secured.post("/extract", json={"text": NO_PO}, headers=bearer("k-retail-submit")).json()["review_id"]
    r = secured.post(f"/review/{rid}/resolve", json={"decision": "reject", "reviewer": "the-cfo"},
                     headers=bearer("k-retail-review"))
    assert r.status_code == 200 and r.json()["resolution"]["reviewer"] == "ap-clerk-7"


def test_oversized_body_is_413_before_parsing(secured):
    r = secured.post("/extract", json={"text": "x" * 30_000}, headers=bearer("k-retail-submit"))
    assert r.status_code == 413 and "request body limit" in r.json()["detail"]


def test_non_retryable_provider_error_is_502_not_503(make_service, queue):
    def reject(req):
        raise InvalidRequestError("schema keyword not supported")

    svc, _ = make_service(handler=reject)
    r = TestClient(create_app(svc, AppSettings())).post("/extract", json={"text": INV1_TEXT})
    assert r.status_code == 502 and r.json()["retryable"] is False
    assert queue.list() == []


def test_batch_items_carry_retryable(make_service):

    def flaky(req):
        if req.metadata["task"] == "classify":
            return CLASSIFY_INVOICE
        raise RateLimitError("slow down")

    svc, _ = make_service(handler=flaky)
    items = asyncio.run(svc.extract_batch([DocumentIn(text=INV1_TEXT)]))
    assert items[0].error and items[0].retryable is True


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_exhausted_deadline_is_an_infrastructure_failure(make_service, queue):
    clock = FakeClock()

    def slow(req):
        clock.t += 200.0   # the classification call alone eats the whole budget
        return CLASSIFY_INVOICE

    svc, _ = make_service(handler=slow, document_deadline_s=120.0, clock=clock)
    r = TestClient(create_app(svc, AppSettings())).post("/extract", json={"text": INV1_TEXT})
    assert r.status_code == 503 and r.json()["error"] == "TimeoutError"
    assert queue.list() == []


def test_low_remaining_deadline_skips_repair_and_degrades_to_review(make_service):
    clock = FakeClock()
    misread = draft(total="3,237.48")

    def handler(req):
        clock.t += 50.0
        return misread

    svc, llm = make_service(handler=handler, document_deadline_s=60.0, call_timeout_s=45.0, clock=clock)
    res = svc.extract(DocumentIn(text=INV1_TEXT, doc_type=DocumentType.INVOICE))
    assert res.route is Route.HUMAN_REVIEW and "REPAIR_SKIPPED_DEADLINE" in res.reasons
    assert res.rule_repairs == 0 and len(llm.requests) == 1
    assert llm.requests[0].timeout_s == 45.0


def test_call_timeout_is_capped_by_remaining_deadline(make_service):
    clock = FakeClock()

    def handler(req):
        clock.t += 100.0
        return CLASSIFY_INVOICE if req.metadata["task"] == "classify" else draft()

    svc, llm = make_service(handler=handler, document_deadline_s=130.0, call_timeout_s=45.0, clock=clock)
    res = svc.extract(DocumentIn(text=INV1_TEXT))
    assert res.route is Route.ACCEPT
    assert [r.timeout_s for r in llm.requests] == [45.0, 30.0]


def test_sqlite_queue_filters_by_tenant(tmp_path):
    q = SQLiteReviewQueue(tmp_path / "r.db")
    for i, tenant in enumerate(["retail", "logistics", None]):
        q.enqueue(ReviewItem(review_id=f"r{i}", request_id="x", tenant=tenant, doc_type=DocumentType.INVOICE,
                             reasons=["MISSING_PO"], document_text="t", prompt_version="v"))
    assert [i.review_id for i in q.list(tenant="retail")] == ["r0"]
    assert [i.review_id for i in q.list(None, tenant="logistics")] == ["r1"]
    assert len(q.list()) == 3
    q.close()
