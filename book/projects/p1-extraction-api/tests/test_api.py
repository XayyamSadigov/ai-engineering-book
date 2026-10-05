# path: book/projects/p1-extraction-api/tests/test_api.py
import pytest
from aie_core import CompletionRequest, ProviderUnavailableError
from conftest import CLASSIFY_INVOICE, GOLD, INV1_TEXT, draft
from fastapi.testclient import TestClient

from extraction_api.api import create_app
from extraction_api.config import AppSettings


def _handler(req: CompletionRequest):
    """Classify everything as an invoice; extract INV-001 correctly, and without a PO when the PO line is gone."""
    if req.metadata["task"] == "classify":
        return CLASSIFY_INVOICE
    if "PO-NW-2026-10412" in req.messages[1].text:
        return draft()
    evidence = [e for e in draft()["evidence"] if e["field"] != "po_number"]
    return draft(po_number=None, evidence=evidence)


@pytest.fixture
def client(make_service):
    svc, _ = make_service(handler=_handler)
    app = create_app(svc, AppSettings(max_batch_size=3))
    return TestClient(app)


def test_extract_accepts_and_echoes_request_id(client, tracer):
    r = client.post("/extract", json={"document_id": "INV-001", "text": INV1_TEXT}, headers={"X-Request-ID": "abc-123"})
    assert r.status_code == 200
    body = r.json()
    assert body["route"] == "accept" and body["request_id"] == "abc-123"
    assert r.headers["X-Request-ID"] == "abc-123"
    http = tracer.find("http.request")[0]
    assert http.attributes["status_code"] == 200 and http.attributes["path"] == "/extract"


def test_unsafe_request_id_is_replaced(client):
    r = client.post("/extract", json={"text": INV1_TEXT}, headers={"X-Request-ID": "bad id\nINJECTED"})
    assert r.headers["X-Request-ID"] != "bad id\nINJECTED" and len(r.headers["X-Request-ID"]) == 32


def test_validation_errors_are_422(client):
    assert client.post("/extract", json={"text": ""}).status_code == 422
    assert client.post("/extract", json={"text": "x", "doc_type": "receipt"}).status_code == 422


def test_review_flow_list_then_correct(client):
    text = INV1_TEXT.replace("PO Number: PO-NW-2026-10412\n", "")
    res = client.post("/extract", json={"document_id": "no-po", "text": text}).json()
    assert res["route"] == "human_review" and res["reasons"] == ["MISSING_PO"]

    pending = client.get("/review").json()
    assert [p["review_id"] for p in pending] == [res["review_id"]]
    assert pending[0]["data"]["po_number"] is None

    bad = client.post(f"/review/{res['review_id']}/resolve",
                      json={"decision": "correct", "reviewer": "ap-1", "corrected_data": {"total": "abc"}})
    assert bad.status_code == 422

    fixed = {**pending[0]["data"], "po_number": "PO-NW-2026-10999"}
    ok = client.post(f"/review/{res['review_id']}/resolve",
                     json={"decision": "correct", "reviewer": "ap-1", "corrected_data": fixed})
    assert ok.status_code == 200
    assert ok.json()["status"] == "corrected" and ok.json()["resolution"]["corrected_data"]["po_number"] == "PO-NW-2026-10999"

    again = client.post(f"/review/{res['review_id']}/resolve", json={"decision": "approve", "reviewer": "ap-2"})
    assert again.status_code == 409
    assert client.get("/review").json() == []
    assert len(client.get("/review", params={"status": "corrected"}).json()) == 1


def test_resolve_unknown_item_is_404(client):
    r = client.post("/review/does-not-exist/resolve", json={"decision": "approve", "reviewer": "x"})
    assert r.status_code == 404


def test_batch_endpoint_and_limit(client):
    docs = [{"document_id": "a", "text": INV1_TEXT}, {"document_id": "b", "text": GOLD["INV-009"]["text"]}]
    r = client.post("/extract/batch", json={"documents": docs})
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["total"] == 2 and body["summary"]["accepted"] == 1 and body["summary"]["human_review"] == 1
    assert body["items"][1]["result"]["request_id"] == f"{body['request_id']}-1"
    too_many = client.post("/extract/batch", json={"documents": docs * 2})
    assert too_many.status_code == 413


def test_provider_outage_is_503_not_a_review_item(make_service, queue):
    def down(req):
        raise ProviderUnavailableError("upstream 503")

    svc, _ = make_service(handler=down)
    c = TestClient(create_app(svc, AppSettings()))
    r = c.post("/extract", json={"text": INV1_TEXT})
    assert r.status_code == 503 and r.json()["error"] == "ProviderUnavailableError"
    assert queue.list() == []   # humans do not re-key documents because a provider is down


def test_document_too_large_is_413(make_service):
    svc, _ = make_service(handler=_handler, max_document_chars=50)
    c = TestClient(create_app(svc, AppSettings()))
    assert c.post("/extract", json={"text": "x" * 51}).status_code == 413
