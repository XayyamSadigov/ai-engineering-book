# path: book/projects/p3-rag-assistant/tests/test_p3_api.py
"""HTTP contract: auth, JSON answers, the SSE streaming contract, admin endpoints, status."""
from __future__ import annotations

import json

import pytest
from conftest import PTO_Q
from fastapi.testclient import TestClient

from rag_assistant.api.app import create_app
from rag_assistant.api.auth import issue_token

SECRET = "test-secret"


@pytest.fixture
def client(make_container):  # type: ignore[no-untyped-def]
    c = make_container(auth_secret=SECRET, inline_ingest=True)
    return TestClient(create_app(c)), c


def bearer(tenant: str = "retail", groups: list[str] | None = None, user: str = "u1") -> dict[str, str]:
    return {"Authorization": "Bearer " + issue_token(SECRET, user_id=user, tenant=tenant, groups=groups or ["all"])}


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_requests_without_a_valid_token_are_rejected(client):
    api, _ = client
    assert api.post("/v1/ask", json={"question": "hi"}).status_code == 401
    forged = issue_token("other-secret", user_id="x", tenant="retail", groups=["all", "it-oncall"])
    r = api.post("/v1/ask", json={"question": "hi"}, headers={"Authorization": f"Bearer {forged}"})
    assert r.status_code == 401


def test_ask_returns_a_cited_answer(client):
    api, _ = client
    r = api.post("/v1/ask", json={"question": PTO_Q}, headers={**bearer(), "X-Request-Id": "req-123"})
    assert r.status_code == 200 and r.headers["x-request-id"] == "req-123"
    body = r.json()
    assert body["mode"] == "answer" and body["answer"]["action"] in ("answer", "answer_with_caveat")
    assert body["citations"][0]["doc_id"] == "hr-pto-policy"
    assert body["index_version"] == "v1"


def test_sse_streaming_contract(client):
    api, _ = client
    r = api.post("/v1/ask", json={"question": PTO_Q, "stream": True}, headers=bearer())
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    kinds = [e for e, _ in events]
    assert kinds[0] == "meta" and kinds[-1] == "done"
    assert events[0][1]["index_version"] == "v1"
    assert "text" in kinds
    seen: set[str] = set()
    for kind, data in events:  # a citation event always precedes the first text that uses it
        if kind == "citation":
            seen.add(data["eid"])
        if kind == "text":
            assert set(data["eids"]) <= seen
    assert events[-1][1]["status"] in ("answered", "partial", "conflict")


def test_sse_with_accept_header_and_abstention(client):
    api, _ = client
    r = api.post("/v1/ask", json={"question": "What is the response target for a SEV1 incident?"},
                 headers={**bearer(tenant="shared"), "Accept": "text/event-stream"})
    events = parse_sse(r.text)
    assert events[-1][0] == "done" and events[-1][1]["action"] == "abstain"
    assert not [e for e, _ in events if e == "text"]


def test_admin_upload_and_delete(client):
    api, c = client
    doc = ("---\nid: retail-store-hours\ntitle: Store Hours\nversion: \"1\"\nupdated_at: 2026-05-01\n"
           "tenant: retail\nacl_groups: [\"all\"]\ntags: [retail, stores]\n---\n\n# Store Hours\n\n"
           "## Weekdays\n\nAll Northwind retail stores open at 09:00 and close at 21:00 on weekdays.\n")
    body = {"filename": "store-hours.md", "content": doc}
    assert api.post("/v1/documents", json=body, headers=bearer()).status_code == 403  # not an admin
    assert api.post("/v1/documents", json=body, headers=bearer("logistics", ["rag-admin"])).status_code == 403
    r = api.post("/v1/documents", json=body, headers=bearer("retail", ["rag-admin"]))
    assert r.status_code == 202, r.text
    q = {"question": "What time do retail stores open and close on weekdays?"}
    assert "retail-store-hours" in [x["doc_id"] for x in api.post("/v1/ask", json=q, headers=bearer()).json()["citations"]]
    assert api.delete("/v1/documents/retail-store-hours", headers=bearer()).status_code == 403
    assert api.delete("/v1/documents/retail-store-hours", headers=bearer("retail", ["rag-admin"])).status_code == 202
    cited = [x["doc_id"] for x in api.post("/v1/ask", json=q, headers=bearer()).json()["citations"]]
    assert "retail-store-hours" not in cited
    assert c.registry.get("retail-store-hours").status == "deleted"
    bad = {"filename": "x.md", "content": "---\nid: x\n---\n\n# no acl\n\ntext text text"}
    assert api.post("/v1/documents", json=bad, headers=bearer("retail", ["rag-admin"])).status_code == 422


def test_admin_cannot_take_over_another_tenants_document_id(client):
    api, c = client
    doc = ("---\nid: prod-logistics-route-planner\ntitle: Hijacked\nversion: \"9\"\nupdated_at: 2026-05-01\n"
           "tenant: retail\nacl_groups: [\"all\"]\n---\n\n# Hijacked\n\n## Body\n\nReplacement text for the planner.\n")
    before = c.registry.get("prod-logistics-route-planner")
    r = api.post("/v1/documents", json={"filename": "x.md", "content": doc}, headers=bearer("retail", ["rag-admin"]))
    assert r.status_code == 403
    assert c.registry.get("prod-logistics-route-planner") == before  # still logistics, content untouched


def test_status_health_and_metrics(client):
    api, _ = client
    api.post("/v1/ask", json={"question": PTO_Q}, headers=bearer())
    status = api.get("/v1/index/status", headers=bearer()).json()
    assert status["active_version"] == "v1" and status["documents_active"] == 24
    assert status["freshness_slo_met"] is True and status["dead_letters"] == 0
    assert api.get("/healthz").json()["status"] == "ok"
    metrics = api.get("/metrics").text
    assert 'rag_requests_total{cache="miss",mode="answer"}' in metrics
    assert 'rag_stage_latency_ms{stage="retrieve",quantile="0.95"}' in metrics


def test_tenant_quota_returns_429(make_container):
    c = make_container(auth_secret=SECRET, tenant_requests_per_minute=4)  # burst = 25% of 4 = 1 request
    api = TestClient(create_app(c))
    codes = [api.post("/v1/ask", json={"question": PTO_Q}, headers=bearer()).status_code for _ in range(3)]
    assert codes[0] == 200 and 429 in codes
    other = api.post("/v1/ask", json={"question": PTO_Q}, headers=bearer(tenant="logistics"))
    assert other.status_code == 200  # one tenant's burst does not starve another


def test_api_refuses_the_published_dev_secret(make_container):  # type: ignore[no-untyped-def]
    with pytest.raises(RuntimeError, match="RAG_AUTH_SECRET"):
        create_app(make_container())  # default settings: the dev secret, no explicit opt-in
    create_app(make_container(allow_dev_auth_secret=True))  # explicit local-development opt-in
