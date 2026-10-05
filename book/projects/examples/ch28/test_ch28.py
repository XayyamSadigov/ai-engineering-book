# path: book/projects/examples/ch28/test_ch28.py
"""Offline tests for the Chapter 28 API skeleton. No network, no API keys."""
from __future__ import annotations

import asyncio
import json
import pathlib
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api_skeleton import Budget, FakeModel, RecordingWebhook, build_app

RETAIL = {"Authorization": "Bearer alice@retail:all,hr", "X-Request-Id": "req-1"}
LOGISTICS = {"Authorization": "Bearer bob@logistics:all", "X-Request-Id": "req-2"}


def parse_sse(body: str) -> list[dict[str, Any]]:
    """Turn a raw text/event-stream body into [{event, id, data}, ...]."""
    events: list[dict[str, Any]] = []
    for block in body.strip().split("\n\n"):
        ev: dict[str, Any] = {}
        for line in block.splitlines():
            key, _, value = line.partition(": ")
            if key == "data":
                ev["data"] = json.loads(value)
            else:
                ev[key] = value
        if ev:
            events.append(ev)
    return events


@pytest.fixture()
def webhook() -> RecordingWebhook:
    return RecordingWebhook()


@pytest.fixture()
def app(webhook: RecordingWebhook):
    return build_app(model=FakeModel(), webhook=webhook)


@pytest.fixture()
def client(app):
    return TestClient(app)


# --- auth and tenant context -------------------------------------------------


def test_missing_bearer_is_401(client: TestClient) -> None:
    r = client.post("/v1/conversations/c1/messages", json={"text": "hi"}, headers={"X-Request-Id": "r"})
    assert r.status_code == 401


def test_token_without_tenant_is_401(client: TestClient) -> None:
    r = client.post("/v1/conversations/c1/messages", json={"text": "hi"}, headers={"Authorization": "Bearer a"})
    assert r.status_code == 401


def test_tenant_comes_from_the_token_not_from_headers(client: TestClient) -> None:
    # A retail token that also claims logistics in a header is served as retail.
    spoofed = {**RETAIL, "X-Tenant-Id": "logistics"}
    body = client.post("/v1/conversations/c9/messages", json={"text": "annual leave policy"}, headers=spoofed).json()
    assert body["lineage"]["evidence_chunk_ids"] == ["ret-pol-001"]
    assert client.get("/v1/conversations/c9/messages", headers=LOGISTICS).json() == []


# --- synchronous request path, streaming and non-streaming -------------------


def test_sse_stream_has_meta_citations_deltas_and_done(client: TestClient) -> None:
    with client.stream(
        "POST", "/v1/conversations/c1/messages?stream=true", json={"text": "annual leave policy"}, headers=RETAIL
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        assert r.headers["x-request-id"] == "req-1"
        events = parse_sse(r.read().decode())

    kinds = [e["event"] for e in events]
    assert kinds[0] == "meta"
    assert kinds[-1] == "done"
    assert "citation" in kinds and "delta" in kinds

    meta = events[0]["data"]
    assert meta["prompt_version"] == "assist.answer@3"
    assert meta["model"] == "fake-model-v0"
    assert meta["index_version"] == "idx-2026-01"

    deltas = [e for e in events if e["event"] == "delta"]
    assert [int(e["id"]) for e in deltas] == list(range(1, len(deltas) + 1)), "ids must be resumable"
    answer = "".join(e["data"]["text"] for e in deltas)
    assert answer == events[-1]["data"]["answer"]
    assert "[ret-pol-001]" in answer


def test_non_streaming_returns_same_answer_with_lineage(client: TestClient) -> None:
    r = client.post("/v1/conversations/c2/messages", json={"text": "annual leave policy"}, headers=RETAIL)
    assert r.status_code == 200
    body = r.json()
    lineage = body["lineage"]
    assert lineage["prompt_version"] == "3"
    assert lineage["index_version"] == "idx-2026-01"
    assert lineage["embedding_model"] == "fake-embedding-v0"
    assert lineage["policy_version"] == "policy-7"
    assert lineage["evidence_chunk_ids"] == ["ret-pol-001"]
    assert lineage["request_id"] == "req-1"

    history = client.get("/v1/conversations/c2/messages", headers=RETAIL).json()
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert history[1]["lineage"]["prompt_version"] == "3"


def test_retrieval_is_tenant_scoped_and_so_is_history(client: TestClient) -> None:
    retail = client.post("/v1/conversations/c3/messages", json={"text": "annual leave policy"}, headers=RETAIL).json()
    logistics = client.post(
        "/v1/conversations/c3/messages", json={"text": "annual leave policy"}, headers=LOGISTICS
    ).json()
    assert retail["lineage"]["evidence_chunk_ids"] == ["ret-pol-001"]
    assert logistics["lineage"]["evidence_chunk_ids"] == ["log-pol-001"]
    # Same conversation id, different tenant: each tenant sees only its own rows.
    assert len(client.get("/v1/conversations/c3/messages", headers=RETAIL).json()) == 2
    assert len(client.get("/v1/conversations/c3/messages", headers=LOGISTICS).json()) == 2


def test_retrieval_cache_key_includes_tenant_groups_and_index(app) -> None:
    from api_skeleton import RequestContext

    chat = app.state.chat
    a = chat.retrieval_cache_key(RequestContext("retail", "u", ("all",), "r"), "leave")
    b = chat.retrieval_cache_key(RequestContext("logistics", "u", ("all",), "r"), "leave")
    c = chat.retrieval_cache_key(RequestContext("retail", "u", ("all", "hr"), "r"), "leave")
    assert len({a, b, c}) == 3
    assert "idx-2026-01" in a


def test_model_budget_exhaustion_yields_error_event() -> None:
    slow = FakeModel(delay_s=0.05)
    app = build_app(model=slow)
    app.state.chat.total_budget_s = 0.12  # retrieve fits, the model stream does not
    client = TestClient(app)
    with client.stream(
        "POST", "/v1/conversations/c4/messages?stream=true", json={"text": "vpn runbook"}, headers=RETAIL
    ) as r:
        events = parse_sse(r.read().decode())
    assert events[-1]["event"] == "error"
    assert events[-1]["data"]["stage"] == "model_total"
    # A partial answer was streamed before the budget ran out: the UI must handle it.
    assert any(e["event"] == "delta" for e in events)


def test_budget_caps_stage_timeout_by_remaining_time() -> None:
    t = [0.0]
    budget = Budget(total_s=8.0, clock=lambda: t[0])
    assert budget.stage_timeout("retrieve") == 1.5
    t[0] = 7.0
    assert budget.stage_timeout("model_total") == pytest.approx(1.0)
    t[0] = 9.0
    assert budget.exhausted() and budget.stage_timeout("persist") == 0.0


# --- async job model -----------------------------------------------------------


def test_job_submit_returns_202_and_location(client: TestClient) -> None:
    r = client.post("/v1/jobs", json={"type": "ingest_document", "payload": {"text": "x" * 450}}, headers=RETAIL)
    assert r.status_code == 202
    job = r.json()
    assert job["state"] == "queued"
    assert r.headers["location"] == f"/v1/jobs/{job['id']}"
    assert client.get(r.headers["location"], headers=RETAIL).json()["state"] == "queued"


def test_worker_runs_job_to_success_and_calls_webhook(app, client: TestClient, webhook: RecordingWebhook) -> None:
    r = client.post(
        "/v1/jobs",
        json={"type": "ingest_document", "payload": {"text": "x" * 450}, "callback_url": "https://hooks.example/j"},
        headers=RETAIL,
    )
    job_id = r.json()["id"]
    done = asyncio.run(app.state.worker.run_once())
    assert done is not None and done.state.value == "succeeded"
    body = client.get(f"/v1/jobs/{job_id}", headers=RETAIL).json()
    assert body["state"] == "succeeded"
    assert body["result"] == {"document_id": body["result"]["document_id"], "chunks": 3}
    assert webhook.sent == [("https://hooks.example/j", {"job_id": job_id, "state": "succeeded"})]


def test_failed_job_is_retried_then_fails(app, client: TestClient) -> None:
    r = client.post("/v1/jobs", json={"type": "ingest_document", "payload": {}}, headers=RETAIL)
    job_id = r.json()["id"]
    for expected_state in ("queued", "queued", "failed"):
        asyncio.run(app.state.worker.run_once())
        assert client.get(f"/v1/jobs/{job_id}", headers=RETAIL).json()["state"] == expected_state
    body = client.get(f"/v1/jobs/{job_id}", headers=RETAIL).json()
    assert body["attempts"] == 3 and body["error"].startswith("ValueError")


def test_cancelled_job_is_skipped_by_worker(app, client: TestClient) -> None:
    job_id = client.post("/v1/jobs", json={"type": "evaluate", "payload": {"cases": 3}}, headers=RETAIL).json()["id"]
    assert client.post(f"/v1/jobs/{job_id}/cancel", headers=RETAIL).json()["state"] == "cancelled"
    asyncio.run(app.state.worker.run_once())
    assert client.get(f"/v1/jobs/{job_id}", headers=RETAIL).json()["state"] == "cancelled"


def test_idempotency_key_returns_original_job(client: TestClient) -> None:
    headers = {**RETAIL, "Idempotency-Key": "ingest-42"}
    first = client.post("/v1/jobs", json={"type": "evaluate", "payload": {"cases": 1}}, headers=headers)
    second = client.post("/v1/jobs", json={"type": "evaluate", "payload": {"cases": 1}}, headers=headers)
    assert first.status_code == 202 and second.status_code == 200
    assert first.json()["id"] == second.json()["id"]
    # Same key in another tenant is a different job.
    other = client.post("/v1/jobs", json={"type": "evaluate"}, headers={**LOGISTICS, "Idempotency-Key": "ingest-42"})
    assert other.status_code == 202 and other.json()["id"] != first.json()["id"]


def test_job_is_invisible_to_other_tenants(client: TestClient) -> None:
    job_id = client.post("/v1/jobs", json={"type": "evaluate"}, headers=RETAIL).json()["id"]
    assert client.get(f"/v1/jobs/{job_id}", headers=LOGISTICS).status_code == 404
    assert client.post(f"/v1/jobs/{job_id}/cancel", headers=LOGISTICS).status_code == 404
    assert client.get("/v1/jobs/does-not-exist", headers=RETAIL).status_code == 404


def test_job_events_stream_closes_on_terminal_state(app, client: TestClient) -> None:
    job_id = client.post("/v1/jobs", json={"type": "evaluate", "payload": {"cases": 2}}, headers=RETAIL).json()["id"]
    asyncio.run(app.state.worker.run_once())
    with client.stream("GET", f"/v1/jobs/{job_id}/events", headers=RETAIL) as r:
        events = parse_sse(r.read().decode())
    assert [e["data"]["state"] for e in events] == ["succeeded"]


# --- deployment files ------------------------------------------------------------


def test_compose_and_schema_files_exist_and_name_the_core_services() -> None:
    here = pathlib.Path(__file__).parent
    compose = (here / "docker-compose.yml").read_text()
    for service in ("api:", "worker:", "postgres:", "redis:", "otel-collector:"):
        assert service in compose
    schema = (here / "schema.sql").read_text().lower()
    for table in ("conversations", "messages", "jobs", "documents", "chunks", "evaluation_runs", "audit_events"):
        assert f"create table {table}" in schema
