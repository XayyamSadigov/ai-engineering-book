# path: book/projects/examples/ch28/test_hardening.py
"""Edge cases behind the architecture's guarantees: cancellation, ownership, idempotency, egress."""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api_skeleton import (FakeModel, InMemoryJobRepo, Job, JobService, JobState, RecordingWebhook, RequestContext,
                          build_app)

RETAIL = {"Authorization": "Bearer alice@retail:all,hr", "X-Request-Id": "req-1"}
MALLORY = {"Authorization": "Bearer mallory@retail:all", "X-Request-Id": "req-2"}


@pytest.fixture()
def app():
    return build_app(model=FakeModel(), webhook=RecordingWebhook())


def test_a_cancel_during_the_run_is_not_overwritten(app) -> None:
    jobs, worker = app.state.jobs, app.state.worker
    ctx = RequestContext("retail", "alice", ("all",), "r")

    async def cancels_itself(job: Job) -> dict[str, Any]:
        jobs.cancel(ctx, job.id)
        return {"done": True}

    worker.handlers["evaluate"] = cancels_itself
    job, _ = jobs.submit(ctx, "evaluate", {}, None, None)
    asyncio.run(worker.run_once())
    assert jobs.get(ctx, job.id).state is JobState.CANCELLED


def test_another_user_in_the_tenant_cannot_read_a_conversation(app) -> None:
    client = TestClient(app)
    client.post("/v1/conversations/c-1/messages", json={"text": "my salary is 9000"}, headers=RETAIL)
    history = client.get("/v1/conversations/c-1/messages", headers=MALLORY)
    assert history.status_code in (200, 404) and "9000" not in history.text


def test_concurrent_submissions_with_one_key_create_one_job() -> None:
    repo = InMemoryJobRepo()
    original = repo.find_by_idempotency_key

    def slow_lookup(*args: Any) -> Any:
        time.sleep(0.01)
        return original(*args)

    repo.find_by_idempotency_key = slow_lookup  # type: ignore[method-assign]
    service = JobService(repo=repo, queue=type("Q", (), {"enqueue": lambda self, i: None})())
    ctx = RequestContext("retail", "alice", ("all",), "r")
    threads = [threading.Thread(target=service.submit, args=(ctx, "evaluate", {}, "K", None)) for _ in range(5)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len({j.id for j in repo._rows.values()}) == 1


@pytest.mark.parametrize("url", ["http://hooks.example/j", "https://169.254.169.254/latest/meta-data",
                                 "https://10.0.0.5/x", "https://localhost/x"])
def test_callback_urls_must_be_public_https(app, url) -> None:
    r = TestClient(app).post("/v1/jobs", json={"type": "evaluate", "payload": {}, "callback_url": url}, headers=RETAIL)
    assert r.status_code == 422


def test_job_payloads_are_bounded(app) -> None:
    r = TestClient(app).post("/v1/jobs", json={"type": "evaluate", "payload": {"text": "x" * 100_000}}, headers=RETAIL)
    assert r.status_code == 422
