# path: book/projects/examples/ch28/test_bridge.py
"""Chapter 28's job model on Chapter 29's queue and worker, through reliability_bridge. Offline."""
from __future__ import annotations

from typing import Any

import pytest

reliability = pytest.importorskip("reliability")

from reliability import InMemoryJobQueue, ManualClock  # noqa: E402

from api_skeleton import (  # noqa: E402
    InMemoryJobRepo,
    Job,
    JobService,
    JobState,
    RecordingWebhook,
    RequestContext,
    evaluate_handler,
    ingest_document_handler,
)
from reliability_bridge import KIND, ReliabilityJobQueue, make_worker, reconcile_dead_letters  # noqa: E402

CTX = RequestContext("retail", "alice", ("all",), "req-1")


@pytest.fixture()
def env() -> dict[str, Any]:
    clock = ManualClock()
    repo = InMemoryJobRepo()
    delivery = InMemoryJobQueue(clock=clock)
    webhook = RecordingWebhook()
    calls = {"n": 0}

    async def flaky_handler(job: Job) -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("upstream reset")  # transient: retried with backoff
        return {"ok": True}

    handlers = {"ingest_document": ingest_document_handler, "evaluate": evaluate_handler, "flaky": flaky_handler}
    worker = make_worker(delivery, repo, handlers, webhook, queue_clock=clock, clock=clock, sleep=clock.sleep)
    jobs = JobService(repo=repo, queue=ReliabilityJobQueue(delivery, repo))
    return {"clock": clock, "repo": repo, "delivery": delivery, "webhook": webhook, "worker": worker, "jobs": jobs}


def test_success_is_mirrored_into_the_jobs_table_and_webhook(env) -> None:
    job, _ = env["jobs"].submit(CTX, "ingest_document", {"text": "x" * 450}, None, "https://hooks.example/j")
    result = env["worker"].run_once()
    assert result is not None and result.outcome.value == "succeeded"
    row = env["repo"].get(job.id)
    assert row.state is JobState.SUCCEEDED and row.attempts == 1 and row.result["chunks"] == 3
    assert env["webhook"].sent == [("https://hooks.example/j", {"job_id": job.id, "state": "succeeded"})]


def test_deterministic_failure_is_dead_lettered_after_one_attempt(env) -> None:
    job, _ = env["jobs"].submit(CTX, "ingest_document", {}, None, None)  # ValueError: payload.text missing
    assert env["worker"].run_once().outcome.value == "dead"
    row = env["repo"].get(job.id)
    assert row.state is JobState.FAILED and row.attempts == 1 and row.error.startswith("ValueError")
    assert [d.payload["job_id"] for d in env["delivery"].dead_letters()] == [job.id]


def test_transient_failure_is_retried_with_backoff(env) -> None:
    job, _ = env["jobs"].submit(CTX, "flaky", {}, None, None)
    assert env["worker"].run_once().outcome.value == "retry"
    assert env["repo"].get(job.id).state is JobState.QUEUED
    assert env["worker"].run_once() is None  # backoff: not yet visible again
    env["clock"].advance(60.0)
    assert env["worker"].run_once().outcome.value == "succeeded"
    row = env["repo"].get(job.id)
    assert row.state is JobState.SUCCEEDED and row.attempts == 2


def test_cancelled_job_is_acked_without_running(env) -> None:
    job, _ = env["jobs"].submit(CTX, "evaluate", {"cases": 3}, None, None)
    env["jobs"].cancel(CTX, job.id)
    assert env["worker"].run_once().outcome.value == "succeeded"  # delivery done; the work was skipped
    assert env["repo"].get(job.id).state is JobState.CANCELLED
    assert env["repo"].get(job.id).result is None


def test_crash_on_final_attempt_is_reconciled_from_dead_letters(env) -> None:
    job, _ = env["jobs"].submit(CTX, "evaluate", {"cases": 1}, None, "https://hooks.example/j")
    env["repo"].get(job.id).state = JobState.RUNNING  # what the wrapper wrote before each crash
    for _ in range(job.max_attempts):  # three workers lease the job and die without ack or nack
        assert env["delivery"].lease(visibility_timeout_s=10.0) is not None
        env["clock"].advance(11.0)
    assert reconcile_dead_letters(env["delivery"], env["repo"], env["webhook"]) == [job.id]
    row = env["repo"].get(job.id)
    assert row.state is JobState.FAILED and row.error == "lease expired on final attempt"
    assert env["webhook"].sent[-1][1] == {"job_id": job.id, "state": "failed"}


def test_submission_port_is_idempotent_and_refuses_bare_dequeue(env) -> None:
    port: ReliabilityJobQueue = env["jobs"].queue
    job, _ = env["jobs"].submit(CTX, "evaluate", {}, None, None)
    port.enqueue(job.id)  # a duplicate enqueue, e.g. from a retried submit
    assert port.depth() == 1
    assert env["delivery"].lease().job.kind == KIND
    with pytest.raises(NotImplementedError):
        port.dequeue()
