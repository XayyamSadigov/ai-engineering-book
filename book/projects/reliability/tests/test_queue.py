# path: book/projects/reliability/tests/test_queue.py
"""One contract, every adapter: the in-memory queue, Redis via fakeredis (offline), real Redis (integration)."""
import os

import pytest

from reliability import InMemoryJobQueue, JobState, ManualClock, RedisJobQueue
from reliability.queue import RetryPolicy

BACKOFF = RetryPolicy(base_delay_s=10.0, max_delay_s=100.0, jitter=False)


def _fakeredis_queue(clock: ManualClock) -> RedisJobQueue:
    fakeredis = pytest.importorskip("fakeredis")
    pytest.importorskip("lupa")
    return RedisJobQueue(fakeredis.FakeRedis(), namespace="test", clock=clock, backoff=BACKOFF)


def _real_redis_queue(clock: ManualClock) -> RedisJobQueue:
    url = os.getenv("REDIS_URL")
    if not url:
        pytest.skip("REDIS_URL not set")
    redis = pytest.importorskip("redis")
    client = redis.Redis.from_url(url)
    client.flushdb()
    return RedisJobQueue(client, namespace="it", clock=clock, backoff=BACKOFF)


@pytest.fixture(params=[
    "memory",
    "fakeredis",
    pytest.param("redis", marks=pytest.mark.integration),
])
def queue_and_clock(request):
    clock = ManualClock()
    if request.param == "memory":
        return InMemoryJobQueue(clock=clock, backoff=BACKOFF), clock
    if request.param == "fakeredis":
        return _fakeredis_queue(clock), clock
    return _real_redis_queue(clock), clock


def test_enqueue_is_idempotent_per_tenant(queue_and_clock):
    q, _ = queue_and_clock
    a = q.enqueue("ingest_document", {"doc": 1}, idempotency_key="k1", tenant_id="retail")
    b = q.enqueue("ingest_document", {"doc": 1}, idempotency_key="k1", tenant_id="retail")
    c = q.enqueue("ingest_document", {"doc": 1}, idempotency_key="k1", tenant_id="logistics")
    assert a.id == b.id and a.id != c.id
    assert q.depth() == 2


def test_lease_hides_job_and_ack_completes(queue_and_clock):
    q, _ = queue_and_clock
    job = q.enqueue("send_reply", {"ticket_id": "T-1"})
    lease = q.lease(visibility_timeout_s=30)
    assert lease is not None and lease.job.id == job.id and lease.job.attempts == 1
    assert q.lease(30) is None                      # invisible while leased
    assert q.ack(lease, {"status": "sent"})
    done = q.get(job.id)
    assert done.state is JobState.SUCCEEDED and done.result == {"status": "sent"}


def test_redelivery_after_lease_expiry(queue_and_clock):
    q, clock = queue_and_clock
    q.enqueue("ingest_document", {"doc": 7})
    first = q.lease(visibility_timeout_s=30)
    clock.advance(31)                               # worker died: no ack, no heartbeat
    second = q.lease(visibility_timeout_s=30)
    assert second is not None and second.job.id == first.job.id
    assert second.job.attempts == 2
    assert not q.ack(first)                         # the dead worker's late ack is refused
    assert q.ack(second)


def test_heartbeat_extends_lease(queue_and_clock):
    q, clock = queue_and_clock
    q.enqueue("long_job", {})
    lease = q.lease(visibility_timeout_s=30)
    clock.advance(25)
    assert q.extend(lease, 30)
    clock.advance(25)
    assert q.lease(30) is None                      # still ours
    assert q.ack(lease)


def test_nack_backs_off_then_redelivers(queue_and_clock):
    q, clock = queue_and_clock
    q.enqueue("classify", {}, max_attempts=3)
    lease = q.lease(30)
    assert q.nack(lease, "ProviderUnavailableError: 503") is JobState.QUEUED
    assert q.lease(30) is None                      # backoff: 10 s for attempt 1
    clock.advance(10.5)
    again = q.lease(30)
    assert again is not None and again.job.attempts == 2 and again.job.last_error.startswith("Provider")


def test_dead_letter_after_max_attempts_and_redrive(queue_and_clock):
    q, clock = queue_and_clock
    job = q.enqueue("classify", {}, max_attempts=2)
    for _ in range(2):
        clock.advance(200)
        lease = q.lease(30)
        state = q.nack(lease, "boom")
    assert state is JobState.DEAD
    assert [j.id for j in q.dead_letters()] == [job.id]
    assert q.depth() == 0
    assert q.redrive(job.id)
    assert q.lease(30).job.attempts == 1


def test_non_retryable_goes_straight_to_dlq(queue_and_clock):
    q, _ = queue_and_clock
    q.enqueue("ingest_document", {"text": ""}, max_attempts=5)
    lease = q.lease(30)
    assert q.nack(lease, "PermanentJobError: payload has no text", retryable=False) is JobState.DEAD


def test_poison_job_that_kills_workers_is_dead_lettered(queue_and_clock):
    q, clock = queue_and_clock
    job = q.enqueue("ingest_document", {"pdf": "crashes-the-parser"}, max_attempts=3)
    for _ in range(3):
        assert q.lease(30) is not None
        clock.advance(31)                           # each worker dies mid-job
    assert q.lease(30) is None
    dead = q.dead_letters()
    assert [j.id for j in dead] == [job.id] and "lease expired" in dead[0].last_error


def test_release_returns_job_without_spending_an_attempt(queue_and_clock):
    q, _ = queue_and_clock
    q.enqueue("ingest_document", {})
    lease = q.lease(30)
    assert q.release(lease)
    again = q.lease(30)
    assert again is not None and again.job.attempts == 1


def test_delayed_enqueue(queue_and_clock):
    q, clock = queue_and_clock
    q.enqueue("digest", {}, delay_s=60)
    assert q.lease(30) is None
    clock.advance(60)
    assert q.lease(30) is not None
