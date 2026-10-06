# path: book/projects/reliability/tests/test_hardening.py
"""Edge cases behind the package's guarantees: timeouts, stragglers, cancellation, quotas, headers."""
import asyncio

import pytest
from aie_core import RetryPolicy
from aie_core.llm.errors import ProviderUnavailableError, TimeoutError as LLMTimeoutError

from reliability import CircuitBreaker, CircuitState, Deadline, ManualClock, RetryBudget, call_with_retry
from reliability.admission import TokenBucket


def _breaker(clock, **kw):
    return CircuitBreaker("dep", min_calls=4, failure_rate_threshold=0.5, window_s=10.0, buckets=10, open_s=30.0,
                          half_open_max_calls=2, clock=clock, **kw)


def test_a_timeout_is_not_retried_when_the_policy_forbids_it():
    calls = []

    def times_out():
        calls.append(1)
        raise LLMTimeoutError("slow")

    with pytest.raises(LLMTimeoutError):
        call_with_retry(times_out, policy=RetryPolicy(max_attempts=3, retry_on_timeout=False), sleep=lambda s: None)
    assert len(calls) == 1


def test_stragglers_from_before_the_trip_do_not_close_a_half_open_circuit(clock):
    b = _breaker(clock)
    gens = [b.admit() for _ in range(5)]                 # five calls admitted while closed
    b.record_failure(ProviderUnavailableError("503"), generation=gens[0])
    b.record_failure(ProviderUnavailableError("503"), generation=gens[1])
    b.record_failure(ProviderUnavailableError("503"), generation=gens[2])
    b.record_success(generation=gens[3])                 # 3 of 4 failed: the circuit opens
    assert b.state is CircuitState.OPEN
    clock.advance(31)
    assert b.state is CircuitState.HALF_OPEN
    b.record_success(generation=gens[4])                 # a late success from before the trip
    assert b.state is CircuitState.HALF_OPEN             # still waiting for real probes


def test_cancellation_is_not_a_dependency_failure(clock):
    b = _breaker(clock)

    async def cancelled():
        raise asyncio.CancelledError

    for _ in range(6):
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(b.acall(cancelled))
    assert b.state is CircuitState.CLOSED


def test_an_oversized_request_leaves_the_bucket_in_debt():
    clock = ManualClock()
    bucket = TokenBucket(60_000, 1_000, clock)
    assert bucket.wait_for(1_000_000) == 0.0            # a full bucket admits it once
    bucket.take(1_000_000)
    assert bucket.wait_for(1) > 900                     # ... and then makes everyone wait it off


@pytest.mark.parametrize("header", ["inf", "nan", "1e400"])
def test_non_finite_deadline_headers_fall_back_to_the_default(header):
    clock = ManualClock()
    d = Deadline.from_header(header, default_s=5.0, clock=clock)
    assert d.remaining() == pytest.approx(5.0)


def test_the_retry_budget_floor_works_below_one_retry_per_second():
    clock = ManualClock()
    budget = RetryBudget(ratio=0.0, min_retries_per_s=0.5, clock=clock)
    clock.advance(100)
    assert budget.try_spend()


def test_an_expired_idempotency_key_admits_a_new_job():
    fakeredis = pytest.importorskip("fakeredis")
    pytest.importorskip("lupa")
    from reliability import RedisJobQueue
    clock = ManualClock()
    r = fakeredis.FakeRedis()
    q = RedisJobQueue(r, namespace="t", clock=clock, result_ttl_s=60)
    first = q.enqueue("k", {}, idempotency_key="K", tenant_id="retail")
    q.ack(q.lease(), result={})
    r.delete(q._jk(first.id))                            # the result TTL elapsed
    second = q.enqueue("k", {}, idempotency_key="K", tenant_id="retail")
    assert second.id != first.id
