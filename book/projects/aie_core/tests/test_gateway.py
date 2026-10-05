# path: book/projects/aie_core/tests/test_gateway.py
import random
import threading

import pytest

from aie_core.llm.errors import (
    InvalidRequestError,
    MalformedResponseError,
    ProviderUnavailableError,
    RateLimitError,
    TimeoutError,
)
from aie_core.llm.gateway import (
    SPAN_NAME,
    InMemoryResponseCache,
    ModelGateway,
    PricingTable,
    RateLimiter,
    RetryPolicy,
    cache_key,
)
from aie_core.llm.providers import FakeLLM
from aie_core.llm.types import Completion, CompletionRequest, Message, Usage
from aie_core.observability import InMemoryTracer

from ._helpers import FakeClock

REQ = CompletionRequest(messages=[Message.system("You are Northwind Assist."), Message.user("hello")])


def gateway(primary, clock: FakeClock, **kwargs) -> ModelGateway:
    kwargs.setdefault("retry", RetryPolicy(max_attempts=3, base_delay_s=0.5, max_delay_s=8.0, jitter=False))
    return ModelGateway(primary, sleep=clock.sleep, asleep=clock.asleep, clock=clock, rng=random.Random(0), **kwargs)


# ------------------------------------------------------------------ retry policy
def test_backoff_doubles_and_caps_without_jitter():
    p = RetryPolicy(base_delay_s=0.5, max_delay_s=3.0, jitter=False)
    rng = random.Random(1)
    assert [p.delay_for(a, None, rng) for a in (1, 2, 3, 4)] == [0.5, 1.0, 2.0, 3.0]


def test_full_jitter_stays_within_window():
    p = RetryPolicy(base_delay_s=1.0, max_delay_s=8.0, jitter=True)
    rng = random.Random(42)
    for attempt in (1, 2, 3):
        for _ in range(50):
            d = p.delay_for(attempt, None, rng)
            assert 0.0 <= d <= min(8.0, 2 ** (attempt - 1))


def test_retry_after_hint_wins_over_backoff():
    p = RetryPolicy(base_delay_s=0.5, jitter=False)
    assert p.delay_for(1, 4.0, random.Random(0)) == 4.0


# ------------------------------------------------------------------ gateway retries
def test_retries_retryable_error_then_succeeds(clock):
    llm = FakeLLM(responses=[ProviderUnavailableError("down"), RateLimitError("slow", retry_after_s=2.0), "recovered"])
    tracer = InMemoryTracer()
    gw = gateway(llm, clock, tracer=tracer)
    c = gw.complete(REQ)
    assert c.text == "recovered" and len(llm.requests) == 3
    assert clock.sleeps == [0.5, 2.0]  # backoff, then provider hint
    assert c.raw["attempt"] == 3
    spans = tracer.find(SPAN_NAME)
    assert [s.attributes["attempt"] for s in spans] == [1, 2, 3]
    assert spans[0].status == "error" and spans[2].status == "ok"
    assert spans[2].attributes["finish_reason"] == "stop" and spans[2].attributes["cache_hit"] is False


def test_non_retryable_error_raises_immediately_without_fallback(clock):
    primary = FakeLLM(responses=[InvalidRequestError("bad schema")])
    fallback = FakeLLM(responses=["should not run"])
    gw = gateway(primary, clock, fallbacks=[fallback])
    with pytest.raises(InvalidRequestError):
        gw.complete(REQ)
    assert fallback.requests == [] and clock.sleeps == []


def test_malformed_response_is_not_retried(clock):
    primary = FakeLLM(responses=[MalformedResponseError("garbage"), "never"])
    gw = gateway(primary, clock)
    with pytest.raises(MalformedResponseError):
        gw.complete(REQ)
    assert len(primary.requests) == 1


def test_retry_budget_exhausted_falls_back_to_next_client(clock):
    primary = FakeLLM(responses=[ProviderUnavailableError("a"), ProviderUnavailableError("b"), ProviderUnavailableError("c")], provider="p1")
    fallback = FakeLLM(responses=["from fallback"], provider="p2")
    gw = gateway(primary, clock, fallbacks=[fallback])
    c = gw.complete(REQ)
    assert c.text == "from fallback" and c.provider == "p2"
    assert len(primary.requests) == 3 and len(fallback.requests) == 1


def test_all_clients_fail_raises_last_error(clock):
    primary = FakeLLM(responses=[ProviderUnavailableError("a")] * 3)
    fallback = FakeLLM(responses=[RateLimitError("b")] * 3)
    gw = gateway(primary, clock, fallbacks=[fallback])
    with pytest.raises(RateLimitError):
        gw.complete(REQ)


def test_deadline_bounds_retries_and_propagates_remaining_budget(clock):
    primary = FakeLLM(responses=[ProviderUnavailableError("a")] * 3)
    fallback = FakeLLM(responses=["late"])
    gw = gateway(primary, clock, fallbacks=[fallback], retry=RetryPolicy(max_attempts=5, base_delay_s=4.0, jitter=False))
    req = REQ.model_copy(update={"timeout_s": 5.0})
    # attempt 1 fails -> delay 4 fits (t=4) -> attempt 2 fails -> delay 8 would exceed deadline -> fallback
    c = gw.complete(req)
    assert c.text == "late"
    assert len(primary.requests) == 2
    assert fallback.requests[0].timeout_s == pytest.approx(1.0)  # 5s budget minus the 4s slept
    assert primary.requests[0].timeout_s == pytest.approx(5.0)


def test_timeout_not_retried_when_policy_forbids(clock):
    primary = FakeLLM(responses=[TimeoutError("slow"), "x"])
    gw = gateway(primary, clock, retry=RetryPolicy(max_attempts=3, retry_on_timeout=False, jitter=False))
    with pytest.raises(TimeoutError):
        gw.complete(REQ)
    assert len(primary.requests) == 1


# ------------------------------------------------------------------ cache
def test_cache_key_ignores_metadata_and_timeout_but_not_temperature():
    base = cache_key(REQ, "m")
    assert cache_key(REQ.model_copy(update={"metadata": {"user": "x"}, "timeout_s": 3}), "m") == base
    assert cache_key(REQ.model_copy(update={"temperature": 0.7}), "m") != base
    assert cache_key(REQ, "other-model") != base
    assert cache_key(REQ.model_copy(update={"messages": [Message.user("hello")]}), "m") != base


def test_cache_hit_skips_provider_and_is_traced(clock):
    llm = FakeLLM(responses=["first", "second"])
    tracer = InMemoryTracer()
    gw = gateway(llm, clock, cache=InMemoryResponseCache(), tracer=tracer)
    a = gw.complete(REQ)
    b = gw.complete(REQ)
    assert a.text == b.text == "first" and len(llm.requests) == 1
    assert b.raw["cache_hit"] is True and a.raw["cache_hit"] is False
    assert tracer.spans[-1].attributes["cache_hit"] is True and tracer.spans[-1].attributes["attempt"] == 0


def test_cache_only_for_deterministic_requests_unless_allowed(clock):
    llm = FakeLLM(responses=["a", "b", "c", "d"])
    gw = gateway(llm, clock, cache=InMemoryResponseCache())
    hot = REQ.model_copy(update={"temperature": 0.8})
    assert gw.complete(hot).text == "a" and gw.complete(hot).text == "b"
    allowed = hot.model_copy(update={"metadata": {"cache": True}})
    assert gw.complete(allowed).text == "c" and gw.complete(allowed).text == "c"
    disabled = REQ.model_copy(update={"metadata": {"cache": False}})
    assert gw.complete(disabled).text == "d"


def test_cache_ttl_and_lru_eviction(clock):
    cache = InMemoryResponseCache(max_size=2, clock=clock)
    done = Completion(message=Message.assistant("x"))
    cache.set("k1", done, ttl_s=10)
    cache.set("k2", done)
    cache.set("k3", done)  # evicts k1 (oldest)
    assert cache.get("k1") is None and cache.get("k2") is not None
    cache.set("k4", done, ttl_s=5)
    clock.advance(6)
    assert cache.get("k4") is None


# ------------------------------------------------------------------ pricing
def test_pricing_table_cost_and_cached_discount():
    table = PricingTable({"m": {"input_per_1m": 1.0, "output_per_1m": 4.0, "cached_input_per_1m": 0.1}}, default={"input_per_1m": 2.0, "output_per_1m": 2.0})
    usage = Usage(input_tokens=1_000_000, output_tokens=500_000, cached_input_tokens=500_000)
    assert table.cost_usd("m", usage) == pytest.approx(0.5 * 1.0 + 0.5 * 0.1 + 0.5 * 4.0)
    assert table.cost_usd("m-2026-01", usage) == table.cost_usd("m", usage)  # prefix match
    assert table.cost_usd("unknown", Usage(input_tokens=1_000_000)) == pytest.approx(2.0)
    assert PricingTable().cost_usd("anything", usage) == 0.0


def test_gateway_attaches_cost(clock):
    llm = FakeLLM(handler=lambda req: Completion(message=Message.assistant("hi"), usage=Usage(input_tokens=1000, output_tokens=100), model="m"))
    tracer = InMemoryTracer()
    gw = gateway(llm, clock, pricing=PricingTable({"m": {"input_per_1m": 10.0, "output_per_1m": 30.0}}), tracer=tracer)
    c = gw.complete(REQ)
    assert c.raw["cost_usd"] == pytest.approx(0.01 + 0.003)
    assert tracer.spans[-1].attributes["cost_usd"] == pytest.approx(0.013)
    assert tracer.spans[-1].attributes["input_tokens"] == 1000


# ------------------------------------------------------------------ rate limiter
def test_rate_limiter_requests_per_minute(clock):
    rl = RateLimiter(requests_per_minute=60, clock=clock, sleep=clock.sleep)
    # bucket starts full: 60 immediate admissions, then one per second
    for _ in range(60):
        rl.acquire()
    assert clock.sleeps == []
    rl.acquire()
    assert clock.sleeps == [pytest.approx(1.0)]


def test_rate_limiter_tokens_per_minute_and_burst(clock):
    rl = RateLimiter(tokens_per_minute=6000, burst=1000, clock=clock, sleep=clock.sleep)
    rl.acquire(tokens=1000)
    rl.acquire(tokens=500)  # needs 500 tokens at 100/s -> 5s
    assert clock.sleeps == [pytest.approx(5.0)]
    assert rl.try_acquire(tokens=1) is False


def test_rate_limiter_respects_deadline(clock):
    rl = RateLimiter(requests_per_minute=60, burst=1, clock=clock, sleep=clock.sleep)
    rl.acquire()
    with pytest.raises(TimeoutError):
        rl.acquire(deadline=clock() + 0.5)


def test_gateway_waits_on_limiter(clock):
    rl = RateLimiter(requests_per_minute=60, burst=1, clock=clock, sleep=clock.sleep)
    gw = gateway(FakeLLM(responses=["a", "b"]), clock, rate_limiter=rl)
    gw.complete(REQ)
    gw.complete(REQ)
    assert clock.sleeps == [pytest.approx(1.0)]


# ------------------------------------------------------------------ concurrency
def test_semaphore_limits_in_flight_calls(clock):
    import time as _time

    active = 0
    peak = 0
    lock = threading.Lock()

    def handler(req):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        _time.sleep(0.02)  # hold the slot long enough for the other threads to queue up
        with lock:
            active -= 1
        return "ok"

    gw = gateway(FakeLLM(handler=handler), clock, max_concurrency=2)
    threads = [threading.Thread(target=gw.complete, args=(REQ,)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert peak == 2


# ------------------------------------------------------------------ streaming
def test_stream_retries_before_first_event_then_relays(clock):
    llm = FakeLLM(responses=[ProviderUnavailableError("x"), "streamed text"], chunk_size=5)
    tracer = InMemoryTracer()
    gw = gateway(llm, clock, tracer=tracer)
    events = list(gw.stream(REQ))
    assert "".join(e.text for e in events if e.type == "text_delta") == "streamed text"
    assert events[-1].type == "done" and clock.sleeps == [0.5]
    assert tracer.spans[-1].attributes["stream"] is True and tracer.spans[-1].attributes["output_tokens"] > 0


def test_stream_falls_back_when_primary_unavailable(clock):
    primary = FakeLLM(responses=[ProviderUnavailableError("x")] * 3, provider="p1")
    fallback = FakeLLM(responses=["ok"], provider="p2")
    gw = gateway(primary, clock, fallbacks=[fallback])
    events = list(gw.stream(REQ))
    assert events[0].type == "text_delta" and len(fallback.requests) == 1


def test_stream_does_not_retry_invalid_request(clock):
    gw = gateway(FakeLLM(responses=[InvalidRequestError("bad")]), clock, fallbacks=[FakeLLM()])
    with pytest.raises(InvalidRequestError):
        list(gw.stream(REQ))


# ------------------------------------------------------------------ async
async def test_async_complete_retries_and_falls_back(clock):
    primary = FakeLLM(responses=[RateLimitError("rl", retry_after_s=1.0), ProviderUnavailableError("x"), ProviderUnavailableError("y")])
    fallback = FakeLLM(responses=["async fallback"])
    gw = gateway(primary, clock, fallbacks=[fallback])
    c = await gw.acomplete(REQ)
    assert c.text == "async fallback" and clock.sleeps == [1.0, 1.0]


async def test_async_stream_and_cache(clock):
    gw = gateway(FakeLLM(responses=["abc", "def"]), clock, cache=InMemoryResponseCache())
    events = [e async for e in gw.astream(REQ)]
    assert "".join(e.text for e in events if e.type == "text_delta") == "abc"
    assert (await gw.acomplete(REQ)).text == "def"
    assert (await gw.acomplete(REQ)).text == "def"  # served from cache


# ------------------------------------------------------------------ deadline semantics
def test_deadline_exceeded_surfaces_as_timeout_with_cause(clock):
    primary = FakeLLM(responses=[ProviderUnavailableError("a")] * 5)
    gw = gateway(primary, clock, retry=RetryPolicy(max_attempts=5, base_delay_s=1.0, jitter=False))
    with pytest.raises(TimeoutError) as info:
        gw.complete(REQ.model_copy(update={"timeout_s": 2.5}))
    assert isinstance(info.value.__cause__, ProviderUnavailableError)
    assert info.value.retryable is False
    assert len(primary.requests) == 2  # t=0 fail, sleep 1, t=1 fail, sleep 2 would pass t=2.5


def test_span_records_queue_time_and_stream_usage(clock):
    tracer = InMemoryTracer()
    gw = gateway(FakeLLM(responses=["abc def"]), clock, tracer=tracer)
    list(gw.stream(REQ))
    span = tracer.find(SPAN_NAME)[-1]
    assert span.attributes["stream"] is True and span.attributes["output_tokens"] >= 1
    assert span.attributes["finish_reason"] == "stop" and "queue_ms" in span.attributes
    assert span.status == "ok"


def test_abandoned_stream_releases_semaphore(clock):
    gw = gateway(FakeLLM(responses=["long text " * 20, "second"], chunk_size=4), clock, max_concurrency=1)
    it = gw.stream(REQ)
    next(it)
    it.close()  # consumer walks away mid-stream
    assert gw.complete(REQ).text == "second"  # would deadlock if the slot leaked


async def test_async_stream_falls_back(clock):
    primary = FakeLLM(responses=[ProviderUnavailableError("x")] * 3, provider="p1")
    fallback = FakeLLM(responses=["ok"], provider="p2")
    tracer = InMemoryTracer()
    gw = gateway(primary, clock, fallbacks=[fallback], tracer=tracer)
    events = [e async for e in gw.astream(REQ)]
    assert events[0].text == "ok" and len(fallback.requests) == 1
    assert [s.status for s in tracer.find(SPAN_NAME)] == ["error", "error", "error", "ok"]


# ------------------------------------------------------------------ cache-hit cost accounting and trace linkage
def test_cache_hit_reports_zero_cost_and_avoided_cost(clock):
    llm = FakeLLM(handler=lambda req: Completion(message=Message.assistant("hi"), usage=Usage(input_tokens=1_000_000), model="m"))
    tracer = InMemoryTracer()
    gw = gateway(llm, clock, cache=InMemoryResponseCache(), pricing=PricingTable({"m": {"input_per_1m": 2.0, "output_per_1m": 0.0}}), tracer=tracer)
    first = gw.complete(REQ)
    second = gw.complete(REQ)
    assert first.raw["cost_usd"] == pytest.approx(2.0) and first.raw["avoided_cost_usd"] == 0.0
    assert second.raw["cost_usd"] == 0.0 and second.raw["avoided_cost_usd"] == pytest.approx(2.0)
    assert sum(c.raw["cost_usd"] for c in (first, second)) == pytest.approx(2.0)
    hit_span = tracer.spans[-1]
    assert hit_span.attributes["cache_hit"] is True
    assert hit_span.attributes["cost_usd"] == 0.0 and hit_span.attributes["avoided_cost_usd"] == pytest.approx(2.0)


def test_gateway_spans_nest_under_caller_span(clock):
    tracer = InMemoryTracer()
    gw = gateway(FakeLLM(responses=[ProviderUnavailableError("x"), "ok", "streamed"]), clock, tracer=tracer)
    with tracer.span("request.handle") as parent:
        gw.complete(REQ)
        list(gw.stream(REQ))
    llm_spans = tracer.find(SPAN_NAME)
    assert len(llm_spans) == 3  # failed attempt, successful attempt, stream
    assert all(s.trace_id == parent.trace_id for s in llm_spans)
    assert all(s.parent_span_id == parent.span_id for s in llm_spans)
    assert llm_spans[-1].attributes["stream"] is True


# ------------------------------------------------------------------ cache scope (tenant isolation)
def test_cache_scope_partitions_entries_and_leaves_unscoped_keys_unchanged(clock):
    retail = REQ.model_copy(update={"metadata": {"cache_scope": "tenant:retail"}})
    logistics = REQ.model_copy(update={"metadata": {"cache_scope": "tenant:logistics"}})
    assert cache_key(retail, "m") != cache_key(logistics, "m") != cache_key(REQ, "m")
    # adding the feature must not invalidate existing unscoped entries
    assert cache_key(REQ.model_copy(update={"metadata": {"user": "x"}}), "m") == cache_key(REQ, "m")
    llm = FakeLLM(responses=["for retail", "for logistics"])
    gw = gateway(llm, clock, cache=InMemoryResponseCache())
    assert gw.complete(retail).text == "for retail"
    assert gw.complete(logistics).text == "for logistics"  # identical request, different scope: no hit
    assert gw.complete(retail).text == "for retail" and len(llm.requests) == 2


def test_require_cache_scope_fails_closed_for_unscoped_requests(clock):
    llm = FakeLLM(responses=["a", "b", "c", "d"])
    gw = gateway(llm, clock, cache=InMemoryResponseCache(), require_cache_scope=True)
    assert gw.complete(REQ).text == "a" and gw.complete(REQ).text == "b"  # never cached
    scoped = REQ.model_copy(update={"metadata": {"cache_scope": "tenant:retail"}})
    assert gw.complete(scoped).text == "c" and gw.complete(scoped).text == "c"


# ------------------------------------------------------------------ async across event loops
def test_async_gateway_works_across_event_loops_under_contention():
    import asyncio

    class SlowAsync:
        provider = "slow"
        default_model = "slow-model"

        async def acomplete(self, req):
            await asyncio.sleep(0.001)
            return Completion(message=Message.assistant("ok"), model="slow-model", provider="slow")

    gw = ModelGateway(SlowAsync(), max_concurrency=1)

    async def burst():
        results = await asyncio.gather(*(gw.acomplete(REQ) for _ in range(3)))
        return [r.text for r in results]

    # Each asyncio.run creates a new loop; a semaphore bound to the first loop would raise
    # "is bound to a different event loop" on the second burst.
    assert asyncio.run(burst()) == ["ok"] * 3
    assert asyncio.run(burst()) == ["ok"] * 3
