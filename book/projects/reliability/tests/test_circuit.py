# path: book/projects/reliability/tests/test_circuit.py
import pytest
from aie_core import FakeLLM, ModelGateway, RetryPolicy
from aie_core import CompletionRequest, Message
from aie_core.llm.errors import InvalidRequestError, ProviderUnavailableError

from reliability import CircuitBreaker, CircuitBreakerClient, CircuitBreakerRegistry, CircuitOpenError, CircuitState


def boom() -> None:
    raise ProviderUnavailableError("503")


def ok() -> str:
    return "ok"


def make(clock, **kw) -> CircuitBreaker:
    params = dict(min_calls=4, failure_rate_threshold=0.5, window_s=10.0, buckets=10, open_s=30.0,
                  half_open_max_calls=2, clock=clock)
    params.update(kw)
    return CircuitBreaker("llm:primary", **params)


def fail_n(b: CircuitBreaker, n: int) -> None:
    for _ in range(n):
        with pytest.raises(ProviderUnavailableError):
            b.call(boom)


def test_does_not_trip_below_minimum_volume(clock):
    b = make(clock)
    fail_n(b, 3)
    assert b.state is CircuitState.CLOSED


def test_full_transition_cycle(clock):
    changes = []
    b = make(clock, on_state_change=lambda n, old, new: changes.append((old.value, new.value)))
    b.call(ok)
    fail_n(b, 3)                                  # 3/4 failures >= 50% with 4 calls
    assert b.state is CircuitState.OPEN
    with pytest.raises(CircuitOpenError) as info:
        b.call(ok)
    assert info.value.retryable and info.value.retry_after_s == pytest.approx(30.0)

    clock.advance(30.0)
    assert b.state is CircuitState.HALF_OPEN
    assert b.call(ok) == "ok"
    assert b.state is CircuitState.HALF_OPEN       # needs 2 probe successes
    assert b.call(ok) == "ok"
    assert b.state is CircuitState.CLOSED
    assert changes == [("closed", "open"), ("open", "half_open"), ("half_open", "closed")]


def test_half_open_failure_reopens_with_longer_period(clock):
    b = make(clock)
    fail_n(b, 4)
    clock.advance(30.0)
    fail_n(b, 1)
    assert b.state is CircuitState.OPEN
    assert b.retry_after() == pytest.approx(60.0)  # open period doubled


def test_half_open_limits_concurrent_probes(clock):
    b = make(clock)
    fail_n(b, 4)
    clock.advance(30.0)
    assert b.allow() and b.allow()
    assert not b.allow()                           # third concurrent probe refused


def test_window_forgets_old_failures(clock):
    b = make(clock)
    fail_n(b, 3)
    clock.advance(11.0)                            # older than the 10 s window
    b.call(ok)
    fail_n(b, 1)
    assert b.state is CircuitState.CLOSED          # only 1 failure out of 2 in the window


def test_caller_errors_do_not_count(clock):
    b = make(clock)

    def bad_request() -> None:
        raise InvalidRequestError("400")

    for _ in range(10):
        with pytest.raises(InvalidRequestError):
            b.call(bad_request)
    assert b.state is CircuitState.CLOSED


def test_slow_calls_count_as_failures(clock):
    b = make(clock, slow_call_s=5.0, window_s=60.0)

    def slow_ok() -> str:
        clock.advance(6.0)
        return "late"

    for _ in range(4):
        b.call(slow_ok)
    assert b.state is CircuitState.OPEN


def test_registry_shares_breakers_per_dependency(clock):
    reg = CircuitBreakerRegistry(min_calls=2, clock=clock)
    assert reg.get("retrieval") is reg.get("retrieval")
    fail_n(reg.get("retrieval"), 2)
    assert reg.open_dependencies() == {"retrieval"}


def test_open_circuit_sends_gateway_straight_to_fallback(clock):
    reg = CircuitBreakerRegistry(min_calls=2, open_s=30.0, clock=clock)
    primary = FakeLLM(responses=[ProviderUnavailableError("503")] * 10, provider="primary")
    backup = FakeLLM(handler=lambda req: "from backup", provider="backup")
    gw = ModelGateway(primary=CircuitBreakerClient(primary, reg.get("llm:primary")),
                      fallbacks=[CircuitBreakerClient(backup, reg.get("llm:backup"))],
                      retry=RetryPolicy(max_attempts=2, base_delay_s=0.1), clock=clock, sleep=clock.sleep)
    req = CompletionRequest(messages=[Message.user("hi")], timeout_s=8.0)
    assert gw.complete(req).provider == "backup"   # 2 failed attempts trip the breaker
    assert reg.get("llm:primary").state is CircuitState.OPEN
    calls_before, sleeps_before = len(primary.requests), len(clock.sleeps)
    assert gw.complete(req).provider == "backup"
    assert len(primary.requests) == calls_before    # primary not contacted at all
    assert len(clock.sleeps) == sleeps_before       # and no backoff wasted: 30 s hint > 8 s deadline
