# path: book/projects/reliability/tests/test_retry.py
import random

import pytest
from aie_core.llm.errors import ProviderUnavailableError

from reliability import Deadline, RetryBudget, call_with_retry
from reliability.retry import RetryPolicy


class Flaky:
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise ProviderUnavailableError("503")
        return "ok"


def test_retries_transient_then_succeeds(clock):
    fn = Flaky(2)
    assert call_with_retry(fn, policy=RetryPolicy(max_attempts=3), sleep=clock.sleep, rng=random.Random(1)) == "ok"
    assert fn.calls == 3 and len(clock.sleeps) == 2


def test_does_not_retry_deterministic_errors(clock):
    calls = []

    def bad() -> None:
        calls.append(1)
        raise ValueError("bad payload")

    with pytest.raises(ValueError):
        call_with_retry(bad, sleep=clock.sleep)
    assert len(calls) == 1


def test_stops_when_backoff_would_overrun_deadline(clock):
    fn = Flaky(5)
    d = Deadline.after(1.0, clock=clock)
    with pytest.raises(ProviderUnavailableError):
        call_with_retry(fn, policy=RetryPolicy(max_attempts=5, base_delay_s=2.0, jitter=False),
                        deadline=d, sleep=clock.sleep)
    assert fn.calls == 1


def test_retry_amplification_without_and_with_budget(clock):
    """Three layers x three attempts = 27 calls per request; a shared budget caps it."""
    policy = RetryPolicy(max_attempts=3, base_delay_s=0.0, jitter=False)

    def run(budget: RetryBudget | None) -> int:
        dep = Flaky(10**6)

        def layer(depth: int):
            if depth == 0:
                return dep()
            return call_with_retry(lambda: layer(depth - 1), policy=policy, budget=budget, sleep=clock.sleep)

        with pytest.raises(ProviderUnavailableError):
            layer(3)
        return dep.calls

    assert run(None) == 27
    budget = RetryBudget(ratio=0.1, min_retries_per_s=0.0, clock=clock)
    assert run(budget) == 1       # no deposits yet beyond 0.3 tokens: retries refused
    assert budget.retries_denied >= 1


def test_budget_allows_ratio_of_traffic(clock):
    budget = RetryBudget(ratio=0.1, min_retries_per_s=0.0, clock=clock)
    for _ in range(100):
        budget.record_request()
    allowed = sum(budget.try_spend() for _ in range(50))
    assert allowed == 10
