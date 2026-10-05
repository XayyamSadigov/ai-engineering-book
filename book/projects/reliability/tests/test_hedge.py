# path: book/projects/reliability/tests/test_hedge.py
"""Hedged requests: tail cutting for idempotent reads, bounded by the retry budget."""
import asyncio

import pytest
from aie_core.llm.errors import ProviderUnavailableError

from reliability import RetryBudget, ahedged


class Replica:
    """Call n sleeps delays[n] seconds and returns its index; records cancellations."""

    def __init__(self, *delays: float, fail: tuple[int, ...] = ()) -> None:
        self.delays, self.fail = delays, fail
        self.calls = 0
        self.cancelled: list[int] = []

    async def __call__(self) -> int:
        n = self.calls
        self.calls += 1
        try:
            await asyncio.sleep(self.delays[n])
        except asyncio.CancelledError:
            self.cancelled.append(n)
            raise
        if n in self.fail:
            raise ProviderUnavailableError(f"replica {n} failed")
        return n


def test_slow_first_copy_is_hedged_and_cancelled():
    rep = Replica(5.0, 0.0)
    result = asyncio.run(ahedged(rep, hedge_after_s=0.02))
    assert result == 1 and rep.calls == 2 and rep.cancelled == [0]


def test_fast_call_sends_no_hedge():
    rep = Replica(0.0, 0.0)
    assert asyncio.run(ahedged(rep, hedge_after_s=0.5)) == 0
    assert rep.calls == 1


def test_empty_budget_switches_hedging_off():
    budget = RetryBudget(ratio=0.0, min_retries_per_s=0.0)
    rep = Replica(0.05, 0.0)
    assert asyncio.run(ahedged(rep, hedge_after_s=0.01, budget=budget)) == 0
    assert rep.calls == 1 and budget.retries_denied == 1


def test_first_success_wins_even_if_another_copy_failed():
    rep = Replica(0.04, 0.0, fail=(1,))
    assert asyncio.run(ahedged(rep, hedge_after_s=0.01)) == 0
    assert rep.calls == 2


def test_all_copies_failing_raises_the_last_error():
    rep = Replica(0.03, 0.0, fail=(0, 1))
    with pytest.raises(ProviderUnavailableError):
        asyncio.run(ahedged(rep, hedge_after_s=0.01))
