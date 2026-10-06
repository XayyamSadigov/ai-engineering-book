# path: book/projects/examples/ch30/test_hardening.py
"""Edge cases behind the chapter's guarantees: unpriced models, failed streams, cancellation, percentiles."""
from __future__ import annotations

import asyncio
import time

import pytest
from aie_core.llm.errors import InvalidRequestError, RateLimitError
from aie_core.llm.providers import FakeLLM

from budgets import BudgetedClient, SpendGuard, SpendPolicy
from latency import LatencyTracker, percentile
from parallel import RequiredStepFailed, Step, fan_out
from test_ch30 import FakeClock, northwind_budget, pricing, req  # noqa: F401  (pricing is a fixture)


def test_an_unpriced_model_is_refused_rather_than_free(pricing) -> None:  # noqa: F811
    guard = SpendGuard({"retail": SpendPolicy(daily_limit_usd=0.01)}, clock=FakeClock())
    client = BudgetedClient(FakeLLM(handler=lambda r: "x"), guard, pricing, default_model="capable-model")
    with pytest.raises(InvalidRequestError):
        client.complete(req("hi", model="capable-modl", metadata={"tenant": "retail"}))


def test_a_stream_that_fails_before_output_costs_nothing(pricing) -> None:  # noqa: F811
    guard = SpendGuard({"retail": SpendPolicy(daily_limit_usd=1.0)}, clock=FakeClock())
    client = BudgetedClient(FakeLLM(responses=[RateLimitError("429")]), guard, pricing, default_model="capable-model")
    with pytest.raises(RateLimitError):
        list(client.stream(req("y", model="capable-model", metadata={"tenant": "retail"})))
    assert guard.spent("retail") == 0.0


def test_a_failed_required_step_stops_its_siblings_at_once() -> None:
    async def fails():
        raise RuntimeError("boom")

    async def slow():
        await asyncio.sleep(2)

    t0 = time.perf_counter()
    with pytest.raises(RequiredStepFailed):
        asyncio.run(fan_out({"a": Step(fails, required=True), "b": Step(slow)}, deadline_s=5.0))
    assert time.perf_counter() - t0 < 1.0


def test_cancelling_the_caller_cancels_the_steps() -> None:
    finished = []

    async def slow():
        await asyncio.sleep(0.3)
        finished.append(1)

    async def main():
        task = asyncio.create_task(fan_out({"b": Step(slow)}, deadline_s=5.0))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.4)

    asyncio.run(main())
    assert finished == []


def test_parallel_calls_to_one_stage_count_wall_time_once() -> None:
    tracker = LatencyTracker(northwind_budget())
    for i in range(2):
        tracker.record({"name": f"r{i}", "attributes": {"request_id": "q", "stage": "retrieve"}, "start": 0.0, "end": 0.5})
    report = tracker.report()
    assert not [v for v in report.violations if v.stage == "retrieve"]


def test_nearest_rank_percentile_is_exact_at_three_nines() -> None:
    assert percentile(list(range(1, 1001)), 99.9) == 999


def test_an_unpriced_returned_model_is_charged_at_the_admitted_price(pricing) -> None:  # noqa: F811
    inner = FakeLLM(handler=lambda r: "ok")

    class Router:
        provider = "router"

        def complete(self, r):   # the provider answers under an id the price table does not know
            return inner.complete(r).model_copy(update={"model": "router-pool-7"})

    guard = SpendGuard({"retail": SpendPolicy(daily_limit_usd=1.0)}, clock=FakeClock())
    BudgetedClient(Router(), guard, pricing, default_model="capable-model").complete(
        req("hi", model="capable-model", metadata={"tenant": "retail"}))
    assert guard.spent("retail") > 0
