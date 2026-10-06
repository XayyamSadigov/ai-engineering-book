# path: book/projects/examples/ch35/test_back_of_envelope.py
from __future__ import annotations

import math

import pytest

from back_of_envelope import (
    Prices,
    Workload,
    cost_per_day,
    daily_tokens,
    estimate,
    inflight_requests,
    peak_rps,
    replicas_needed,
    tokens_per_second,
    vector_storage_bytes,
)


def test_tokens_per_second_basic() -> None:
    # 10 rps x 2,000 input tokens = 20k input tokens/s; 10 x 500 = 5k output tokens/s
    assert tokens_per_second(10, 2_000) == 20_000
    assert tokens_per_second(10, 500) == 5_000


def test_inflight_is_littles_law() -> None:
    # 10 rps lasting 5 s each -> about 50 requests in flight
    assert inflight_requests(10, 5) == 50
    assert inflight_requests(20, 4) == 80


def test_daily_tokens() -> None:
    # 3 rps average over 24 h = 259,200 requests; x 2,000 tokens = 518.4M input tokens
    requests = 3 * 24 * 3600
    assert requests == 259_200
    assert daily_tokens(requests, 2_000) == 518_400_000


def test_cost_per_day_separates_cached_tokens() -> None:
    # 27M input at $2/M + 1.8M output at $8/M (illustrative) = 54 + 14.4
    assert cost_per_day(27e6, 1.8e6, 2.0, 8.0) == pytest.approx(68.4)
    # moving half the input to a $0.20/M cache line: 13.5*2 + 13.5*0.2 + 14.4
    assert cost_per_day(13.5e6, 1.8e6, 2.0, 8.0, cached_input_tokens=13.5e6, price_cached_per_m=0.2) == pytest.approx(44.1)
    assert cost_per_day(0, 0, 2.0, 8.0, fixed_cost=120.0) == 120.0


def test_peak_rps_from_daily_volume() -> None:
    # 6,000 questions over 8 h = 0.208/s average; x3 peak factor = 0.625 rps
    assert peak_rps(6_000, active_hours=8, peak_factor=3) == pytest.approx(0.625)
    with pytest.raises(ValueError):
        peak_rps(1, active_hours=0)


def test_vector_storage() -> None:
    # 1M chunks x 1,024 dims x 4 bytes = 4.1 GB raw; x1.5 index overhead = 6.1 GB
    assert vector_storage_bytes(1_000_000, 1_024) == pytest.approx(6.144e9)


def test_replicas_needed_rounds_up_with_headroom() -> None:
    # 6,000 output tokens/s demand, replica sustains 3,000/s: 2 before headroom, 3 with 1.5x
    assert replicas_needed(6_000, 3_000, headroom=1.0) == 2
    assert replicas_needed(6_000, 3_000) == 3
    assert replicas_needed(0, 3_000) == 1
    with pytest.raises(ValueError):
        replicas_needed(100, 0)


def test_case1_knowledge_assistant_numbers() -> None:
    """The Case 1 figures quoted in the chapter must come out of the same formulas."""
    wl = Workload(
        name="northwind-knowledge",
        daily_requests=6_000,
        input_tokens=4_500,
        output_tokens=300,
        avg_latency_s=6.0,
        active_hours=8,
        peak_factor=5,
    )
    est = estimate(wl, Prices(input_per_m=2.0, output_per_m=8.0))
    assert est.peak_rps == pytest.approx(1.04, abs=0.01)
    assert est.peak_input_tps == pytest.approx(4_687.5, abs=1)
    assert est.peak_output_tps == pytest.approx(312.5, abs=1)
    assert est.inflight == pytest.approx(6.25, abs=0.01)
    assert est.daily_input_tokens == 27_000_000
    assert est.daily_output_tokens == 1_800_000
    assert est.cost_per_day == pytest.approx(68.4)
    assert est.cost_per_request == pytest.approx(0.0114, abs=0.0001)
    assert est.notes


def test_case2_prefix_cache_changes_the_cost_line() -> None:
    """Support copilot: 72,000 suggestions/day, 5,800 in / 250 out, half the prefix cached."""
    base = Workload("copilot", 72_000, 5_800, 250, avg_latency_s=4.0, peak_factor=2)
    cached = Workload("copilot", 72_000, 5_800, 250, avg_latency_s=4.0, peak_factor=2, cached_fraction=0.5)
    prices = Prices(input_per_m=2.0, output_per_m=8.0, cached_input_per_m=0.2)
    e0, e1 = estimate(base, prices), estimate(cached, prices)
    assert e0.peak_rps == pytest.approx(5.0)
    assert e0.inflight == pytest.approx(20.0)
    assert e0.cost_per_day == pytest.approx(979.2)
    assert e1.daily_cached_tokens == pytest.approx(208.8e6)
    assert e1.cost_per_day == pytest.approx(603.36)
    assert math.isclose(e1.cost_per_day / e0.cost_per_day, 0.616, abs_tol=0.001)


def test_case1_storage_fits_one_postgres_instance() -> None:
    """1 million chunks x 1,024 dims x 4 bytes with 1.5x HNSW overhead is about 6.1 GB."""
    raw = vector_storage_bytes(1_000_000, 1_024, index_overhead=1.0)
    assert raw == pytest.approx(4.096e9)
    assert vector_storage_bytes(1_000_000, 1_024) == pytest.approx(6.144e9)


def test_case2_voice_llm_cost_is_the_smaller_line() -> None:
    """240,000 turns/day at 2,500 in (70 percent cached) and 80 out; speech is a fixed 1,200 USD/day."""
    turns = Workload("voice", 240_000, 2_500, 80, avg_latency_s=1.2, peak_factor=1.0, cached_fraction=0.7)
    prices = Prices(input_per_m=2.0, output_per_m=8.0, cached_input_per_m=0.2)
    llm = estimate(turns, prices)
    assert llm.daily_input_tokens == pytest.approx(180e6)
    assert llm.daily_cached_tokens == pytest.approx(420e6)
    assert llm.cost_per_day == pytest.approx(597.6)  # 360 + 84 + 153.6, "598" in the chapter
    speech = 48_000 * (0.01 + 0.015)  # call minutes x illustrative STT + TTS per minute
    total = estimate(turns, prices, fixed_cost=speech).cost_per_day
    assert speech == pytest.approx(1_200)
    assert total / 8_000 == pytest.approx(0.225, abs=0.001)  # about 0.22 USD per call
    assert speech / total == pytest.approx(0.667, abs=0.01)  # speech is about two thirds
    # Little's Law on calls, not requests: 1,500 calls/hour x 360 s average duration
    assert inflight_requests(1_500 / 3_600, 360) == pytest.approx(150)


def test_case3_review_rate_dominates_model_cost() -> None:
    invoices = Workload("invoices", 8_000, 2_600, 400, avg_latency_s=20.0)
    capable = estimate(invoices, Prices(2.0, 8.0)).cost_per_day
    small = estimate(invoices, Prices(0.2, 0.8)).cost_per_day
    assert capable == pytest.approx(67.2)
    assert small == pytest.approx(6.72)
    review_cost = 8_000 * 0.15 * (2 / 60) * 30.0  # 15 percent reviewed, 2 minutes each, 30 USD/hour
    assert review_cost == pytest.approx(1_200)
    assert review_cost / 15.0 > 50  # review is well over fifty times the cascade's ~15 USD of tokens
    # each point of review rate: 80 invoices, about 2.7 hours, about 80 USD per day
    assert 8_000 * 0.01 * (2 / 60) * 30.0 == pytest.approx(80)


def test_case4_prefix_cache_is_what_meets_the_agent_budget() -> None:
    autocomplete = Workload("autocomplete", 120_000, 2_000, 30, avg_latency_s=0.6, peak_factor=2,
                            cached_fraction=0.8)
    ac = estimate(autocomplete, Prices(0.2, 0.8, cached_input_per_m=0.02))
    assert ac.peak_rps == pytest.approx(8.33, abs=0.01)
    assert ac.inflight == pytest.approx(5.0, abs=0.01)
    assert ac.cost_per_day == pytest.approx(16.32)
    assert ac.cost_per_day / 400 < 0.05  # under 0.05 USD per engineer per day

    steps = dict(daily_requests=30_000, input_tokens=30_000, output_tokens=600, avg_latency_s=4.0, peak_factor=2.5)
    prices = Prices(2.0, 8.0, cached_input_per_m=0.2)
    uncached = estimate(Workload("agent", **steps), prices)
    cached = estimate(Workload("agent", cached_fraction=0.8, **steps), prices)
    assert uncached.peak_rps == pytest.approx(2.6, abs=0.01)
    assert uncached.cost_per_day == pytest.approx(1_944)
    assert uncached.cost_per_day / 1_200 == pytest.approx(1.62)  # over the 1 USD/task ceiling
    assert cached.cost_per_day == pytest.approx(648)
    assert cached.cost_per_day / 1_200 == pytest.approx(0.54)


def test_inputs_that_would_produce_silent_wrong_answers_are_refused() -> None:
    import pytest
    from back_of_envelope import Prices, Workload, estimate, replicas_needed
    with pytest.raises(ValueError):
        Workload("x", 1000, 1000, 0, 1, cached_fraction=1.5)
    with pytest.raises(ValueError):
        estimate(Workload("x", 1000, 1000, 0, 1, cached_fraction=0.8), Prices(2, 8))   # cached price missing
    with pytest.raises(ValueError):
        replicas_needed(6000, 3000, headroom=0.5)
    idle = estimate(Workload("x", 0, 1000, 100, 1), Prices(2, 8), fixed_cost=120)
    assert idle.cost_per_request == float("inf")
