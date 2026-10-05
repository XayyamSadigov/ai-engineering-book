# path: book/projects/examples/ch30/demo.py
"""Print the chapter's worked example: the optimization ladder for a Northwind Assist answer.

Run: python demo.py   (from this directory, with aie_core installed). All prices are illustrative.
"""
from __future__ import annotations

from aie_core.llm.gateway import PricingTable

from cost import CostBreakdown, CostModel, CostScenario
from latency import LatencyBudget

PRICES = {
    "capable-model": {"input_per_1m": 2.0, "cached_input_per_1m": 0.2, "output_per_1m": 8.0},
    "small-model": {"input_per_1m": 0.2, "cached_input_per_1m": 0.02, "output_per_1m": 0.8},
}
QUESTIONS_PER_DAY = 6_000


def ladder(cm: CostModel) -> list[CostBreakdown]:
    common = dict(embedding_tokens=30, embedding_price_per_1m=0.02)
    v0 = CostScenario(name="v0 naive", model="capable-model", input_tokens=9_000, output_tokens=600, retry_rate=0.08, success_rate=0.86, **common)
    trimmed = v0.model_copy(update=dict(name="1 trim context + cap output", input_tokens=4_500, output_tokens=300, retry_rate=0.04, success_rate=0.88, rerank_calls=1, rerank_price_per_call=0.0005))
    prefix = trimmed.model_copy(update=dict(name="2 stable prefix cached", cached_input_tokens=800))
    hit = prefix.model_copy(update=dict(name="cache hit", input_tokens=0, cached_input_tokens=0, output_tokens=0, rerank_calls=0, retry_rate=0.0, success_rate=0.97))
    cached = cm.mix("3 + semantic cache, 8% hits", [(0.08, hit), (0.92, prefix)])
    small = prefix.model_copy(update=dict(name="small route", model="small-model", success_rate=0.84))
    routed = cm.mix("4 + route 60% to small model", [(0.08, hit), (0.552, small), (0.368, prefix)])
    return [cm.breakdown(v0), cm.breakdown(trimmed), cm.breakdown(prefix), cached, routed]


def main() -> None:
    cm = CostModel(PricingTable(PRICES))
    rows = ladder(cm)
    base = rows[0].per_successful_task
    print(f"{'step':34} {'USD/attempt':>12} {'USD/success':>12} {'vs v0':>7} {'USD/month':>10}")
    for b in rows:
        monthly = b.per_successful_task * QUESTIONS_PER_DAY * 30
        print(f"{b.scenario:34} {b.per_attempt:12.5f} {b.per_successful_task:12.5f} {(b.per_successful_task / base - 1) * 100:6.0f}% {monthly:10.0f}")
    budget = LatencyBudget.allocate(
        8_000,
        {"auth": 1, "retrieve": 6, "rerank": 3, "generate": 62, "persist": 3, "validate": 1},
        ttft_ms=2_000,
        before_first_token=["auth", "retrieve", "rerank"],
        reserve_fraction=0.05,
    )
    print("\nstage budgets (ms):", {s.name: s.budget_ms for s in budget.stages}, "reserve", round(budget.reserve_ms, 1))


if __name__ == "__main__":
    main()
