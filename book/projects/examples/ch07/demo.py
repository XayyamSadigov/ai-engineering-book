# path: book/projects/examples/ch07/demo.py
"""Run the Chapter 7 pipeline end to end on fakes: selection table, cascade sweep, routing.

    python demo.py
"""
from __future__ import annotations

from aie_core import CompletionRequest, Message, PricingTable, ToolSpec

from cascade_eval import (UtilityModel, baseline, best_by_router_accuracy, best_by_utility,
                          calibration_table, collect_outcomes, expected_calibration_error, sweep,
                          to_markdown)
from catalog import northwind_catalog
from fakes import gold_from_cases, make_large_model, make_small_model
from router import northwind_router
from selection import choose, paired_disagreements, run_selection
from tasks import label_confidence, load_ticket_cases, parse_label, score_label


def main() -> None:
    catalog = northwind_catalog()
    pricing = PricingTable(catalog.pricing())
    cases = load_ticket_cases()
    gold = gold_from_cases(cases)

    small = make_small_model()
    general = make_large_model(gold)
    reasoning = make_large_model(gold, error_every=30, latency_ms=6500.0, model="reasoner-2026-01", salt="r")

    print("## 1. Selection on", len(cases), "tickets (illustrative prices and latencies)\n")
    report = run_selection({"nw-small": small, "nw-general": general, "nw-reasoning": reasoning},
                           cases, score_label, pricing)
    print(report.to_markdown())
    pick = choose(report, min_quality=0.80, max_p95_ms=3000)
    print("\nCheapest meeting quality_low >= 0.80 and p95 <= 3000 ms:", pick.candidate if pick else None)
    print("Paired disagreements small vs general:", paired_disagreements(report, "nw-small", "nw-general"))

    print("\n## 2. Cascade small -> general\n")
    outcomes = collect_outcomes(cases, make_small_model(), make_large_model(gold), score_label,
                                label_confidence, pricing)
    print("Calibration of the small model's self-reported confidence:")
    for b in calibration_table(outcomes):
        print(f"  [{b.low:.1f},{b.high:.1f}) n={b.n:2d} mean_conf={b.mean_confidence:.2f} accuracy={b.accuracy:.2f}")
    print(f"  ECE={expected_calibration_error(outcomes):.3f}\n")
    for label, u in [("cheap errors", UtilityModel(cost_silent_error=0.0002)),
                     ("moderate errors", UtilityModel(cost_silent_error=0.001)),
                     ("expensive errors", UtilityModel(cost_silent_error=0.05))]:
        bases = [baseline(outcomes, "small", u), baseline(outcomes, "large", u)]
        rows = sweep(outcomes, u, thresholds=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
        print(f"Utility model: {label} (cost of a silent error {u.cost_silent_error})")
        print(to_markdown([*bases, *rows]))
        full = [*bases, *sweep(outcomes, u)]
        print(f"best by utility: {best_by_utility(full).policy}; "
              f"best by router accuracy: {best_by_router_accuracy(full).policy}\n")

    print("## 3. Routing decisions\n")
    router = northwind_router(catalog, {"nw-small": make_small_model(), "nw-general": make_large_model(gold),
                                        "nw-reasoning": reasoning, "nw-longctx": make_large_model(gold)},
                              confidence=label_confidence, validator=lambda c: parse_label(c) is not None)
    r = router.complete(cases[0].request)
    print("ticket:", r.decision.route, "served_by", r.served_by, "escalated", r.escalated,
          [a.outcome for a in r.attempts])
    long_tools = CompletionRequest(
        messages=[Message.user("incident timeline " * 60_000)], max_tokens=2_000,
        tools=[ToolSpec(name="search_tickets", description="search tickets")])
    d = router.route(long_tools)
    print("long context + tools:", d.route, "->", d.candidates, d.substitutions)


if __name__ == "__main__":
    main()
