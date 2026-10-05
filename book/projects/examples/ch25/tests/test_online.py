# path: book/projects/examples/ch25/tests/test_online.py
from __future__ import annotations

from datetime import datetime, timedelta

from taskevals.online import (
    ArmCounts,
    CanaryMonitor,
    FeedbackEvent,
    TraceRecord,
    corrections_to_cases,
    join_feedback,
    outcome_metrics,
    simulate_false_alarms,
)

T0 = datetime(2026, 6, 1, 9, 0)


def _trace(i: int, prompt: str, category: str) -> TraceRecord:
    return TraceRecord(trace_id=f"tr-{i}", timestamp=T0 + timedelta(minutes=i), task="ticket.classify",
                       tenant="retail", versions={"prompt": prompt, "model": "m1", "index": "idx-7"},
                       input={"subject": f"ticket {i}"}, output={"category": category})


def test_join_attributes_feedback_and_drops_orphans_and_late_events() -> None:
    traces = [_trace(1, "v1", "hardware"), _trace(2, "v1", "vpn_network"), _trace(3, "v2", "returns")]
    events = [
        FeedbackEvent(trace_id="tr-1", timestamp=T0 + timedelta(hours=1), kind="thumbs_down"),
        FeedbackEvent(trace_id="tr-1", timestamp=T0 + timedelta(hours=2), kind="correction", field="category",
                      value="account_access", actor="agent:j.doe"),
        FeedbackEvent(trace_id="tr-2", timestamp=T0 + timedelta(days=30), kind="label", field="category",
                      value="vpn_network"),  # outside the window
        FeedbackEvent(trace_id="tr-404", timestamp=T0, kind="thumbs_up"),  # trace not found
        FeedbackEvent(trace_id="tr-3", timestamp=T0 + timedelta(days=3), kind="label", field="category",
                      value="returns"),
    ]
    outcomes, stats = join_feedback(traces, events)
    assert (stats.joined, stats.orphan_events, stats.late_events, stats.traces_with_feedback) == (3, 1, 1, 2)
    m = outcome_metrics(outcomes)
    assert m["v1"].correction_rate == 0.5 and m["v1"].negative_rate == 1.0 and m["v1"].feedback_coverage == 0.5
    assert m["v2"].labeled_accuracy == 1.0 and m["v1"].labeled_n == 0


def test_corrections_become_regression_cases_with_trace_lineage() -> None:
    outcomes, _ = join_feedback([_trace(1, "v1", "hardware"), _trace(2, "v1", "vpn_network")], [
        FeedbackEvent(trace_id="tr-1", timestamp=T0 + timedelta(hours=1), kind="correction", field="category",
                      value="account_access"),
        FeedbackEvent(trace_id="tr-2", timestamp=T0 + timedelta(hours=1), kind="correction", field="category",
                      value="vpn_network"),  # "correction" to the same value is not a failure
    ])
    cases = corrections_to_cases(outcomes)
    assert [c.id for c in cases] == ["PROD-tr-1"]
    assert cases[0].expected == {"category": "account_access"}
    assert cases[0].metadata["versions"]["prompt"] == "v1" and "origin:production-correction" in cases[0].tags


def test_canary_rolls_back_on_critical_event_immediately() -> None:
    mon = CanaryMonitor(looks=5, min_n=200)
    d = mon.observe(ArmCounts(n=10, failures=1), ArmCounts(n=10, failures=0, critical=1))
    assert d.decision == "rollback" and "critical" in d.reason


def test_canary_waits_for_minimum_sample_then_detects_harm() -> None:
    mon = CanaryMonitor(looks=5, min_n=200)
    assert mon.observe(ArmCounts(n=100, failures=8), ArmCounts(n=100, failures=20)).decision == "continue"
    d = mon.observe(ArmCounts(n=2000, failures=160), ArmCounts(n=2000, failures=240))
    assert d.decision == "rollback" and d.z is not None and d.z > mon.z_crit


def test_canary_promotes_only_at_final_look_when_non_inferior() -> None:
    mon = CanaryMonitor(looks=3, min_n=200, margin=0.02)
    ctrl, can = ArmCounts(n=5000, failures=400), ArmCounts(n=5000, failures=405)
    assert mon.observe(ctrl, can).decision == "continue"
    assert mon.observe(ctrl, can).decision == "continue"
    final = mon.observe(ctrl, can)
    assert final.decision == "promote" and final.upper_bound is not None and final.upper_bound < 0.02


def test_canary_holds_when_inconclusive_at_final_look() -> None:
    mon = CanaryMonitor(looks=1, min_n=50, margin=0.005)
    assert mon.observe(ArmCounts(n=300, failures=24), ArmCounts(n=300, failures=27)).decision == "hold"


def test_peeking_inflates_false_alarms_and_planned_looks_control_them() -> None:
    rates = simulate_false_alarms(p=0.1, looks=10, n_per_look=300, sims=400, seed=7)
    assert rates["naive_peeking"] > 0.10  # far above the nominal 5%
    assert rates["bonferroni_looks"] <= 0.06
