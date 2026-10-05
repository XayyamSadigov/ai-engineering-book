# path: book/projects/examples/ch31/tests/test_slo_alerts.py
"""SLO burn-rate metrics and multi-window alert rules (Chapter 29 arithmetic, Chapter 31 alerts)."""
import random
from pathlib import Path

import pytest
from aie_core.observability import InMemoryTracer

from instrument import AITracer, CapturePolicy
from metrics import AlertRule, compute, evaluate, load_rules
from trace_store import TraceStore

HERE = Path(__file__).resolve().parents[1]
END = 1_000_000.0


def workload(fail_every: int, *, healthy_tail_s: float = 0.0, slow_ms: float = 0.0) -> TraceStore:
    """Two hours of requests every 10 s. Every `fail_every`-th request fails (0 = none), except
    during the last `healthy_tail_s` seconds; `slow_ms` sets every request's duration."""
    now = [0.0]
    sink = InMemoryTracer()
    tracer = AITracer(sink, capture=CapturePolicy(salt="t"), rng=random.Random(3), clock=lambda: now[0])
    start = END - 7200
    for i in range(720):
        t0 = start + i * 10
        now[0] = t0
        fail = fail_every and i % fail_every == 0 and t0 < END - healthy_tail_s
        try:
            with tracer.span("request", **{"tenant.id": "retail", "app.route": "rag.answer"}):
                now[0] = t0 + (slow_ms or 900.0) / 1000
                if fail:
                    raise RuntimeError("provider unavailable")
        except RuntimeError:
            pass
    return TraceStore.from_spans(sink.spans)


def rule(name: str) -> AlertRule:
    return next(r for r in load_rules(HERE / "alerts.yaml") if r.name == name)


def test_burn_rate_is_bad_share_over_budget():
    store = workload(fail_every=10)                     # 10% failed against a 0.5% budget
    assert compute("slo.availability_burn_rate", store) == pytest.approx(20.0)
    assert compute("slo.completion_burn_rate", workload(0, slow_ms=9000)) == pytest.approx(20.0)
    assert compute("slo.completion_burn_rate", workload(0)) == 0.0


def test_fast_burn_pages_only_when_the_short_window_still_burns():
    ongoing = evaluate([rule("availability_fast_burn")], workload(fail_every=10), now=END)
    assert [r.fired for r in ongoing] == [True]
    recovered = evaluate([rule("availability_fast_burn")], workload(fail_every=10, healthy_tail_s=600), now=END)
    assert [r.fired for r in recovered] == [False]
    assert "short window" in recovered[0].reason


def test_confirm_window_requires_threshold_and_a_shorter_window():
    base = {"name": "x", "metric": "slo.availability_burn_rate", "window": "1h"}
    with pytest.raises(ValueError):
        AlertRule.from_dict({**base, "confirm_window": "1h", "condition": {"op": ">", "threshold": 1}})
    with pytest.raises(ValueError):
        AlertRule.from_dict({**base, "confirm_window": "5m", "condition": {"op": ">", "baseline_ratio": 2}})
