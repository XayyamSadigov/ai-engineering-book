"""Alerts are code: the rules load against Chapter 31's metric set and evaluate on real capstone traces."""
from __future__ import annotations

import time
from pathlib import Path

import metrics  # type: ignore[import-not-found]   # examples/ch31, on sys.path via northwind_assist._paths
import trace_store  # type: ignore[import-not-found]
from conftest import chat, make

from northwind_assist.observability.tracing import spans_of

RULES = Path(__file__).resolve().parents[1] / "ops" / "alerts.yaml"
QUESTIONS = [
    "How many unused PTO days can I carry over into next year?",
    "What is the laptop replacement process?",
    "What is the capital of Mars?",
]


def _store(container) -> trace_store.TraceStore:
    return trace_store.TraceStore([trace_store.SpanRecord.from_span(s) for s in spans_of(container.tracer)])


def _fired(results) -> set[str]:
    return {r.rule for r in results if r.fired}


def test_alert_rules_load_and_name_only_known_metrics():
    rules = metrics.load_rules(RULES)          # raises on an unknown metric or a malformed condition
    names = {r.name for r in rules}
    assert {"request_p95_slo", "cross_tenant_retrieval", "telemetry_gaps", "abstention_rate_shift"} <= names
    assert all(r.runbook for r in rules)


def test_capstone_traces_are_complete_and_healthy_traffic_fires_nothing():
    c = make()
    for q in QUESTIONS:
        chat(c, "ana", q)
    chat(c, "ana", "create ticket: VPN drops every hour at store 0412", session="alerts-1")
    store = _store(c)
    assert metrics.compute("telemetry.incomplete_rate", store) == 0.0   # Chapter 31 lineage keys on every root
    assert metrics.compute("quality.abstention_rate", store) > 0.0       # the Mars question abstains
    assert _fired(metrics.evaluate(metrics.load_rules(RULES), store, now=time.time() + 1)) == set()


def test_final_acl_check_violation_pages_through_the_cross_tenant_rule():
    c = make()
    real = c.kb.pipeline

    class Contaminated:
        """Fault injection: the pipeline's final check reports a hit it had to drop."""

        def __init__(self, inner):
            self.inner = inner

        def retrieve(self, query, history=None):
            result = self.inner.retrieve(query, history)
            return result.model_copy(update={"trace": {**result.trace, "acl_violations": ["inc-logistics:chunk0"]}})

    c.kb.pipeline = lambda **kw: Contaminated(real(**kw))   # type: ignore[method-assign]
    chat(c, "ana", "What is the laptop replacement process?")
    store = _store(c)
    root = next(t for t in store).root
    assert root.attributes["acl.violations"] == 1
    assert "cross_tenant_retrieval" in _fired(metrics.evaluate(metrics.load_rules(RULES), store, now=time.time() + 1))
