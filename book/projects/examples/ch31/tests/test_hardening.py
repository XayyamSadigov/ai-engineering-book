# path: book/projects/examples/ch31/tests/test_hardening.py
"""Edge cases behind the chapter's guarantees: exception text, key shapes, root keys, thin baselines."""
import random

from aie_core.observability import InMemoryTracer

from instrument import AITracer, CapturePolicy, redact, trace_request
from metrics import AlertRule, compute, evaluate
from semconv import Attr, SpanName
from trace_store import REQUIRED_ROOT_KEYS, SpanRecord, TraceStore


def test_exception_text_obeys_the_capture_ceiling():
    sink = InMemoryTracer()
    tracer = AITracer(sink, capture=CapturePolicy(mode="off", salt="t", tenant_ceiling={"logistics": "off"}),
                      rng=random.Random(1))
    try:
        with trace_request(tracer, route="r", tenant="logistics"):
            raise ValueError("no customer for jane@corp.com card 4111 1111 1111 1111")
    except ValueError:
        pass
    events = [e for s in sink.spans for e in s.events if e.get("type") == "exception"]
    assert events and all("jane@corp.com" not in str(e) and "4111" not in str(e) for e in events)


def test_current_api_key_shapes_are_redacted():
    for key in ("sk-ant-api03-abcdefghijklmnopqrstuvwx", "sk-proj-abcdefghijklmnopqrst",
                "ghp_abcdefghijklmnopqrstuvwxyz0123", "AKIAABCDEFGHIJKLMNOP"):
        assert "[SECRET]" in redact(f"key {key} end"), key


def _span(trace, span_id, parent, name, start, attrs, status="ok", ms=100.0):
    return SpanRecord(trace_id=trace, span_id=span_id, parent_span_id=parent, name=name, start=start,
                      end=start + ms / 1000, duration_ms=ms, status=status, attributes=attrs)


def test_a_root_that_lost_a_required_key_is_incomplete():
    root_attrs = {k: "x" for k in REQUIRED_ROOT_KEYS if k != Attr.INDEX_VERSION}
    store = TraceStore([
        _span("t1", "r", None, SpanName.REQUEST, 0.0, root_attrs),
        _span("t1", "c", "r", SpanName.RETRIEVAL, 0.0, {Attr.INDEX_VERSION: "idx-1"}),
    ])
    assert not store.traces()[0].complete and store.completeness()[Attr.INDEX_VERSION] == 1.0


def test_telemetry_rules_see_traces_without_a_root():
    orphans = [_span(f"t{i}", "c", "missing-root", SpanName.RETRIEVAL, 100.0 + i, {}) for i in range(60)]
    rule = AlertRule("telemetry_gaps", "telemetry.incomplete_rate", ">", window=3600, threshold=0.05, min_samples=30)
    [result] = evaluate([rule], TraceStore(orphans), now=3600.0)
    assert result.fired and result.samples == 60


def test_a_thin_baseline_is_not_a_reference():
    def request(trace, start, status):
        return _span(trace, "r", None, SpanName.REQUEST, start, {Attr.TENANT: "retail"}, status=status)
    base = [request("b0", 10.0, "error")]                                   # one failed request
    current = [request(f"c{i}", 90_000.0 + i, "ok") for i in range(40)]
    rule = AlertRule("drop", "slo.availability_burn_rate", ">", window=3600, baseline_ratio=2.0,
                     baseline_window=86_400, min_samples=30)
    [result] = evaluate([rule], TraceStore(base + current), now=93_000.0)
    assert not result.fired and result.reason == "insufficient data"


def test_completion_burn_counts_served_requests_only():
    fast_errors = [_span(f"e{i}", "r", None, SpanName.REQUEST, float(i), {}, status="error") for i in range(60)]
    slow_ok = [_span(f"s{i}", "r", None, SpanName.REQUEST, 100.0 + i, {}, ms=60_000.0) for i in range(40)]
    mixed = compute("slo.completion_burn_rate", TraceStore(fast_errors + slow_ok))
    only_served = compute("slo.completion_burn_rate", TraceStore(slow_ok))
    assert mixed == only_served   # fast failures neither help nor hurt the latency objective
