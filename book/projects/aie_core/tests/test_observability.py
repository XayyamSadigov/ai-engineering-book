# path: book/projects/aie_core/tests/test_observability.py
import json

import pytest

from aie_core.observability import InMemoryTracer, JsonlTracer, NoopTracer, OTelTracer, Span, Tracer, get_tracer
from aie_core.settings import Settings


def test_span_lifecycle_and_exception_recording():
    tracer = InMemoryTracer()
    with pytest.raises(ValueError):
        with tracer.span("llm.complete", provider="fake") as span:
            span.set_attribute("model", "m")
            raise ValueError("boom")
    s = tracer.spans[0]
    assert isinstance(s, Span) and s.name == "llm.complete"
    assert s.attributes == {"provider": "fake", "model": "m"}
    assert s.status == "error" and s.events[0]["exception.type"] == "ValueError"
    assert s.end is not None and s.duration_ms >= 0


def test_jsonl_tracer_writes_one_object_per_span(tmp_path):
    path = tmp_path / "t" / "traces.jsonl"
    tracer = JsonlTracer(path)
    with tracer.span("a", x=1):
        pass
    with tracer.span("b"):
        pass
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["name"] == "a" and first["attributes"] == {"x": 1} and first["status"] == "ok"


def test_noop_and_base_tracer_do_not_fail():
    for tracer in (NoopTracer(), Tracer()):
        with tracer.span("x") as s:
            s.set_attribute("k", "v")


def test_get_tracer_from_settings(tmp_path):
    assert isinstance(get_tracer(Settings(trace_sink="none")), NoopTracer)
    t = get_tracer(Settings(trace_sink="jsonl", trace_path=str(tmp_path / "x.jsonl")))
    assert isinstance(t, JsonlTracer) and t.path.name == "x.jsonl"


def test_otel_tracer_exports_when_sdk_present():
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = OTelTracer.__new__(OTelTracer)
    tracer._otel = provider.get_tracer("test")
    with tracer.span("llm.complete", provider="fake", cost_usd=0.5, tools=["a"]):
        pass
    spans = exporter.get_finished_spans()
    assert len(spans) == 1 and spans[0].name == "llm.complete"
    assert spans[0].attributes["provider"] == "fake" and spans[0].attributes["tools"] == '["a"]'
    _ = trace


def test_nested_spans_share_trace_and_record_parent():
    tracer = InMemoryTracer()
    with tracer.span("outer") as outer:
        with tracer.span("inner") as inner:
            pass
    with tracer.span("other_root") as root2:
        pass
    assert outer.trace_id and outer.parent_span_id is None
    assert inner.trace_id == outer.trace_id and inner.parent_span_id == outer.span_id
    assert root2.trace_id != outer.trace_id and root2.parent_span_id is None
    assert tracer.spans[0].to_dict()["parent_span_id"] == outer.span_id
    assert tracer.spans[0].to_dict()["trace_id"] == outer.trace_id


def test_context_is_reset_after_exception():
    from aie_core.observability import current_span

    tracer = InMemoryTracer()
    with pytest.raises(RuntimeError):
        with tracer.span("boom"):
            assert current_span() is not None
            raise RuntimeError("x")
    assert current_span() is None


def test_hand_built_span_links_to_current():
    tracer = InMemoryTracer()
    with tracer.span("parent") as parent:
        child = Span(name="manual").link_to_current()
    assert child.trace_id == parent.trace_id and child.parent_span_id == parent.span_id
    root = Span(name="root").link_to_current()
    assert len(root.trace_id) == 32 and root.parent_span_id is None
