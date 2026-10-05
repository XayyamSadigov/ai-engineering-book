# path: book/projects/guardrails/tests/test_telemetry.py
from __future__ import annotations

from aie_core.observability import InMemoryTracer

from guardrails import RedactingTracer, scrub


def test_scrub_nested():
    data = {"q": "mail jane@northwind.example", "args": ["AKIAABCDEFGHIJKLMNOP", 3], "n": 1}
    out = scrub(data)
    assert "jane@" not in out["q"] and "AKIA" not in out["args"][0] and out["args"][1] == 3 and out["n"] == 1


def test_redacting_tracer_scrubs_before_export():
    inner = InMemoryTracer()
    tracer = RedactingTracer(inner)
    try:
        with tracer.span("llm.call", prompt="full prompt with secrets", user="jane@northwind.example") as s:
            s.set_attribute("tool.args", {"card": "4111 1111 1111 1111"})
            raise ValueError("failed for token AKIAABCDEFGHIJKLMNOP")
    except ValueError:
        pass
    exported = repr(inner.spans[0].to_dict())
    for leaked in ("jane@northwind.example", "4111 1111 1111 1111", "AKIAABCDEFGHIJKLMNOP", "full prompt"):
        assert leaked not in exported


# ---------------------------------------------------------------- redaction before any backend (Chapter 31)
import importlib
import sys
from pathlib import Path

import pytest

CH31 = Path(__file__).resolve().parents[2] / "examples" / "ch31"
LEAKS = ("jane@northwind.example", "4111 1111 1111 1111", "AKIAABCDEFGHIJKLMNOP")


def _ch31(module: str):
    if not CH31.exists():
        pytest.skip("examples/ch31 not present")
    sys.path.insert(0, str(CH31))
    try:
        return importlib.import_module(module)
    finally:
        sys.path.remove(str(CH31))


def _exercise(tracer) -> None:
    with tracer.span("request", user="jane@northwind.example"):
        with tracer.span("tool.call") as s:
            s.set_attribute("tool.args", {"card": "4111 1111 1111 1111"})
            s.attributes["tool.raw"] = "reply to jane@northwind.example"   # direct write, bypassing set_attribute
        try:
            with tracer.span("llm.call") as s:
                tracer.capture(s, "completion.text", "answer for jane@northwind.example")
                raise RuntimeError("provider rejected key AKIAABCDEFGHIJKLMNOP")
        except RuntimeError:
            pass


def test_redaction_happens_before_the_ch31_backend_copies_attributes():
    instrument = _ch31("instrument")
    backend: list[str] = []

    class RecordingBackendTracer(instrument.AITracer):
        """Stands in for OTelAITracer: copies attributes and events in _backend_end, before the sink."""

        def _backend_end(self, span, handle):
            backend.append(repr(span.attributes) + repr(span.events))

    sink = InMemoryTracer()
    tracer = RedactingTracer(RecordingBackendTracer(sink, capture=instrument.CapturePolicy(mode="full")))
    _exercise(tracer)
    assert len(backend) == 3 and len(sink.spans) == 3
    exported = " ".join(backend) + repr([s.to_dict() for s in sink.spans])
    for leaked in LEAKS:
        assert leaked not in exported, leaked
    llm = next(s for s in sink.spans if s.name == "llm.call")
    assert llm.status == "error" and llm.trace_id == sink.spans[-1].trace_id   # tree and error status survive


def test_redaction_reaches_real_opentelemetry_spans():
    pytest.importorskip("opentelemetry.sdk")
    otel_setup = _ch31("otel_setup")
    instrument = _ch31("instrument")
    provider, exporter = otel_setup.build_provider("guardrails-test", exporter="memory")
    tracer = RedactingTracer(otel_setup.OTelAITracer(provider, capture=instrument.CapturePolicy(mode="full")))
    _exercise(tracer)
    finished = exporter.get_finished_spans()
    assert len(finished) == 3
    dumped = repr([(dict(s.attributes), [(e.name, dict(e.attributes)) for e in s.events]) for s in finished])
    for leaked in LEAKS:
        assert leaked not in dumped, leaked
