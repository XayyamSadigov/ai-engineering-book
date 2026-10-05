# path: book/projects/examples/ch31/otel_setup.py
"""OpenTelemetry SDK wiring for the book's AI spans.

`OTelAITracer` keeps the `AITracer` behavior (capture policy, error classes, resource
attributes, aie_core compatibility) and additionally opens a *real* OpenTelemetry span for
every stage, made current in the OTel context. Three consequences:

* trace and span ids come from OpenTelemetry, so they match what auto-instrumented HTTP and
  database clients emit inside the same request;
* W3C `traceparent` propagation to downstream services works with the standard propagators;
* any OTLP-speaking backend (a collector, Jaeger, Tempo, a vendor) receives the spans.

Exporters: "console" (stdout, development), "otlp" (needs `opentelemetry-exporter-otlp`,
not installed by default), "memory" (tests), "none". Batch processing is used for network
exporters so the request path never waits on telemetry; tests use the simple processor so
spans are visible immediately.
"""
from __future__ import annotations

import json
import os
from typing import Any, Literal

from opentelemetry import context as otel_context
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor, SpanExporter
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.trace import Status, StatusCode

from aie_core.observability import JsonlTracer, NoopTracer, Tracer

from instrument import AITracer, CapturePolicy, LinkedSpan

ExporterKind = Literal["console", "otlp", "memory", "none"]


def _otlp_exporter(endpoint: str | None) -> SpanExporter:
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    except ImportError as exc:
        raise ImportError("OTLP export needs `pip install opentelemetry-exporter-otlp-proto-http`") from exc
    # endpoint defaults to OTEL_EXPORTER_OTLP_ENDPOINT, the standard variable, when None
    return OTLPSpanExporter(endpoint=endpoint) if endpoint else OTLPSpanExporter()


def build_provider(
    service_name: str,
    *,
    exporter: ExporterKind = "console",
    resource_attributes: dict[str, Any] | None = None,
    endpoint: str | None = None,
    sample_ratio: float = 1.0,
) -> tuple[TracerProvider, SpanExporter | None]:
    """A private TracerProvider (not the global one, so tests stay isolated)."""
    resource = Resource.create({"service.name": service_name, **(resource_attributes or {})})
    # ParentBased: a child follows its parent's decision, so a trace is never half-sampled
    provider = TracerProvider(resource=resource, sampler=ParentBased(TraceIdRatioBased(sample_ratio)))
    span_exporter: SpanExporter | None
    if exporter == "memory":
        span_exporter = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    elif exporter == "console":
        span_exporter = ConsoleSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(span_exporter))
    elif exporter == "otlp":
        span_exporter = _otlp_exporter(endpoint)
        provider.add_span_processor(BatchSpanProcessor(span_exporter, max_queue_size=4096, max_export_batch_size=512))
    else:
        span_exporter = None
    return provider, span_exporter


def _otel_value(value: Any) -> Any:
    """OTel attributes accept primitives and homogeneous lists of primitives; JSON the rest."""
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        return list(value)
    if isinstance(value, (list, tuple)) and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
        return [float(v) for v in value]
    return json.dumps(value, default=str, sort_keys=True)


class OTelAITracer(AITracer):
    """AITracer whose spans are real OpenTelemetry spans. `sink` optionally mirrors to JSONL."""

    def __init__(self, provider: TracerProvider, *, sink: Tracer | None = None, **kwargs: Any) -> None:
        super().__init__(sink=sink or NoopTracer(), **kwargs)
        self.provider = provider
        self._otel = provider.get_tracer("northwind.assist", schema_url=None)

    def _backend_start(self, span: LinkedSpan, parent: LinkedSpan | None) -> Any:
        otel_span = self._otel.start_span(span.name, start_time=int(span.start * 1e9))
        ctx = otel_span.get_span_context()
        span.trace_id = format(ctx.trace_id, "032x")
        span.span_id = format(ctx.span_id, "016x")
        token = otel_context.attach(otel_trace.set_span_in_context(otel_span))
        return otel_span, token

    def _backend_end(self, span: LinkedSpan, handle: Any) -> None:
        otel_span, token = handle
        for key, value in span.attributes.items():
            if value is not None:
                otel_span.set_attribute(key, _otel_value(value))
        for ev in span.events:
            otel_span.add_event(str(ev.get("type", "event")), {k: str(v) for k, v in ev.items() if k not in ("type", "time")})
        if span.status == "error":
            otel_span.set_status(Status(StatusCode.ERROR, str(span.attributes.get("error.class", ""))))
        otel_span.end(end_time=int((span.end or span.start) * 1e9))
        otel_context.detach(token)


def tracer_from_env(env: dict[str, str] | None = None) -> AITracer:
    """Build the process tracer from environment variables (see the README table)."""
    env = dict(os.environ if env is None else env)
    capture = CapturePolicy(
        mode=env.get("CAPTURE_MODE", "hashed"),  # type: ignore[arg-type]
        sample_rate=float(env.get("CAPTURE_SAMPLE_RATE", "0")),
        salt=env.get("CAPTURE_SALT", "change-me"),
    )
    resource = {"service.name": env.get("OTEL_SERVICE_NAME", "northwind-assist"),
                "deployment.environment": env.get("DEPLOY_ENV", "dev"),
                "app.version": env.get("APP_VERSION", "dev")}
    sink_kind = env.get("TRACE_SINK", "none")
    if sink_kind == "otel":
        provider, _ = build_provider(resource["service.name"], exporter=env.get("OTEL_EXPORTER", "otlp"),  # type: ignore[arg-type]
                                     resource_attributes={k: v for k, v in resource.items() if k != "service.name"},
                                     endpoint=env.get("OTEL_EXPORTER_OTLP_ENDPOINT"),
                                     sample_ratio=float(env.get("OTEL_SAMPLE_RATIO", "1.0")))
        return OTelAITracer(provider, resource=resource, capture=capture,
                            emit_genai_aliases=env.get("EMIT_GENAI_ALIASES", "false").lower() == "true")
    sink: Tracer = JsonlTracer(env.get("TRACE_PATH", "traces.jsonl")) if sink_kind == "jsonl" else NoopTracer()
    return AITracer(sink, resource=resource, capture=capture)


__all__ = ["build_provider", "OTelAITracer", "tracer_from_env"]
