# path: book/capstone/northwind-assist/northwind_assist/observability/tracing.py
"""One tracer per process, from Chapter 31, wrapped in Chapter 27's RedactingTracer.

Chapter 31's AITracer gives every span a trace id and parent id through its own contextvar, so
the capstone uses it (or its OpenTelemetry subclass) everywhere and nothing else: mixing it with
plain aie_core tracers in one request breaks the tree.

guardrails.RedactingTracer opens each span on the wrapped tracer and scrubs attributes, events
and pending captured content on write and when the body exits, which is before the wrapped
tracer's own end-of-span work (finalize, OTel backend copy, export). Every backend therefore
receives scrubbed data. Pass the RedactingTracer everywhere; tracer-specific attributes such as
`capture_policy` and `sink` are delegated to the AITracer inside.
"""
from __future__ import annotations

import os
from typing import Any

from aie_core.observability import InMemoryTracer, JsonlTracer, NoopTracer, Tracer
from guardrails import RedactingTracer
from instrument import AITracer, CapturePolicy  # type: ignore[import-not-found]


def build_tracer(env: dict[str, str] | None = None, *, sink: Tracer | None = None) -> RedactingTracer:
    """TRACE_SINK=otel -> OpenTelemetry (OTLP to the collector) mirrored to JSONL when TRACE_PATH
    is set; jsonl -> JSONL file; memory -> in-memory (tests); none -> no export."""
    env = dict(os.environ if env is None else env)
    capture = CapturePolicy(mode=env.get("CAPTURE_MODE", "hashed"),  # type: ignore[arg-type]
                            sample_rate=float(env.get("CAPTURE_SAMPLE_RATE", "0")),
                            salt=env.get("CAPTURE_SALT", "change-me"))
    resource = {"service.name": env.get("OTEL_SERVICE_NAME", "northwind-assist"),
                "deployment.environment": env.get("NA_ENVIRONMENT", "dev"),
                "app.version": env.get("NA_APP_VERSION", "1.0.0")}
    kind = env.get("TRACE_SINK", "none")
    if sink is None:
        if kind == "memory":
            sink = InMemoryTracer()
        elif kind in ("jsonl", "otel") and env.get("TRACE_PATH"):
            sink = JsonlTracer(env["TRACE_PATH"])
        else:
            sink = NoopTracer()
    if kind == "otel":
        try:
            from otel_setup import OTelAITracer, build_provider  # type: ignore[import-not-found]
        except ImportError:  # SDK missing: fall back to JSONL rather than failing the boot
            fallback = sink if not isinstance(sink, NoopTracer) else JsonlTracer("traces.jsonl")
            return RedactingTracer(AITracer(fallback, resource=resource, capture=capture))
        provider, exporter = build_provider(resource["service.name"], exporter=env.get("OTEL_EXPORTER", "otlp"),
                                     resource_attributes={k: v for k, v in resource.items() if k != "service.name"},
                                     endpoint=env.get("OTEL_EXPORTER_OTLP_ENDPOINT"),
                                     sample_ratio=float(env.get("OTEL_SAMPLE_RATIO", "1.0")))
        otel = OTelAITracer(provider, sink=sink, resource=resource, capture=capture)
        otel.exporter = exporter          # OTEL_EXPORTER=memory: tests read finished spans from it
        return RedactingTracer(otel)
    return RedactingTracer(AITracer(sink, resource=resource, capture=capture))


def spans_of(tracer: Any) -> list[Any]:
    """Exported spans when the sink keeps them (memory sink); used by tests and the eval run."""
    sink = getattr(tracer, "sink", None)
    return list(getattr(sink, "spans", []))


__all__ = ["build_tracer", "spans_of"]
