# path: book/projects/aie_core/aie_core/observability.py
"""Minimal tracing: spans with attributes, exported to JSONL, OpenTelemetry, or nowhere.

Chapter 31 owns the full tracing schema. This module exists so that every library call can
emit one span with model-level attributes (tokens, cost, cache hit, attempt) without
forcing a dependency on an OpenTelemetry SDK.
"""
from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

# The span currently open in this thread / asyncio task. Children read it to inherit the
# trace id and to record their parent; Tracer.span sets and resets it.
_current_span: ContextVar["Span | None"] = ContextVar("aie_core_current_span", default=None)


@dataclass
class Span:
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)
    start: float = field(default_factory=time.time)
    end: float | None = None
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    trace_id: str = ""  # shared by every span of one logical request; set by link_to_current()
    parent_span_id: str | None = None
    status: str = "ok"
    events: list[dict[str, Any]] = field(default_factory=list)

    def link_to_current(self) -> "Span":
        """Inherit trace_id from the span open in this context (or start a new trace) and
        record that span as the parent. Idempotent; safe to call on a hand-built span."""
        parent = _current_span.get()
        if parent is not None:
            self.trace_id = self.trace_id or parent.trace_id
            self.parent_span_id = self.parent_span_id or parent.span_id
        if not self.trace_id:
            self.trace_id = uuid.uuid4().hex
        return self

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def record_exception(self, exc: BaseException) -> None:
        self.status = "error"
        self.events.append(
            {
                "type": "exception",
                "exception.type": type(exc).__name__,
                "exception.message": str(exc),
                "exception.stacktrace": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
                "time": time.time(),
            }
        )

    def finish(self) -> None:
        if self.end is None:
            self.end = time.time()

    @property
    def duration_ms(self) -> float:
        end = self.end if self.end is not None else time.time()
        return (end - self.start) * 1000

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "start": self.start,
            "end": self.end,
            "duration_ms": round(self.duration_ms, 3),
            "status": self.status,
            "attributes": self.attributes,
            "events": self.events,
        }


class Tracer:
    """Base tracer. Subclasses override `export`; `span()` handles timing and exceptions."""

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        s = Span(name=name, attributes=dict(attributes)).link_to_current()
        token = _current_span.set(s)
        try:
            yield s
        except BaseException as exc:
            s.record_exception(exc)
            raise
        finally:
            _current_span.reset(token)
            s.finish()
            self.export(s)

    def export(self, span: Span) -> None:  # pragma: no cover - overridden
        pass


class NoopTracer(Tracer):
    def export(self, span: Span) -> None:
        return None


class InMemoryTracer(Tracer):
    """Keeps spans in a list; the tracer tests use."""

    def __init__(self) -> None:
        self.spans: list[Span] = []

    def export(self, span: Span) -> None:
        self.spans.append(span)

    def find(self, name: str) -> list[Span]:
        return [s for s in self.spans if s.name == name]


class JsonlTracer(Tracer):
    """One JSON object per line, appended; cheap, greppable, good enough for local debugging."""

    def __init__(self, path: str | Path = "traces.jsonl") -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def export(self, span: Span) -> None:
        line = json.dumps(span.to_dict(), ensure_ascii=False, default=str)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


class OTelTracer(Tracer):
    """Bridge to OpenTelemetry. Spans are created when they *finish* so the attributes are
    complete; timestamps are carried over from our Span. Because of that, OTel spans are
    exported flat (no OTel parent/child nesting); our own linkage is carried as the
    attributes `aie.trace_id` and `aie.parent_span_id`, which is what Chapter 31 reads."""

    def __init__(self, service_name: str = "aie_core") -> None:
        try:
            from opentelemetry import trace as otel_trace
        except ImportError as exc:  # pragma: no cover - exercised only without the SDK
            raise ImportError("OTelTracer requires `opentelemetry-sdk` (pip install aie-core[otel])") from exc
        self._otel = otel_trace.get_tracer(service_name)

    def export(self, span: Span) -> None:
        from opentelemetry.trace import Status, StatusCode

        start_ns = int(span.start * 1e9)
        end_ns = int((span.end or time.time()) * 1e9)
        otel_span = self._otel.start_span(span.name, start_time=start_ns)
        otel_span.set_attribute("aie.trace_id", span.trace_id)
        otel_span.set_attribute("aie.span_id", span.span_id)
        if span.parent_span_id:
            otel_span.set_attribute("aie.parent_span_id", span.parent_span_id)
        for key, value in span.attributes.items():
            if isinstance(value, (str, bool, int, float)):
                otel_span.set_attribute(key, value)
            elif value is not None:
                otel_span.set_attribute(key, json.dumps(value, default=str))
        for ev in span.events:
            otel_span.add_event(ev.get("type", "event"), {k: str(v) for k, v in ev.items() if k != "type"})
        if span.status == "error":
            otel_span.set_status(Status(StatusCode.ERROR))
        otel_span.end(end_time=end_ns)


def current_span() -> Span | None:
    """The span open in the current thread or task, if any."""
    return _current_span.get()


def get_tracer(settings: Any | None = None) -> Tracer:
    """Build a tracer from Settings (imported lazily to avoid a cycle)."""
    if settings is None:
        from .settings import Settings

        settings = Settings()
    sink = getattr(settings, "trace_sink", "none")
    if sink == "jsonl":
        return JsonlTracer(getattr(settings, "trace_path", "traces.jsonl"))
    if sink == "otel":
        return OTelTracer()
    if sink == "memory":
        return InMemoryTracer()  # a new instance per call: read spans from the client's `.tracer`
    return NoopTracer()


__all__ = ["Span", "Tracer", "NoopTracer", "InMemoryTracer", "JsonlTracer", "OTelTracer", "get_tracer", "current_span"]
