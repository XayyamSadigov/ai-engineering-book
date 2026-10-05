# path: book/projects/guardrails/guardrails/telemetry.py
"""Redact before the sink. Wraps any `aie_core` tracer so span attributes, events, exception
messages, and pending captured content are scrubbed of secrets and PII *before* the wrapped
tracer sees them, not by a cleanup job after they were stored.

Placement matters. A tracer such as Chapter 31's `OTelAITracer` copies attributes into its
backend (an OpenTelemetry span) when the span ends, before any sink's `export` runs. Scrubbing
only in `export` would therefore leave raw values in the backend. `RedactingTracer.span()` opens
the span on the wrapped tracer and scrubs on every write and again when the body exits, which is
before the wrapped tracer's own end-of-span work, so every backend and every sink receives
scrubbed data. `export()` still scrubs spans built by hand elsewhere (a gateway's streaming path)
before handing them on. Pass the `RedactingTracer` itself everywhere a tracer is needed; other
attributes (`capture_policy`, `resource`, ...) are delegated to the wrapped tracer.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from aie_core.observability import Span, Tracer

from .pii import redact_pii
from .secrets import redact_secrets

DEFAULT_DROP_KEYS: tuple[str, ...] = ("prompt", "messages", "completion.text")
DROPPED = "[dropped by policy]"


def scrub(value: Any, max_depth: int = 6) -> Any:
    """Recursively redact strings inside dicts, lists and tuples."""
    if max_depth <= 0:
        return "[depth limit]"
    if isinstance(value, str):
        text, _ = redact_secrets(value)
        return redact_pii(text).text
    if isinstance(value, dict):
        return {k: scrub(v, max_depth - 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(scrub(v, max_depth - 1) for v in value)
    return value


class RedactingTracer(Tracer):
    def __init__(self, inner: Tracer, drop_keys: tuple[str, ...] = DEFAULT_DROP_KEYS) -> None:
        self.inner = inner
        self.drop_keys = drop_keys

    # ------------------------------------------------------------------ scrubbing
    def _clean(self, key: str, value: Any) -> Any:
        return DROPPED if key in self.drop_keys else scrub(value)

    def scrub_span(self, span: Span) -> None:
        """Scrub a span in place: attributes, events, and any content waiting to be captured."""
        for key in list(span.attributes):
            span.attributes[key] = self._clean(key, span.attributes[key])
        span.events = [scrub(e) for e in span.events]
        pending = getattr(span, "pending_content", None)  # Chapter 31 LinkedSpan: rendered at span end
        if isinstance(pending, dict):
            for key in list(pending):
                pending[key] = DROPPED if key in self.drop_keys else scrub(pending[key])

    def _guard(self, span: Span) -> None:
        """Make later writes through the span's methods scrub on write (instance-level override)."""
        def set_attribute(key: str, value: Any) -> None:
            span.attributes[key] = self._clean(key, value)

        original_record = span.record_exception

        def record_exception(exc: BaseException) -> None:
            original_record(exc)
            span.events[-1] = scrub(span.events[-1])

        span.set_attribute = set_attribute  # type: ignore[method-assign]
        span.record_exception = record_exception  # type: ignore[method-assign]

    # ------------------------------------------------------------------ Tracer interface
    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        clean = {k: self._clean(k, v) for k, v in attributes.items()}
        with self.inner.span(name, **clean) as s:
            self._guard(s)
            try:
                yield s
            finally:
                # Runs before the wrapped tracer's end-of-span hooks (finalize, backend copy, export).
                self.scrub_span(s)

    def export(self, span: Span) -> None:
        self.scrub_span(span)
        self.inner.export(span)

    def __getattr__(self, name: str) -> Any:
        # Only called for attributes not found on self: delegate tracer-specific API to the inner tracer.
        if name in ("inner", "drop_keys"):
            raise AttributeError(name)
        return getattr(self.inner, name)


__all__ = ["scrub", "RedactingTracer", "DEFAULT_DROP_KEYS"]
