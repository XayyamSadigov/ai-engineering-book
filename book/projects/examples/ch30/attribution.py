# path: book/projects/examples/ch30/attribution.py
"""Attach request-scoped attributes (tenant, request_id, feature) to every span.

The gateway's `llm.complete` span knows the model, tokens and cost, and current `aie_core`
links it to its parent through trace and span ids, but it does not know which tenant, request
or feature it served. Latency tracking and chargeback both need
that join. `AttributingTracer` wraps any `Tracer` and stamps the attributes bound with
`bind()` onto each span at export time, using a context variable so concurrent asyncio
tasks keep their own values.
"""
from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Any, Iterator

from aie_core.observability import Span, Tracer

_BOUND: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar("ch30_bound_attrs", default={})


@contextmanager
def bind(**attributes: Any) -> Iterator[dict[str, Any]]:
    """Bind attributes for the current context; nested binds merge and restore on exit."""
    merged = {**_BOUND.get(), **attributes}
    token = _BOUND.set(merged)
    try:
        yield merged
    finally:
        _BOUND.reset(token)


def bound_attributes() -> dict[str, Any]:
    return dict(_BOUND.get())


class AttributingTracer(Tracer):
    """Delegating tracer: span attributes win over bound ones, so a span can override."""

    def __init__(self, inner: Tracer) -> None:
        self.inner = inner

    def export(self, span: Span) -> None:
        for key, value in _BOUND.get().items():
            span.attributes.setdefault(key, value)
        self.inner.export(span)


__all__ = ["AttributingTracer", "bind", "bound_attributes"]
