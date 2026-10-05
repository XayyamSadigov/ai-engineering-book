# path: book/projects/p6-research-team/research_team/tracing.py
"""Trace propagation across agents.

aie_core links a span to its parent through a context variable, which works inside one thread
and one process. A team run crosses both kinds of boundary: researchers run in a thread pool
(team.py copies the context into each task so the native links survive), and a production team
may run children in other processes or services, where no context variable reaches. So every
agent also gets a tracer wrapper that stamps each span it emits with the team's trace id, the
parent span id of the dispatch that created it, the task id, and the role. These attributes are
what crosses a process or network boundary (Chapter 31 maps them onto OpenTelemetry context),
and a trace backend (or a grep) can rebuild the tree from them alone:
team.run > team.dispatch > agent.run > agent.step > agent.tool.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from aie_core.observability import Span, Tracer


class PropagatingTracer(Tracer):
    def __init__(self, base: Tracer, *, trace_id: str, parent_span_id: str | None, **attributes: Any) -> None:
        self.base = base
        self.trace_id = trace_id
        self.parent_span_id = parent_span_id
        self.attributes = attributes

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        stamped = {"trace.id": self.trace_id, "parent.span_id": self.parent_span_id, **self.attributes, **attributes}
        with self.base.span(name, **stamped) as s:
            yield s

    def export(self, span: Span) -> None:  # spans are exported by the base tracer
        return None


def span_tree(spans: list[Span]) -> dict[str | None, list[Span]]:
    """Group spans by `parent.span_id` so tests and debugging tools can walk the hierarchy."""
    tree: dict[str | None, list[Span]] = {}
    for s in spans:
        tree.setdefault(s.attributes.get("parent.span_id"), []).append(s)
    return tree


__all__ = ["PropagatingTracer", "span_tree"]
