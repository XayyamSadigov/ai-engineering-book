# path: book/projects/examples/ch04/prompts/tracing.py
"""Attach prompt identity to traces.

Every model call made from a registered prompt runs inside a `prompt.call` span that
carries `prompt.id`, `prompt.version`, and `prompt.hash`. The same keys travel in
`CompletionRequest.metadata`, so a gateway, a provider log, or an eval record can join
on them. aie_core links spans by trace and parent id, so the gateway's `llm.complete`
span becomes a child of `prompt.call`; Chapter 31 adds a full tracing schema on top. Here
the job is narrower: answer "which prompt version produced this output?" for any request.
"""
from __future__ import annotations

from typing import Any

from aie_core.llm.client import LLMClient
from aie_core.llm.types import Completion
from aie_core.observability import NoopTracer, Tracer

from .registry import RenderedPrompt

SPAN_NAME = "prompt.call"


def traced_complete(
    client: LLMClient,
    rendered: RenderedPrompt,
    tracer: Tracer | None = None,
    **overrides: Any,
) -> Completion:
    """Send a rendered prompt through any LLMClient (usually a ModelGateway) inside a span."""
    tracer = tracer or NoopTracer()
    req = rendered.to_request(**overrides)
    attributes = {
        **rendered.ref.span_attributes(),
        "prompt.render_ms": round(rendered.render_ms, 3),
        "prompt.messages": len(rendered.messages),
        "temperature": req.temperature,
    }
    if overrides:
        attributes["prompt.overrides"] = ",".join(sorted(overrides))
    with tracer.span(SPAN_NAME, **attributes) as span:
        completion = client.complete(req)
        span.set_attribute("provider", completion.provider)
        span.set_attribute("model", completion.model)
        span.set_attribute("input_tokens", completion.usage.input_tokens)
        span.set_attribute("output_tokens", completion.usage.output_tokens)
        # A drop to zero at a version change is the signature of a broken stable prefix.
        span.set_attribute("cached_input_tokens", completion.usage.cached_input_tokens)
        span.set_attribute("finish_reason", completion.finish_reason)
    return completion


__all__ = ["SPAN_NAME", "traced_complete"]
