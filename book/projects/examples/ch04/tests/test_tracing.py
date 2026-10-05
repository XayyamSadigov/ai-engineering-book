# path: book/projects/examples/ch04/tests/test_tracing.py
from __future__ import annotations

from aie_core.llm.gateway import ModelGateway
from aie_core.llm.providers import FakeLLM
from aie_core.observability import InMemoryTracer
from prompts import traced_complete


def test_prompt_identity_is_on_the_span_and_the_request(registry, ticket_cases):
    tracer = InMemoryTracer()
    llm = FakeLLM(responses=[{"category": "pos_payments", "evidence": "declines every card"}])
    rendered = registry.get("ticket.classify", "prod").render(ticket_cases[0].variables)
    completion = traced_complete(llm, rendered, tracer)
    (span,) = tracer.find("prompt.call")
    assert span.attributes["prompt.id"] == "ticket.classify"
    assert span.attributes["prompt.version"] == "1.0.0"
    assert len(span.attributes["prompt.hash"]) == 16
    assert span.attributes["input_tokens"] == completion.usage.input_tokens > 0
    assert span.attributes["cached_input_tokens"] == completion.usage.cached_input_tokens
    assert llm.last_request.metadata["prompt.version"] == "1.0.0"


def test_works_through_the_gateway_and_both_spans_exist(registry, ticket_cases):
    tracer = InMemoryTracer()
    gateway = ModelGateway(FakeLLM(responses=['{"category": "other", "evidence": ""}']), tracer=tracer)
    rendered = registry.get("ticket.classify", "1.1.0").render(ticket_cases[1].variables)
    traced_complete(gateway, rendered, tracer)
    assert [s.name for s in tracer.spans] == ["llm.complete", "prompt.call"]  # inner span finishes first
    call, inner = tracer.find("prompt.call")[0], tracer.find("llm.complete")[0]
    assert call.attributes["prompt.version"] == "1.1.0"
    assert inner.parent_span_id == call.span_id and inner.trace_id == call.trace_id


def test_failed_call_still_exports_an_error_span(registry, ticket_cases):
    from aie_core.llm.errors import RateLimitError

    tracer = InMemoryTracer()
    llm = FakeLLM(responses=[RateLimitError("slow down")])
    rendered = registry.get("ticket.classify", "prod").render(ticket_cases[0].variables)
    try:
        traced_complete(llm, rendered, tracer)
    except RateLimitError:
        pass
    (span,) = tracer.find("prompt.call")
    assert span.status == "error" and span.attributes["prompt.version"] == "1.0.0"
