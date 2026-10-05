# path: book/projects/examples/ch31/tests/test_otel.py
from aie_core.llm.gateway import ModelGateway
from aie_core.llm.providers import FakeLLM
from aie_core.llm.types import CompletionRequest, Message
from aie_core.observability import InMemoryTracer, NoopTracer

from instrument import AITracer, CapturePolicy, retrieval_span, trace_request, traced_generation
from otel_setup import OTelAITracer, build_provider, tracer_from_env
from semconv import Attr, SpanName


def test_otel_spans_nest_and_carry_attributes():
    provider, exporter = build_provider("test", exporter="memory")
    mirror = InMemoryTracer()
    tracer = OTelAITracer(provider, sink=mirror, capture=CapturePolicy(salt="s"), emit_genai_aliases=True)
    gw = ModelGateway(FakeLLM(["hi"], model="m"), tracer=tracer)
    with trace_request(tracer, route="rag.answer", tenant="retail", versions={Attr.PROMPT_VERSION: "7"}):
        with retrieval_span(tracer, "q", index_version="idx", top_k=3) as r:
            r.results(["a", "b"], [0.9, 0.5])
        traced_generation(tracer, gw, CompletionRequest(messages=[Message.user("x")], model="m"),
                          prompt_id="rag-answer", prompt_version="7")
    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert set(spans) == {SpanName.REQUEST, SpanName.RETRIEVAL, SpanName.GENERATE, SpanName.LLM_ATTEMPT}
    root = spans[SpanName.REQUEST]
    assert len({s.context.trace_id for s in spans.values()}) == 1
    assert spans[SpanName.RETRIEVAL].parent.span_id == root.context.span_id
    assert spans[SpanName.LLM_ATTEMPT].parent.span_id == spans[SpanName.GENERATE].context.span_id
    assert tuple(spans[SpanName.RETRIEVAL].attributes[Attr.RETRIEVAL_IDS]) == ("a", "b")
    assert spans[SpanName.LLM_ATTEMPT].attributes["gen_ai.request.model"] == "m"
    assert spans[SpanName.LLM_ATTEMPT].attributes[Attr.PROMPT_VERSION] == "7"
    # the JSONL/in-memory mirror uses the same ids as OpenTelemetry
    assert {s.trace_id for s in mirror.spans} == {format(root.context.trace_id, "032x")}


def test_otel_error_status():
    from opentelemetry.trace import StatusCode

    provider, exporter = build_provider("test", exporter="memory")
    tracer = OTelAITracer(provider)
    try:
        with tracer.span(SpanName.TOOL):
            raise TimeoutError("slow")
    except TimeoutError:
        pass
    span = exporter.get_finished_spans()[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes[Attr.ERROR_CLASS] == "timeout"


def test_sampler_ratio_zero_records_nothing():
    provider, exporter = build_provider("test", exporter="memory", sample_ratio=0.0)
    tracer = OTelAITracer(provider)
    with trace_request(tracer, route="r", tenant="retail"):
        with tracer.span(SpanName.ROUTER):
            pass
    assert exporter.get_finished_spans() == ()


def test_tracer_from_env(tmp_path):
    t = tracer_from_env({"TRACE_SINK": "none"})
    assert isinstance(t, AITracer) and isinstance(t.sink, NoopTracer)
    t = tracer_from_env({"TRACE_SINK": "jsonl", "TRACE_PATH": str(tmp_path / "x.jsonl"), "CAPTURE_MODE": "off"})
    assert t.capture_policy.mode == "off"
    t = tracer_from_env({"TRACE_SINK": "otel", "OTEL_EXPORTER": "memory"})
    assert isinstance(t, OTelAITracer)
