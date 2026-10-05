# path: book/projects/examples/ch31/tests/test_instrument.py
import json
import random

import pytest

from aie_core.llm.errors import RateLimitError
from aie_core.llm.gateway import ModelGateway, PricingTable, RetryPolicy
from aie_core.llm.providers import FakeLLM
from aie_core.llm.types import CompletionRequest, Message
from aie_core.observability import InMemoryTracer, JsonlTracer, Span

from instrument import (AITracer, CapturePolicy, agent_run, agent_step, context_span, guardrail_span, redact,
                        retrieval_span, tool_span, trace_request, traced, traced_generation, traced_tool)
from semconv import Attr, ErrorClass, SpanName, classify_exception, genai_aliases, normalize
from trace_store import SpanRecord, TraceStore


def make(capture=None, **kw):
    sink = InMemoryTracer()
    return AITracer(sink, capture=capture or CapturePolicy(salt="t"), rng=random.Random(1), **kw), sink


def test_spans_form_one_tree_with_gateway_attempts_nested():
    tracer, sink = make()
    gw = ModelGateway(FakeLLM(["hello"], model="m"), tracer=tracer, pricing=PricingTable({"m": {"input_per_1m": 1.0}}))
    with trace_request(tracer, route="rag.answer", tenant="retail", versions={Attr.PROMPT_VERSION: "7"}):
        traced_generation(tracer, gw, CompletionRequest(messages=[Message.user("hi")], model="m"),
                          prompt_id="rag-answer", prompt_version="7")
    names = [s.name for s in sink.spans]
    assert names == [SpanName.LLM_ATTEMPT, SpanName.GENERATE, SpanName.REQUEST]
    attempt, gen, root = sink.spans
    assert len({s.trace_id for s in sink.spans}) == 1
    assert attempt.parent_span_id == gen.span_id and gen.parent_span_id == root.span_id
    # legacy gateway keys are normalized to the book's conventions
    assert attempt.attributes[Attr.LLM_MODEL] == "m" and Attr.LLM_INPUT_TOKENS in attempt.attributes
    assert "model" not in attempt.attributes


def test_prompt_identity_and_versions_propagate_to_llm_attempt():
    tracer, sink = make()
    gw = ModelGateway(FakeLLM(["ok"]), tracer=tracer)
    with trace_request(tracer, route="r", tenant="retail", versions={Attr.INDEX_VERSION: "idx-9"}):
        with tracer.span("prompt.call", **{Attr.PROMPT_ID: "assist.answer", "prompt.hash": "abc123"}):
            gw.complete(CompletionRequest(messages=[Message.user("x")]))
    attempt = sink.spans[0]
    assert attempt.attributes[Attr.PROMPT_ID] == "assist.answer"
    assert attempt.attributes["prompt.hash"] == "abc123"
    assert attempt.attributes[Attr.INDEX_VERSION] == "idx-9"
    assert attempt.attributes[Attr.TENANT] == "retail"


def test_streamed_attempt_gets_the_same_propagated_attributes():
    # The gateway builds the streaming span by hand and hands it to export() when the stream ends;
    # it must carry tenant, prompt identity, and versions exactly like a non-streamed attempt.
    tracer, sink = make()
    gw = ModelGateway(FakeLLM(["streamed answer"]), tracer=tracer)
    with trace_request(tracer, route="r", tenant="retail", versions={Attr.INDEX_VERSION: "idx-9"}) as root:
        with tracer.span("prompt.call", **{Attr.PROMPT_ID: "assist.answer", Attr.PROMPT_VERSION: "8"}):
            events = list(gw.stream(CompletionRequest(messages=[Message.user("x")])))
    assert any(e.type == "text_delta" for e in events)
    attempt = next(s for s in sink.spans if s.name == SpanName.LLM_ATTEMPT)
    assert attempt.trace_id == root.trace_id
    for key, value in {Attr.TENANT: "retail", Attr.PROMPT_ID: "assist.answer",
                       Attr.PROMPT_VERSION: "8", Attr.INDEX_VERSION: "idx-9"}.items():
        assert attempt.attributes[key] == value, key


def test_retry_attempt_is_classified_and_recovered():
    tracer, sink = make()
    gw = ModelGateway(FakeLLM([RateLimitError("429"), "ok"]), tracer=tracer, retry=RetryPolicy(base_delay_s=0.0),
                      sleep=lambda s: None)
    with trace_request(tracer, route="r", tenant="retail"):
        traced_generation(tracer, gw, CompletionRequest(messages=[Message.user("x")]), prompt_id="p", prompt_version="1")
    failed = [s for s in sink.spans if s.status == "error"]
    assert failed[0].attributes[Attr.ERROR_CLASS] == ErrorClass.RATE_LIMITED.value
    tree = TraceStore.from_spans(sink.spans).traces()[0]
    assert tree.error_classes == set() and tree.recovered_errors == {"rate_limited"}


def test_exception_sets_error_type_and_class():
    tracer, sink = make()
    with pytest.raises(PermissionError):
        with trace_request(tracer, route="r", tenant="retail"):
            raise PermissionError("nope")
    assert sink.spans[0].attributes[Attr.ERROR_CLASS] == ErrorClass.PERMISSION_DENIED.value
    assert sink.spans[0].attributes[Attr.ERROR_TYPE] == "PermissionError"


@pytest.mark.parametrize("mode,has_hash,has_content", [("off", False, False), ("hashed", True, False),
                                                       ("redacted", True, True), ("full", True, True)])
def test_capture_modes(mode, has_hash, has_content):
    tracer, sink = make(CapturePolicy(mode=mode, salt="s", on_error_mode=None))
    with trace_request(tracer, route="r", tenant="retail"):
        with retrieval_span(tracer, "mail jane@northwind.example", index_version="i", top_k=3) as r:
            r.results(["a"], [0.9])
    span = sink.spans[0]
    assert span.attributes[f"{Attr.RETRIEVAL_QUERY}.chars"] == len("mail jane@northwind.example")
    assert (f"{Attr.RETRIEVAL_QUERY}.hash" in span.attributes) is has_hash
    assert (f"{Attr.RETRIEVAL_QUERY}.content" in span.attributes) is has_content
    if mode == "redacted":
        assert "[EMAIL]" in span.attributes[f"{Attr.RETRIEVAL_QUERY}.content"]
    assert span.attributes[Attr.CAPTURE_MODE] == mode


def test_tenant_ceiling_beats_sampling_and_errors():
    policy = CapturePolicy(mode="hashed", sample_rate=1.0, sampled_mode="full", tenant_ceiling={"logistics": "hashed"})
    assert policy.decide("t1", "retail", errored=False) == "full"
    assert policy.decide("t1", "logistics", errored=True) == "hashed"


def test_capture_on_error_upgrades_only_failing_spans():
    tracer, sink = make(CapturePolicy(mode="hashed", on_error_mode="redacted", salt="s"))
    with trace_request(tracer, route="r", tenant="retail"):
        with tool_span(tracer, "ok_tool", {"q": "fine"}) as t:
            t.result("fine")
        with pytest.raises(RuntimeError):
            with tool_span(tracer, "bad_tool", {"q": "boom"}):
                raise RuntimeError("boom")
    ok, bad = sink.spans[0], sink.spans[1]
    assert ok.attributes[Attr.CAPTURE_MODE] == "hashed" and f"{Attr.TOOL_ARGS}.content" not in ok.attributes
    assert bad.attributes[Attr.CAPTURE_MODE] == "redacted" and bad.attributes[Attr.TOOL_STATUS] == "error"
    assert bad.attributes[Attr.ERROR_CLASS] == ErrorClass.TOOL_FAILURE.value


def test_sampling_is_deterministic_per_trace():
    policy = CapturePolicy(mode="off", sample_rate=0.3, sampled_mode="redacted")
    ids = [f"{i:032x}" for i in range(2000)]
    first = [policy.decide(t, None, False) for t in ids]
    assert first == [policy.decide(t, None, False) for t in ids]
    assert 0.25 < first.count("redacted") / len(ids) < 0.35


def test_redaction_patterns():
    text = "Contact a.kim@northwind.example, +1 415 555 0134, card 4111 1111 1111 1111, key sk-abcdef1234567890, NW-123456"
    out = redact(text)
    for marker in ("[EMAIL]", "[PHONE]", "[CARD]", "[SECRET]", "[EMPLOYEE_ID]"):
        assert marker in out
    assert "northwind.example" not in out and "4111" not in out


def test_keyed_hash_differs_by_salt():
    assert CapturePolicy(salt="a").digest("x") != CapturePolicy(salt="b").digest("x")


def test_cross_tenant_retrieval_is_marked_contamination():
    tracer, sink = make()
    with trace_request(tracer, route="r", tenant="retail"):
        with retrieval_span(tracer, "q", index_version="i", top_k=2) as r:
            r.results(["a", "b"], [0.9, 0.8], tenants=["shared", "logistics"])
    assert sink.spans[0].attributes[Attr.ERROR_CLASS] == ErrorClass.RETRIEVAL_CONTAMINATION.value


def test_context_guardrail_agent_helpers():
    tracer, sink = make()
    with trace_request(tracer, route="agent", tenant="logistics"):
        with context_span(tracer, budget_tokens=100) as c:
            c.packed(["a"], ["b"], 90)
        with agent_run(tracer, "a", max_steps=2) as run:
            with agent_step(tracer, 1, "search"):
                with guardrail_span(tracer, "tool_policy", stage="tool") as g:
                    g.decide("block", "side effect", error_class=ErrorClass.UNSAFE_CONTENT)
            run.stop("max_steps", error_class=ErrorClass.LOOP)
    by = {s.name: s for s in sink.spans}
    assert by[SpanName.CONTEXT].attributes[Attr.CONTEXT_TRUNCATED] is True
    assert by[SpanName.GUARDRAIL].attributes[Attr.GUARD_DECISION] == "block"
    assert by[SpanName.AGENT_RUN].attributes[Attr.ERROR_CLASS] == "loop"


def test_decorators():
    tracer, sink = make()

    @traced_tool(tracer, "lookup")
    def lookup(*, name: str) -> dict:
        return {"name": name}

    @traced(tracer, "custom.stage", stage="x")
    def work() -> int:
        return 3

    with trace_request(tracer, route="r", tenant="retail"):
        assert lookup(name="a") == {"name": "a"}
        assert work() == 3
    tool = next(s for s in sink.spans if s.name == SpanName.TOOL)
    assert tool.attributes[Attr.TOOL_NAME] == "lookup" and tool.attributes[Attr.TOOL_STATUS] == "ok"
    assert any(s.name == "custom.stage" and s.attributes["stage"] == "x" for s in sink.spans)


def test_handmade_span_is_adopted_into_current_trace():
    tracer, sink = make()
    with trace_request(tracer, route="r", tenant="retail") as root:
        raw = Span(name=SpanName.LLM_ATTEMPT, attributes={"model": "m", "stream": True})
        raw.finish()
        tracer.export(raw)
    adopted = sink.spans[0]
    assert adopted.trace_id == root.trace_id and adopted.parent_span_id == root.span_id
    assert adopted.attributes[Attr.LLM_MODEL] == "m"


def test_jsonl_round_trip(tmp_path):
    path = tmp_path / "t.jsonl"
    tracer = AITracer(JsonlTracer(path), capture=CapturePolicy(salt="s"))
    with trace_request(tracer, route="r", tenant="retail", response_id="resp-1"):
        with tracer.span(SpanName.ROUTER):
            pass
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert {r["name"] for r in rows} == {SpanName.ROUTER, SpanName.REQUEST}
    assert all(r["trace_id"] for r in rows)
    tree = TraceStore.from_jsonl(path).traces()[0]
    assert tree.root.name == SpanName.REQUEST and tree.get(Attr.RESPONSE_ID) == "resp-1"


def test_semconv_helpers():
    assert normalize(SpanName.LLM_ATTEMPT, {"model": "m", "x": 1}) == {Attr.LLM_MODEL: "m", "x": 1}
    assert normalize(SpanName.RETRIEVAL, {"model": "m"}) == {"model": "m"}
    assert genai_aliases({Attr.LLM_MODEL: "m"}) == {"gen_ai.request.model": "m"}
    assert classify_exception(RateLimitError("x")) == ErrorClass.RATE_LIMITED
    assert classify_exception(ValueError("x")) == ErrorClass.UNKNOWN


def test_genai_dual_write():
    tracer, sink = make(emit_genai_aliases=True)
    with tracer.span(SpanName.TOOL, **{Attr.TOOL_NAME: "t"}):
        pass
    assert sink.spans[0].attributes["gen_ai.tool.name"] == "t"


def test_cache_hit_is_avoided_cost_not_spend():
    from aie_core.llm.gateway import InMemoryResponseCache

    tracer, sink = make()
    gw = ModelGateway(FakeLLM(["same", "same"], model="m"), tracer=tracer, cache=InMemoryResponseCache(),
                      pricing=PricingTable({"m": {"input_per_1m": 1000.0, "output_per_1m": 1000.0}}))
    req = CompletionRequest(messages=[Message.user("what is the PTO carry-over?")], model="m")
    for _ in range(2):
        with trace_request(tracer, route="r", tenant="retail"):
            gw.complete(req)
    trees = TraceStore.from_spans(sink.spans).traces()
    miss, hit = trees
    assert miss.cost_usd > 0 and miss.avoided_cost_usd == 0
    assert hit.cost_usd == 0 and hit.avoided_cost_usd == miss.cost_usd
    # also correct when loading spans written by a plain aie_core JsonlTracer
    raw = {"name": "llm.complete", "span_id": "s", "attributes": {"cache_hit": True, "cost_usd": 0.5}}
    rec = SpanRecord.from_dict(raw)
    assert rec.attributes[Attr.LLM_COST] == 0.0 and rec.attributes[Attr.LLM_AVOIDED_COST] == 0.5
