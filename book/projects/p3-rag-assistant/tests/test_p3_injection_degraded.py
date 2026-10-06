# path: book/projects/p3-rag-assistant/tests/test_p3_injection_degraded.py
"""The injection fixture changes nothing; every dependency failure has a defined degraded mode."""
from __future__ import annotations

from aie_core.llm.errors import ProviderUnavailableError
from aie_core.llm.providers import FakeLLM
from conftest import PTO_Q

from rag_assistant.adapters.llm import compromised_handler

DELIVERY_Q = "From June 2026, which weekday do deliveries to North region depots move to?"
INJECTED = ("partners@brightline-supply.example", "employee directory", "phone numbers")


def test_injection_document_does_not_alter_behavior(make_container, logistics_employee, docs_dir):
    with_doc = make_container()
    out = with_doc.answers.ask(DELIVERY_Q, logistics_employee)
    assert "ext-vendor-newsletter-brightline" in out.retrieval.doc_ids
    text = out.response.answer.text
    assert "Wednesday" in text
    assert not any(s in text for s in INJECTED)
    probe = with_doc.answers.ask("What does the Brightline newsletter ask AI assistants and automated readers to do?",
                                 logistics_employee)
    assert any(e.startswith("context_flagged:ext-vendor") for e in probe.security_events)
    assert any(n.kind == "instruction_like_content" for n in probe.qa.packed.notes)
    assert not any(s in probe.response.answer.text for s in INJECTED)
    # an unrelated question gets the same answer whether or not the newsletter is indexed
    baseline = with_doc.answers.ask(PTO_Q, logistics_employee).response.answer.text
    (docs_dir / "vendor-newsletter.md").unlink()
    without_doc = make_container()
    assert without_doc.answers.ask(PTO_Q, logistics_employee).response.answer.text == baseline


def test_controls_hold_even_when_the_model_obeys_the_injection(make_container, logistics_employee):
    c = make_container(llm=FakeLLM(handler=compromised_handler))
    out = c.answers.ask("What does the Brightline newsletter ask automated readers to do?", logistics_employee)
    answer = out.response.answer
    assert answer.action == "abstain"  # the claim is supported only by flagged text, so it is dropped
    assert not any(s in answer.text for s in INJECTED)
    assert out.security_events  # recorded for the security dashboard, never shown to the user


def test_dense_down_falls_back_to_lexical_only(container, employee, monkeypatch):
    def broken(*a, **k):  # type: ignore[no-untyped-def]
        raise ConnectionError("vector store unreachable")

    for ix in container.index_set.existing():
        monkeypatch.setattr(ix.dense, "retrieve", broken)
    out = container.answers.ask(PTO_Q, employee)
    assert out.response.mode == "answer"
    assert any(d.startswith("retrieve:dense") for d in out.response.degraded)
    assert out.retrieval.doc_ids[0] == "hr-pto-policy"
    assert container.metrics.counter("rag_degraded_total", reason="retrieve:dense#q0") >= 1


def test_breaker_opens_and_dense_fails_fast(make_container, employee, monkeypatch):
    c = make_container(breaker_min_calls=2, retrieval_cache=False, answer_cache=False)
    calls = {"n": 0}

    def broken(*a, **k):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        raise ConnectionError("down")

    for ix in c.index_set.existing():
        monkeypatch.setattr(ix.dense, "retrieve", broken)
    for _ in range(4):
        c.answers.ask(PTO_Q, employee)
    assert c.breakers.get("dense").state.value == "open"
    assert calls["n"] < 4  # later requests never touched the broken dependency


def test_reranker_down_uses_fusion_order(container, employee, monkeypatch):
    from ragkit.retrieval import LexicalOverlapReranker

    def broken(self, *a, **k):  # type: ignore[no-untyped-def]
        raise TimeoutError("reranker timeout")

    monkeypatch.setattr(LexicalOverlapReranker, "rerank", broken)
    out = container.answers.ask(PTO_Q, employee)
    assert "rerank:fallback" in out.response.degraded
    assert out.response.mode == "answer" and out.retrieval.hits


def test_llm_down_returns_sources_only(make_container, employee):
    c = make_container(llm=FakeLLM(responses=[ProviderUnavailableError("503", status_code=503)] * 5))
    out = c.answers.ask(PTO_Q, employee)
    r = out.response
    assert r.mode == "sources_only" and r.answer is None
    assert r.sources and r.sources[0].doc_id == "hr-pto-policy"
    assert any(d.startswith("generate:") for d in r.degraded)


def test_all_retrievers_down_is_unavailable(container, employee, monkeypatch):
    def broken(*a, **k):  # type: ignore[no-untyped-def]
        raise ConnectionError("down")

    for ix in container.index_set.existing():
        monkeypatch.setattr(ix.dense, "retrieve", broken)
        monkeypatch.setattr(ix.bm25, "retrieve", broken)
    assert container.answers.ask("anything at all about laptops", employee).response.mode == "unavailable"


def test_spent_budget_skips_generation(make_container, employee):
    c = make_container(request_deadline_s=8.0, min_generation_s=9.0)  # the budget cannot fit a generation
    out = c.answers.ask(PTO_Q, employee)
    assert out.response.mode == "sources_only" and "generate:deadline" in out.response.degraded


def test_degraded_results_are_not_cached(container, employee, monkeypatch):
    def broken(*a, **k):  # type: ignore[no-untyped-def]
        raise ConnectionError("down")

    for ix in container.index_set.existing():
        monkeypatch.setattr(ix.dense, "retrieve", broken)
    container.answers.ask(PTO_Q, employee)
    monkeypatch.undo()
    out = container.answers.ask(PTO_Q, employee)
    assert out.response.cache == "miss" and not out.response.degraded


def test_request_produces_one_trace_with_stage_spans(container, employee):
    container.tracer.spans.clear()
    container.answers.ask(PTO_Q, employee, request_id="req-trace-1")
    names = {s.name for s in container.tracer.spans}
    assert {"rag.request", "retrieval.pipeline", "rag.answer", "rag.generate", "guardrail.input",
            "guardrail.output"} <= names
    root = container.tracer.find("rag.request")[0]
    assert root.attributes["request.id"] == "req-trace-1"
    answer_span = container.tracer.find("rag.answer")[0]
    assert answer_span.trace_id == root.trace_id


def test_slow_dense_retriever_is_cut_by_the_stage_budget(make_container, employee, monkeypatch):
    import time as _time

    c = make_container(retrieve_timeout_s=0.2, retrieval_cache=False, answer_cache=False)
    for ix in c.index_set.existing():
        original = ix.dense.retrieve

        def slow(query, _orig=original):  # type: ignore[no-untyped-def]
            _time.sleep(1.0)
            return _orig(query)

        monkeypatch.setattr(ix.dense, "retrieve", slow)
    t0 = _time.perf_counter()
    out = c.answers.ask(PTO_Q, employee)
    assert _time.perf_counter() - t0 < 0.9  # did not wait for the slow store
    assert any(d.startswith("retrieve:dense") for d in out.response.degraded)
    assert out.response.mode == "answer" and out.retrieval.doc_ids[0] == "hr-pto-policy"


def test_slow_reranker_still_applies_authority(make_container, employee, monkeypatch):
    """A relevance-reranker timeout keeps the fused order but still runs authority and supersession."""
    import time

    from ragkit.retrieval import LexicalOverlapReranker

    original = LexicalOverlapReranker.rerank

    def slow(self, *a, **k):  # type: ignore[no-untyped-def]
        time.sleep(0.3)
        return original(self, *a, **k)

    c = make_container(rerank_timeout_s=0.05)
    monkeypatch.setattr(LexicalOverlapReranker, "rerank", slow)
    out = c.answers.ask(PTO_Q, employee)
    assert out.retrieval.doc_ids[0] == "hr-pto-policy"  # policy above the FAQ it supersedes
    assert "rerank:fallback" in out.response.degraded
    assert not any(d == "rerank:TimeoutError" for d in out.response.degraded)  # not the pipeline-level fallback
