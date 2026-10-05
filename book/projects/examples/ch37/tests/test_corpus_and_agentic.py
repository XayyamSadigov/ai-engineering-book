# path: book/projects/examples/ch37/tests/test_corpus_and_agentic.py
from __future__ import annotations

from aie_core.llm.providers import FakeLLM
from agentic_rag import AgenticRAG, Budget, scripted_controller
from corpus import EMPLOYEE, ONCALL, FullTextIndex, LexicalIndex, section_chunks

QUESTION = "Why did INC-2025-1142 happen and how is the PayBridge certificate renewed now?"


def _index() -> LexicalIndex:
    return LexicalIndex(section_chunks())


# ------------------------------------------------------------------ corpus fixtures
def test_lexical_index_filters_by_acl_before_scoring():
    idx = _index()
    oncall = idx.search("INC-2025-1142 root cause", ONCALL, k=5)
    employee = idx.search("INC-2025-1142 root cause", EMPLOYEE, k=5)
    assert oncall[0].chunk.doc_id == "inc-2025-11-pos-outage"
    assert all(h.chunk.doc_id != "inc-2025-11-pos-outage" for h in employee)


def test_fts5_phrase_query_with_sql_acl_and_date_filter():
    chunks = section_chunks()
    fts = FullTextIndex(chunks)
    by_id = {c.id: c for c in chunks}
    hits = fts.search('"certificate inventory"', ONCALL, k=5)
    docs = {by_id[cid].doc_id for cid, _ in hits}
    assert {"prod-retail-pos-overview", "inc-2025-11-pos-outage"} <= docs
    assert "inc-2025-11-pos-outage" not in {by_id[c].doc_id for c, _ in fts.search('"certificate inventory"', EMPLOYEE)}
    recent = fts.search("carryover", EMPLOYEE, updated_after="2026-01-01")
    assert {by_id[c].doc_id for c, _ in recent} == {"hr-pto-policy"}  # the stale FAQ is filtered by metadata


# ------------------------------------------------------------------ agentic loop
def test_multi_hop_question_collects_evidence_from_both_hops():
    llm = FakeLLM(handler=scripted_controller(["INC-2025-1142 root cause", "PayBridge certificate renewal inventory"]))
    result = AgenticRAG(llm, _index()).run(QUESTION, ONCALL)
    assert result.status == "answered"
    assert result.searches == 2 and result.llm_calls == 3
    cited_docs = {e.doc_id for e in result.citations}
    assert {"inc-2025-11-pos-outage", "prod-retail-pos-overview"} <= cited_docs
    payments = next(e for e in result.ledger if e.section.endswith("Payments"))
    assert payments.query == "PayBridge certificate renewal inventory"  # provenance: which query found it


def test_ledger_never_contains_documents_the_user_cannot_read():
    llm = FakeLLM(handler=scripted_controller(["INC-2025-1142 root cause", "PayBridge certificate expired stores"]))
    result = AgenticRAG(llm, _index()).run(QUESTION, EMPLOYEE)
    assert result.ledger, "the employee still sees the POS overview"
    assert all(e.doc_id != "inc-2025-11-pos-outage" for e in result.ledger)


def test_search_budget_is_enforced():
    queries = [f"certificate topic {i} paybridge stores" for i in range(10)]
    llm = FakeLLM(handler=scripted_controller(queries))
    result = AgenticRAG(llm, _index(), Budget(max_searches=2, max_steps=8)).run(QUESTION, ONCALL)
    assert result.status in {"budget_exhausted", "abstained"}
    assert result.searches <= 2
    assert result.llm_calls <= 8


def test_repeated_query_is_not_executed_and_loop_stalls_out():
    llm = FakeLLM(handler=scripted_controller(["paybridge certificate"] * 5))
    result = AgenticRAG(llm, _index()).run(QUESTION, ONCALL)
    assert result.searches == 1
    assert result.stop_reason == "stalled"


def test_unknown_citation_is_rejected_and_loop_continues():
    responses = [
        {"action": "search", "query": "INC-2025-1142 root cause"},
        {"action": "answer", "answer": "made up", "cited": ["E9"]},
        {"action": "abstain", "missing": "cannot verify"},
    ]
    result = AgenticRAG(FakeLLM(responses=responses), _index()).run(QUESTION, ONCALL)
    assert any("unknown or missing citations ['E9']" in s.note for s in result.steps)
    assert result.status == "abstained"


def test_sufficiency_gate_rejects_premature_answer():
    responses = [
        {"action": "search", "query": "lumen pos release process pilot stores"},
        {"action": "answer", "answer": "too early", "cited": ["E1"]},
        {"action": "search", "query": "INC-2025-1142 root cause PayBridge certificate renewal"},
        {"action": "answer", "answer": "grounded", "cited": ["E1", "E4", "E5"]},
    ]
    result = AgenticRAG(FakeLLM(responses=responses), _index()).run(QUESTION, ONCALL)
    assert result.steps[1].note.startswith("rejected: coverage")
    assert result.status == "answered" and result.answer == "grounded"


def test_evidence_token_budget_caps_the_ledger():
    llm = FakeLLM(handler=scripted_controller(["INC-2025-1142", "PayBridge certificate", "store server restart"]))
    result = AgenticRAG(llm, _index(), Budget(max_evidence_tokens=400, min_coverage=0.0)).run(QUESTION, ONCALL)
    assert sum(e.tokens for e in result.ledger) <= 400
