# path: book/projects/examples/ch37/tests/test_agentic_runtime.py
"""The agentic retrieval pattern on agentkit's AgentRuntime: same ledger and gate, harness loop."""
from __future__ import annotations

from aie_core.llm.providers import FakeLLM
from aie_core.llm.types import ToolCall
from agentic_rag import Budget
from agentic_rag_runtime import RuntimeAgenticRAG, ledger_from_events
from agentkit import Note, TerminationReason, replay
from corpus import EMPLOYEE, ONCALL, LexicalIndex, section_chunks

QUESTION = "Why did INC-2025-1142 happen and how is the PayBridge certificate renewed now?"
INDEX = LexicalIndex(section_chunks())


def search(i: int, query: str) -> list[ToolCall]:
    return [ToolCall(id=f"s{i}", name="search_evidence", arguments={"query": query})]


def test_multi_hop_run_is_answered_logged_and_replayable():
    llm = FakeLLM(responses=[
        search(1, "INC-2025-1142 root cause"),
        search(2, "PayBridge certificate renewal inventory"),
        "The certificate expired unnoticed [E1] [E2] [E3]; renewal now follows the inventory [E4] [E5] [E6].",
    ])
    rag = RuntimeAgenticRAG(llm, INDEX)
    result, ledger = rag.run(QUESTION, ONCALL, run_id="ar-1")
    assert result.ok and result.state.usage.tool_calls == 2
    assert {"inc-2025-11-pos-outage", "prod-retail-pos-overview"} <= {e.doc_id for e in ledger.entries}
    payments = next(e for e in ledger.entries if e.section.endswith("Payments"))
    assert payments.query == "PayBridge certificate renewal inventory"   # provenance survives in the log
    assert replay(rag.store.load("ar-1")).identical                      # the harness gives replay for free


def test_premature_answer_is_rejected_by_the_gate_inside_the_definition_of_done():
    llm = FakeLLM(responses=[
        search(1, "lumen pos release process pilot stores"),
        "Too early [E1].",
        search(2, "INC-2025-1142 root cause PayBridge certificate renewal"),
        "Grounded [E1] [E4] [E5].",
    ])
    result, _ = RuntimeAgenticRAG(llm, INDEX).run(QUESTION, ONCALL)
    rejected = [n for n in result.events_of(Note) if n.kind == "dod_rejected"]
    assert len(rejected) == 1 and "does not mention" in rejected[0].text
    assert result.ok and result.final_answer == "Grounded [E1] [E4] [E5]."


def test_unknown_label_is_rejected_and_abstaining_is_accepted():
    llm = FakeLLM(responses=[search(1, "INC-2025-1142 root cause"), "Made up [E9].",
                             "INSUFFICIENT_EVIDENCE: cannot verify the renewal process."])
    result, _ = RuntimeAgenticRAG(llm, INDEX).run(QUESTION, ONCALL)
    assert any("unknown labels: ['E9']" in n.text for n in result.events_of(Note) if n.kind == "dod_rejected")
    assert result.ok and result.final_answer.startswith("INSUFFICIENT_EVIDENCE")


def test_ledger_never_contains_documents_the_user_cannot_read():
    llm = FakeLLM(responses=[search(1, "INC-2025-1142 root cause"), "INSUFFICIENT_EVIDENCE"])
    _, ledger = RuntimeAgenticRAG(llm, INDEX).run(QUESTION, EMPLOYEE)
    assert all(e.doc_id != "inc-2025-11-pos-outage" for e in ledger.entries)


def test_repeated_query_is_stopped_by_the_harness_not_by_custom_loop_code():
    llm = FakeLLM(responses=[search(1, "paybridge certificate"), search(2, "paybridge certificate"), "unused"])
    result, _ = RuntimeAgenticRAG(llm, INDEX).run(QUESTION, ONCALL)
    assert result.stop_reason is TerminationReason.REPEATED_ACTION
    assert result.state.usage.tool_calls == 1


def test_search_budget_maps_onto_the_tool_call_budget():
    calls = [search(i, f"certificate topic {i} paybridge stores") for i in range(6)]
    result, _ = RuntimeAgenticRAG(FakeLLM(responses=[*calls, "unused"]), INDEX,
                                  Budget(max_searches=2, max_steps=8)).run(QUESTION, ONCALL)
    assert result.stop_reason is TerminationReason.MAX_TOOL_CALLS
    assert result.state.usage.tool_calls == 2


def test_ledger_is_rebuilt_from_the_event_log_alone():
    llm = FakeLLM(responses=[search(1, "INC-2025-1142 root cause"), "INSUFFICIENT_EVIDENCE"])
    rag = RuntimeAgenticRAG(llm, INDEX)
    _, ledger = rag.run(QUESTION, ONCALL, run_id="ar-2")
    rebuilt = ledger_from_events(rag.store.load("ar-2"), Budget().max_evidence_tokens)
    assert [e.label for e in rebuilt.entries] == [e.label for e in ledger.entries] and rebuilt.entries
