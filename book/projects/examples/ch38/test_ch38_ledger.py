# path: book/projects/examples/ch38/test_ch38_ledger.py
"""Task ledger, compaction that keeps requests valid and literals alive, and rehydration."""
from __future__ import annotations

from typing import Any

import pytest

from aie_core.llm.providers import FakeLLM
from aie_core.llm.tokens import count_message_tokens
from aie_core.llm.types import CompletionRequest, Message, Role, ToolCall
from agentkit import AgentRuntime, Budget, FunctionTool, LoopConfig

from durable import Database, SimulatedCrash, SqliteEventStore
from ledger import (
    CompactingLLM, LedgerUpdate, TaskLedger, compact_messages, digest_observation, harness_facts,
    ledger_from_events, make_ledger_tool,
)

FILLER = "The service degraded under load while the scan table grew without a covering index. " * 5


def search_tool(crash_on: set[int] | None = None) -> FunctionTool:
    seen = {"n": 0}

    def search_incidents(query: str) -> str:
        seen["n"] += 1
        if crash_on and seen["n"] in crash_on:
            crash_on.discard(seen["n"])
            raise SimulatedCrash("worker killed mid-search")
        n = abs(hash(query)) % 1000 if query != "q3" else 7
        return (f"[incident-{query}] Result for {query}. {FILLER}"
                f"Related INC-{1000 + n} in runbooks/trackline-{query}.md; ConnectionTimeoutError at line {n}.")

    return FunctionTool("search_incidents", "Search incident reports.",
                        {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"],
                         "additionalProperties": False}, fn=search_incidents)


def planner(req: CompletionRequest) -> Any:
    step = int(req.metadata["step"])
    if step == 1:
        return [ToolCall(id="l1", name="update_ledger", arguments={
            "add_items": ["collect incidents", "identify root cause", "write summary"],
            "set_status": {}, "decisions": ["scope: logistics tenant only"]})]
    if 2 <= step <= 11:
        return [ToolCall(id=f"s{step}", name="search_incidents", arguments={"query": f"q{step}"})]
    if step == 12:
        return [ToolCall(id="l2", name="update_ledger", arguments={
            "set_status": {"p1": "done", "p2": "done"}, "facts": {"root_cause": "missing scan index"}})]
    return "Root cause: missing scan index, see INC-1007 [incident-q3]."


def valid_pairing(messages: list[Message]) -> bool:
    open_ids: set[str] = set()
    for m in messages:
        if m.role is Role.ASSISTANT:
            open_ids = {tc.id for tc in m.tool_calls or []}
        elif m.role is Role.TOOL and m.tool_call_id not in open_ids:
            return False
    return True


def build(store, inner: FakeLLM, crash_on=None):
    llm = CompactingLLM(inner, max_input_tokens=1200, keep_recent_steps=2,
                        ledger_source=lambda rid: ledger_from_events(store.load(rid)),
                        facts_source=lambda rid: harness_facts(store.load(rid)))
    rt = AgentRuntime(llm, [make_ledger_tool(store), search_tool(crash_on)], store=store,
                      budget=Budget(max_steps=20), config=LoopConfig(max_no_progress_steps=5))
    return rt, llm


def test_long_run_stays_within_budget_and_keeps_literals():
    store = SqliteEventStore(Database())
    inner = FakeLLM(handler=planner)
    rt, llm = build(store, inner)
    result = rt.run("Why were Trackline lookups slow last week? Summarize with incident ids.", run_id="long-1")
    assert result.ok, result.detail
    assert llm.stats, "compaction never triggered"
    assert all(count_message_tokens(r.messages) <= 1200 for r in inner.requests)
    assert all(valid_pairing(r.messages) for r in inner.requests)
    late = inner.requests[-1].messages
    compacted = next(m.text for m in late if m.text.startswith("[harness:compacted]"))
    assert "INC-1007" in compacted and "runbooks/trackline-q3.md" in compacted   # literals survive
    assert "[done] p1 collect incidents" in compacted                            # ledger is authoritative
    assert "root_cause=missing scan index" in compacted
    assert len(result.state.messages) > len(late)          # the log itself was never compacted


def test_ledger_rehydrates_from_the_event_log_after_a_crash():
    db = Database()
    store = SqliteEventStore(db)
    inner = FakeLLM(handler=planner)
    rt, _ = build(store, inner, crash_on={4})
    with pytest.raises(SimulatedCrash):
        rt.run("Why were Trackline lookups slow?", run_id="long-2")
    fresh_store = SqliteEventStore(db)                     # a new process: nothing in memory
    ledger = ledger_from_events(fresh_store.load("long-2"))
    assert [p.text for p in ledger.plan] == ["collect incidents", "identify root cause", "write summary"]
    assert ledger.decisions == ["scope: logistics tenant only"]
    rt2, _ = build(fresh_store, inner)
    assert rt2.resume("long-2").ok


def test_ledger_rejects_unknown_items_and_versions_each_edit():
    led = TaskLedger(objective="x").apply(LedgerUpdate(add_items=["a", "b"]))
    assert led.version == 1 and [p.id for p in led.plan] == ["p1", "p2"]
    with pytest.raises(ValueError):
        led.apply(LedgerUpdate(set_status={"p9": "done"}))
    led2 = led.apply(LedgerUpdate(questions=["which tenant?"])).apply(LedgerUpdate(resolve_questions=["which tenant?"]))
    assert led2.open_questions == [] and led2.version == 3


def test_digest_keeps_identifiers_from_anywhere_in_the_output():
    content = "first line\n" + "noise " * 200 + "\nfailed in app/sla.py line 42: KeyError; TCK-311; 3 failed"
    d = digest_observation("run_tests", {"selector": "tests"}, content)
    for literal in ["app/sla.py", "line 42", "KeyError", "TCK-311", "3 failed"]:
        assert literal in d
    assert len(d) < 400


def test_compaction_is_a_no_op_for_short_histories():
    msgs = [Message.system("s"), Message.user("goal"),
            Message(role=Role.ASSISTANT, content="", tool_calls=[ToolCall(id="a", name="t", arguments={})]),
            Message.tool("a", "ok")]
    assert compact_messages(msgs, keep_recent_steps=2, ledger=None, facts=None) == msgs
