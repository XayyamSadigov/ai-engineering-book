# path: book/projects/agentkit/tests/test_replay_and_store.py
"""Replay, event stores, crash recovery, adapters, and verifier units."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from aie_core.llm.providers import FakeLLM
from aie_core.llm.types import ToolSpec
from agentkit import (
    AgentRuntime, AgentState, DefinitionOfDone, FunctionTool, ErrorClass, InMemoryEventStore, JsonlEventStore, RecordedLLM,
    TerminationReason, ToolCallApproved, ToolResult, adapt_tool, all_of, any_of, citations_grounded, contains_all,
    decision_signatures, derive_state, json_schema, matches, replay, truncate_observation, validate_arguments,
)

from agentkit.dod import CITATION

from .conftest import call

ANSWER = "Confirm replica lag, then switch over [it-database-failover-runbook]."


def _record(tools, store=None, run_id="rec-1"):
    llm = FakeLLM(responses=[
        call("search_runbooks", query="failover replica"),
        call("search_runbooks", query="zzz nothing"),
        ANSWER,
    ])
    rt = AgentRuntime(llm, tools, dod=DefinitionOfDone(citations_grounded(1)), store=store)
    return rt.run("The retail primary database is down. What is the procedure?", run_id=run_id)


def test_replay_reproduces_decisions_without_executing_tools(tools, counter):
    original = _record(tools)
    executed = counter.count("search_runbooks")
    report = replay(original.events)                      # RecordedLLM by default
    assert report.identical, report.summary()
    assert report.result is not None and report.result.final_answer == original.final_answer
    assert counter.count("search_runbooks") == executed   # no tool ran during replay
    assert [r.content for r in report.result.events_of(ToolResult)] == \
           [r.content for r in original.events_of(ToolResult)]


def test_counterfactual_replay_reports_divergence_and_misses(tools):
    original = _record(tools)
    new_planner = FakeLLM(responses=[
        call("search_runbooks", query="failover replica"),    # same first decision: served from the log
        call("search_runbooks", query="patroni switchover"),  # new decision: no recorded result
        ANSWER,
    ])
    report = replay(original.events, new_planner, system_prompt="You are a cautious on-call assistant.")
    assert not report.identical
    assert report.first_divergence == 1
    assert report.misses and report.misses[0]["arguments"] == {"query": "patroni switchover"}
    assert report.result is not None and report.result.events_of(ToolResult)[1].error_class is ErrorClass.IMPOSSIBLE


def test_harness_replay_with_stricter_verifier(tools):
    """Re-run old trajectories through a new Definition of Done: no model or tool calls needed."""
    original = _record(tools)
    stricter = DefinitionOfDone(contains_all("replica lag", "staging sign-off"))
    report = replay(original.events, dod=stricter)
    assert report.result is not None
    assert report.result.stop_reason is TerminationReason.MODEL_ERROR   # recorded decisions ran out
    assert report.result.state.dod_rejections == 1


def test_jsonl_store_roundtrip(tmp_path, tools):
    store = JsonlEventStore(tmp_path)
    result = _record(tools, store=store, run_id="jsonl-run")
    loaded = store.load("jsonl-run")
    assert [e.model_dump() for e in loaded] == [e.model_dump() for e in result.events]
    assert derive_state(loaded).final_answer == ANSWER
    assert store.runs() == ["jsonl-run"]
    assert decision_signatures(loaded) == decision_signatures(result.events)


def test_resume_after_crash_reexecutes_with_same_idempotency_key(tools, counter):
    store = InMemoryEventStore()
    llm = FakeLLM(responses=[call("send_reply", ticket_id="TCK-3", body="fixed"), "Reply sent."])
    rt = AgentRuntime(llm, tools, store=store, approver=lambda rec, st: True)
    full = rt.run("reply to TCK-3", run_id="crash-run")
    # Simulate a crash right after approval, before the result was recorded.
    cut = next(i for i, e in enumerate(full.events) if isinstance(e, ToolCallApproved)) + 1
    crashed = InMemoryEventStore()
    for e in full.events[:cut]:
        crashed.append(e)
    rt2 = AgentRuntime(FakeLLM(responses=["Reply sent."]), tools, store=crashed)
    result = rt2.resume("crash-run")
    assert result.ok
    assert counter.keys == ["crash-run:1.0", "crash-run:1.0"]   # the tool can deduplicate by key


def test_adapt_foreign_tool():
    class ForeignResult:
        def __init__(self, ok: bool, content: Any) -> None:
            self.ok, self.content, self.error = ok, content, None

    class ForeignTool:
        name = "lookup_employee"
        spec = {"name": "lookup_employee", "description": "Find an employee.",
                "parameters": {"type": "object", "properties": {"email": {"type": "string"}}, "required": ["email"]}}
        side_effect = "read"

        def execute(self, arguments: dict[str, Any]) -> ForeignResult:
            return ForeignResult(True, {"name": "Dana Ortiz", "team": "IT"})

    tool = adapt_tool(ForeignTool())
    assert isinstance(tool.spec, ToolSpec) and tool.idempotent
    llm = FakeLLM(responses=[call("lookup_employee", email="dana@northwind.example"), "Dana Ortiz is in IT."])
    result = AgentRuntime(llm, [ForeignTool()]).run("Who is dana@northwind.example?")
    assert result.ok and "Dana Ortiz" in result.events_of(ToolResult)[0].content


def test_verifier_units():
    state = AgentState()

    class Out(BaseModel):
        cause: str
        severity: int

    assert json_schema(Out)('{"cause": "disk", "severity": 2}', state).passed
    assert not json_schema(Out)('{"cause": "disk"}', state).passed
    assert matches(r"^Root cause:", "a root-cause line")("Root cause: disk", state).passed
    combo = any_of(contains_all("disk"), contains_all("memory"))
    assert combo("memory pressure", state).passed
    both = all_of(contains_all("disk"), contains_all("memory"))
    verdict = both("disk only", state)
    assert not verdict.passed and "memory" in verdict.reason


def test_argument_validation_and_truncation_units():
    schema = {"type": "object", "properties": {"n": {"type": "integer", "minimum": 1}, "tags": {
        "type": "array", "items": {"type": "string"}}}, "required": ["n"], "additionalProperties": False}
    assert validate_arguments(schema, {"n": 2, "tags": ["a"]}) == []
    errs = validate_arguments(schema, {"n": True, "tags": [1], "extra": 1})
    assert any("integer" in e for e in errs) and any("unexpected" in e for e in errs) and any("tags[0]" in e for e in errs)
    assert "missing required" in validate_arguments(schema, {})[0]
    t = truncate_observation("a" * 100, 1000)
    assert not t.truncated and t.text == "a" * 100


def test_recorded_llm_serves_in_order(tools):
    original = _record(tools)
    rec = RecordedLLM.from_events(original.events)
    from aie_core.llm.types import CompletionRequest, Message
    first = rec.complete(CompletionRequest(messages=[Message.user("ignored")]))
    assert first.tool_calls[0].arguments == {"query": "failover replica"}


def test_default_citation_pattern_accepts_passage_ids():
    """Passage ids in the doc#section style (Chapters 10 and 13) pass the default pattern."""
    passage = FunctionTool(
        name="search_policies", description="Search HR policy passages.",
        parameters={"type": "object", "properties": {"query": {"type": "string"}},
                    "required": ["query"], "additionalProperties": False},
        fn=lambda query: "[hr-pto-policy#c3] Up to 10 days of PTO carry over.",
    )
    llm = FakeLLM(responses=[call("search_policies", query="carryover"),
                             "You can carry over 10 days [hr-pto-policy#c3]."])
    result = AgentRuntime(llm, [passage], dod=DefinitionOfDone(citations_grounded(1))).run("carryover?")
    assert result.ok, result.stop_reason
    # an invented passage id still fails
    assert CITATION.findall("see [hr-pto-policy#c9]") == ["hr-pto-policy#c9"]
