"""Crash recovery, approval binding, and policy edge cases."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aie_core.llm.providers import FakeLLM
from aie_core.llm.types import Role, ToolCall
from agentkit import (
    AgentRuntime, Budget, DefaultPolicy, DefinitionOfDone, FunctionTool, InMemoryEventStore, JsonlEventStore,
    PolicyDecision, SideEffect, TerminationReason, ToolOutput, replay, tool_was_called,
)
from agentkit.errors import ErrorClass
from agentkit.events import ToolCallRequested, ToolResult

from .conftest import call


def _params(**props: Any) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


@pytest.fixture
def log() -> list[tuple[str, str]]:
    return []


@pytest.fixture
def kit(log: list[tuple[str, str]]) -> list[FunctionTool]:
    def search(query: str) -> str:
        log.append(("search", query))
        return f"[doc-{query}] text"

    def boom(x: str) -> str:
        raise KeyError("bug")

    def send(to: str) -> str:
        log.append(("send", to))
        return "sent"

    return [FunctionTool("search", "s", _params(query={"type": "string"}), search),
            FunctionTool("boom", "b", _params(x={"type": "string"}), boom),
            FunctionTool("send", "x", _params(to={"type": "string"}), send,
                         side_effect=SideEffect.EXTERNAL, idempotent=False)]


def _truncated(events: list, upto: int) -> InMemoryEventStore:
    store = InMemoryEventStore()
    for e in events[:upto]:
        store.append(e)
    return store


def test_crash_mid_batch_records_the_rest_of_the_batch(kit: list[FunctionTool]) -> None:
    batch = [ToolCall(id="a", name="search", arguments={"query": "x"}),
             ToolCall(id="b", name="search", arguments={"query": "y"})]
    full = AgentRuntime(FakeLLM(responses=[batch, "done"]), kit).run("g", run_id="mid")
    cut = next(i for i, e in enumerate(full.events) if isinstance(e, ToolCallRequested)) + 1
    seen: list = []
    llm = FakeLLM(responses=["done"])
    original = llm.complete
    llm.complete = lambda req: (seen.append(req), original(req))[1]  # type: ignore[method-assign]
    result = AgentRuntime(llm, kit, store=_truncated(full.events, cut)).resume("mid")
    assert result.ok
    answered = {m.tool_call_id for m in seen[0].messages if m.role is Role.TOOL}
    assert answered == {"a", "b"}


def test_crash_before_final_answer_is_judged_still_verifies(kit: list[FunctionTool]) -> None:
    full = AgentRuntime(FakeLLM(responses=["the answer"]), kit).run("g", run_id="fin")
    store = _truncated(full.events, 2)            # GoalSet, ModelDecision(final)
    result = AgentRuntime(FakeLLM(responses=[]), kit, store=store).resume("fin")
    assert result.ok and result.final_answer == "the answer"


def test_crash_after_fatal_result_still_stops_fatal(kit: list[FunctionTool]) -> None:
    full = AgentRuntime(FakeLLM(responses=[call("boom", x="1"), "done"]), kit).run("g", run_id="fatal")
    assert full.stop_reason is TerminationReason.FATAL_ERROR
    cut = next(i for i, e in enumerate(full.events) if isinstance(e, ToolResult)) + 1
    result = AgentRuntime(FakeLLM(responses=["done"]), kit, store=_truncated(full.events, cut)).resume("fatal")
    assert result.stop_reason is TerminationReason.FATAL_ERROR


def test_stale_approval_is_refused(kit: list[FunctionTool], log: list[tuple[str, str]]) -> None:
    rt = AgentRuntime(FakeLLM(responses=[call("send", to="alice@corp"), call("send", i=1, to="mallory@evil"),
                                         "done"]), kit)
    first = rt.run("g", run_id="appr")
    shown = first.state.pending_approval.request_id
    second = rt.resume("appr", approve=True, request_id=shown)
    assert second.state.pending_approval.arguments == {"to": "mallory@evil"}
    with pytest.raises(ValueError, match="awaiting approval"):
        rt.resume("appr", approve=True, request_id=shown)   # a late click on the first request
    assert log == [("send", "alice@corp")]


def test_replay_uses_the_extended_budget(kit: list[FunctionTool]) -> None:
    store = InMemoryEventStore()
    rt = AgentRuntime(FakeLLM(responses=[call("search", query="a"), call("search", query="b"), "done"]), kit,
                      store=store, budget=Budget(max_steps=1))
    assert rt.run("g", run_id="ext").stop_reason is TerminationReason.MAX_STEPS
    assert rt.resume("ext", budget=Budget(max_steps=5)).ok
    report = replay(store.load("ext"))
    assert report.first_divergence is None and report.result.ok


def test_torn_last_line_is_dropped_and_the_run_resumes(kit: list[FunctionTool], tmp_path: Path) -> None:
    store = JsonlEventStore(tmp_path)
    AgentRuntime(FakeLLM(responses=[call("search", query="a"), "ok"]), kit, store=store,
                 budget=Budget(max_steps=1)).run("g", run_id="torn")
    path = tmp_path / "torn.jsonl"
    path.write_text(path.read_text() + '{"run_id": "torn", "seq": 99, "ty')
    fresh = JsonlEventStore(tmp_path)
    result = AgentRuntime(FakeLLM(responses=["ok"]), kit, store=fresh).resume("torn", budget=Budget(max_steps=5))
    assert result.ok
    assert len(JsonlEventStore(tmp_path).load("torn")) == len(result.events)


def test_corruption_mid_file_fails_loudly(kit: list[FunctionTool], tmp_path: Path) -> None:
    store = JsonlEventStore(tmp_path)
    AgentRuntime(FakeLLM(responses=["ok"]), kit, store=store).run("g", run_id="bad")
    path = tmp_path / "bad.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join([lines[0], "{not json", *lines[1:]]) + "\n")
    with pytest.raises(ValueError):
        JsonlEventStore(tmp_path).load("bad")


def test_a_failed_call_does_not_satisfy_tool_was_called() -> None:
    def create_ticket(summary: str) -> ToolOutput:
        return ToolOutput.failure("ticket service down", ErrorClass.IMPOSSIBLE)

    tool = FunctionTool("create_ticket", "c", _params(summary={"type": "string"}), create_ticket)
    result = AgentRuntime(FakeLLM(responses=[call("create_ticket", summary="s")] + ["Ticket opened."] * 4), [tool],
                          dod=DefinitionOfDone(tool_was_called("create_ticket"))).run("g")
    assert not result.ok and result.final_answer is None   # never accepted as done


def test_an_allowing_rule_cannot_skip_approval(kit: list[FunctionTool], log: list[tuple[str, str]]) -> None:
    def tenant(tool: Any, args: dict[str, Any], principal: dict[str, Any]) -> PolicyDecision | None:
        if tool.name == "send":
            return PolicyDecision(allowed=args["to"].endswith("@corp"), reason="tenant scope")
        return None

    result = AgentRuntime(FakeLLM(responses=[call("send", to="bob@corp"), "done"]), kit,
                          policy=DefaultPolicy(rules=[tenant])).run("g")
    assert result.stop_reason is TerminationReason.APPROVAL_REQUIRED and log == []


def test_crash_after_accepted_answer_completes_without_a_new_model_call(kit: list[FunctionTool]) -> None:
    full = AgentRuntime(FakeLLM(responses=["first answer"]), kit).run("g", run_id="acc")
    cut = next(i for i, e in enumerate(full.events) if type(e).__name__ == "FinalAnswer") + 1
    result = AgentRuntime(FakeLLM(responses=["second answer"]), kit,
                          store=_truncated(full.events, cut)).resume("acc")
    assert result.ok and result.final_answer == "first answer"


@pytest.mark.parametrize("tail", [b'{"run_id": "t", "seq": 99, "text": "\xd0', b""])
def test_torn_utf8_tail_and_missing_newline(kit: list[FunctionTool], tmp_path: Path, tail: bytes) -> None:
    AgentRuntime(FakeLLM(responses=["ok"]), kit, store=JsonlEventStore(tmp_path),
                 budget=Budget(max_steps=1)).run("g", run_id="t")
    path = tmp_path / "t.jsonl"
    data = path.read_bytes()
    path.write_bytes((data + tail) if tail else data.rstrip(b"\n"))   # torn mid-character, or newline lost
    events = JsonlEventStore(tmp_path).load("t")
    assert path.read_bytes() == data and len(events) == data.count(b"\n")
