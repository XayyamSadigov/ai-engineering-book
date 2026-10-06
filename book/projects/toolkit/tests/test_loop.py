# path: book/projects/toolkit/tests/test_loop.py
from __future__ import annotations

from aie_core.llm.providers import FakeLLM
from aie_core.llm.types import Message, Role, ToolCall

from toolkit import ToolLoop


def tc(i: str, name: str, **args) -> ToolCall:
    return ToolCall(id=i, name=name, arguments=args)


def test_loop_executes_calls_and_feeds_results_back(harness, ctx):
    llm = FakeLLM(responses=[[tc("1", "echo", text="ping")], "The tool said ping."])
    result = ToolLoop(llm, harness["ex"]).run([Message.user("echo ping")], ctx)
    assert result.stop_reason == "final" and result.final_text == "The tool said ping."
    second = llm.requests[1]
    roles = [m.role for m in second.messages]
    assert roles == [Role.USER, Role.ASSISTANT, Role.TOOL]
    assert second.messages[1].tool_calls[0].id == "1" and second.messages[2].tool_call_id == "1"
    assert {t.name for t in llm.requests[0].tools} == {"echo", "send"}


def test_pending_approval_forces_text_turn(harness, ctx):
    llm = FakeLLM(responses=[[tc("1", "send", to="a@northwind.example", body="hi")], "Waiting for approval."])
    result = ToolLoop(llm, harness["ex"]).run([Message.user("send it")], ctx)
    assert result.stop_reason == "pending_approval" and len(result.pending_approvals) == 1
    assert llm.requests[1].tool_choice == "none" and harness["send"].calls == 0


def test_pending_approval_in_last_round_still_gets_text_turn(harness, ctx):
    llm = FakeLLM(responses=[[tc("1", "send", to="a@northwind.example", body="hi")], "Waiting for approval."])
    result = ToolLoop(llm, harness["ex"], max_rounds=1).run([Message.user("send it")], ctx)
    assert result.stop_reason == "pending_approval" and result.final_text == "Waiting for approval."
    assert len(llm.requests) == 2 and llm.requests[1].tool_choice == "none"


def test_tools_not_offered_are_never_executed(harness, ctx):
    reader = ctx.model_copy(update={"scopes": frozenset({"tickets:read"})})
    llm = FakeLLM(responses=[[tc("1", "send", to="a@northwind.example", body="hi")], "Cannot send."])
    result = ToolLoop(llm, harness["ex"]).run([Message.user("send")], reader)
    assert [t.name for t in llm.requests[0].tools] == ["echo"]
    assert result.tool_results[0].error["code"] == "tool_not_available"
    assert harness["approvals"].pending() == []


def test_task_filter_and_max_rounds(harness, ctx):
    llm = FakeLLM(handler=lambda req: [tc(f"c{len(req.messages)}", "echo", text=f"t{len(req.messages)}")])
    result = ToolLoop(llm, harness["ex"], max_rounds=3).run([Message.user("loop")], ctx, tool_names=["echo"])
    assert result.stop_reason == "max_rounds" and result.rounds == 3 and len(result.tool_results) == 3
    assert [t.name for t in llm.requests[0].tools] == ["echo"]


def test_repeated_identical_call_stops(harness, ctx):
    llm = FakeLLM(handler=lambda req: [tc(f"c{len(req.messages)}", "echo", text="same")])
    result = ToolLoop(llm, harness["ex"], max_rounds=10, max_identical_calls=2).run([Message.user("x")], ctx)
    assert result.stop_reason == "repeated_call" and harness["echo"].calls == 2
