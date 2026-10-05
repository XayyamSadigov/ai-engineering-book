# path: book/projects/agentkit/tests/test_toolkit_integration.py
"""agentkit driving Chapter 16's governed ToolExecutor. Skipped when toolkit is not installed."""
from __future__ import annotations

import pytest
from pydantic import BaseModel

from aie_core.llm.providers import FakeLLM
from agentkit import AgentRuntime, ErrorClass, SideEffect, ToolResult, adapt_tool, executor_tools

from .conftest import call

toolkit = pytest.importorskip("toolkit")


class SearchArgs(BaseModel):
    query: str


class ReplyArgs(BaseModel):
    ticket_id: str
    body: str


def _executor():
    reg = toolkit.ToolRegistry()
    reg.register(toolkit.Tool(name="search_tickets", description="Search past tickets.", args_model=SearchArgs,
                              handler=lambda args, ctx: [{"id": "TCK-2026-0001", "subject": args.query}]))
    reg.register(toolkit.Tool(name="send_reply", description="Send a reply.", args_model=ReplyArgs,
                              handler=lambda args, ctx: {"sent": True}, side_effect=toolkit.SideEffect.EXTERNAL,
                              idempotent=False))
    ex = toolkit.ToolExecutor(reg, toolkit.PolicyEngine(), approvals=toolkit.ApprovalManager(),
                              idempotency=toolkit.InMemoryIdempotencyStore(), sleep=lambda s: None)
    ctx = toolkit.ToolContext(user_id="u-17", tenant="retail", session_id="s-1")
    return reg, ex, ctx


def test_executor_tools_run_through_toolkit_governance():
    reg, ex, ctx = _executor()
    tools = executor_tools(ex, ctx)
    assert {t.name for t in tools} == {"search_tickets", "send_reply"}
    assert next(t for t in tools if t.name == "send_reply").side_effect is SideEffect.EXTERNAL
    llm = FakeLLM(responses=[
        call("search_tickets", query="register declines cards"),
        call("send_reply", ticket_id="TCK-2026-0001", body="We are on it."),
        "Found TCK-2026-0001; the reply is awaiting approval.",
    ])
    result = AgentRuntime(llm, tools).run("Handle the card decline ticket")
    assert result.ok
    first, second = result.events_of(ToolResult)
    assert first.ok and "TCK-2026-0001" in first.content
    assert not second.ok and second.error_class is ErrorClass.PERMISSION      # toolkit owns the approval gate
    assert "pending_approval" in second.content


def test_adapt_toolkit_tool_spec_method():
    reg, _, _ = _executor()
    adapted = adapt_tool(type("Wrapped", (), {"name": "search_tickets", "spec": reg.get("search_tickets").spec,
                                              "side_effect": "reversible_write",
                                              "execute": lambda self, a: "ok"})())
    assert adapted.spec.name == "search_tickets" and adapted.side_effect is SideEffect.WRITE


def test_bound_tools_via_executor_bind_single_approval_gate():
    reg, ex, ctx = _executor()
    assert hasattr(ex, "bind")
    tools = executor_tools(ex, ctx)
    reply = next(t for t in tools if t.name == "send_reply")
    assert reply.side_effect is SideEffect.EXTERNAL and reply.approval_by_executor
    llm = FakeLLM(responses=[call("send_reply", ticket_id="TCK-2026-0001", body="On it."), "Awaiting approval."])
    result = AgentRuntime(llm, tools).run("reply")
    assert result.ok                                   # agentkit did not pause; toolkit gated the call once
    out = result.events_of(ToolResult)[0]
    assert not out.ok and out.error_class is ErrorClass.PERMISSION and "pending_approval" in out.content


def test_bound_tool_validation_error_maps_to_validation():
    _, ex, ctx = _executor()
    bound = {t.name: t for t in ex.bind(ctx)}
    tool = adapt_tool(bound["search_tickets"])
    from agentkit import ToolContext
    out = tool.execute({"query": 3}, ToolContext("r", 1, "c", "1.0", "r:1.0"))   # wrong type: toolkit validation
    assert not out.ok and out.error_class is ErrorClass.VALIDATION


class TicketArgs(BaseModel):
    subject: str


def _ticket_executor():
    created: list[str] = []
    reg = toolkit.ToolRegistry()
    reg.register(toolkit.Tool(name="create_ticket", description="Open a ticket.", args_model=TicketArgs,
                              handler=lambda args, ctx: created.append(args.subject) or {"id": f"T{len(created)}"},
                              side_effect=toolkit.SideEffect.REVERSIBLE_WRITE, idempotent=False))
    ex = toolkit.ToolExecutor(reg, toolkit.PolicyEngine(), idempotency=toolkit.InMemoryIdempotencyStore(),
                              sleep=lambda s: None)
    return ex, toolkit.ToolContext(user_id="u-17", tenant="retail", session_id="s-1"), created


def _two_runs(tools):
    for run_id in ("run-a", "run-b"):
        llm = FakeLLM(responses=[call("create_ticket", subject="Card reader down at store 12"), "Opened."])
        assert AgentRuntime(llm, tools).run("open a ticket", run_id=run_id).ok


def test_default_content_key_suppresses_duplicates_across_runs():
    ex, ctx, created = _ticket_executor()
    _two_runs(executor_tools(ex, ctx))                       # default idempotency="content"
    assert created == ["Card reader down at store 12"], "same action in a second run executes once"


def test_run_scoped_key_is_opt_in_and_repeats_across_runs():
    ex, ctx, created = _ticket_executor()
    _two_runs(executor_tools(ex, ctx, idempotency="run"))
    assert len(created) == 2, "run-scoped keys deduplicate only within one run"


def test_callable_key_policy_uses_a_business_key():
    ex, ctx, created = _ticket_executor()
    seen: list[str] = []

    def by_subject(name, arguments, actx):
        seen.append(actx.run_id)
        return f"{name}:{arguments['subject'].lower()}"

    _two_runs(executor_tools(ex, ctx, idempotency=by_subject))
    assert len(created) == 1 and seen == ["run-a", "run-b"]


def test_unknown_idempotency_mode_is_rejected():
    ex, ctx, _ = _ticket_executor()
    with pytest.raises(ValueError):
        executor_tools(ex, ctx, idempotency="session")
