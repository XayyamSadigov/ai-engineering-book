# path: book/projects/agentkit/tests/test_runtime.py
"""Loop behavior: completion, termination conditions, policy, approval, errors, budgets, tracing."""
from __future__ import annotations

import itertools

import pytest

from aie_core.llm.errors import ProviderUnavailableError
from aie_core.llm.gateway import PricingTable
from aie_core.llm.providers import FakeLLM
from aie_core.llm.types import Role
from aie_core.observability import InMemoryTracer
from agentkit import (
    AgentRuntime, AgentStatus, Budget, DefaultPolicy, DefinitionOfDone, ErrorClass, FinalAnswer, FunctionTool,
    InMemoryEventStore, LoopConfig, ModelDecision, Note, Stopped, TerminationReason, ToolCallApproved,
    ToolCallDenied, ToolResult, TransientToolError, citations_grounded, derive_state, tool_was_called,
)

from .conftest import call

ANSWER = "Clear the cached VPN profile and re-enroll the certificate [it-vpn-access-runbook]."


def test_completes_task_and_state_is_derivable(tools, counter):
    llm = FakeLLM(responses=[call("search_runbooks", query="vpn login loops"), ANSWER])
    rt = AgentRuntime(llm, tools, dod=DefinitionOfDone(citations_grounded(1)))
    result = rt.run("A user's VPN login keeps looping. What should support do?")

    assert result.ok and result.status is AgentStatus.COMPLETED
    assert result.final_answer == ANSWER
    assert result.trajectory() == ["search_runbooks"]
    assert counter.count("search_runbooks") == 1
    kinds = [e.type for e in result.events]
    assert kinds[0] == "goal_set" and kinds[-1] == "stopped"
    assert "final_answer" in kinds
    # state rebuilt from the log equals the live projection
    assert derive_state(result.events).model_dump() == result.state.model_dump()
    # the second model call saw the observation as a tool message
    second = llm.requests[1]
    assert second.messages[-1].role is Role.TOOL and "it-vpn-access-runbook" in second.messages[-1].text
    assert {t.name for t in second.tools} == {"search_runbooks", "send_reply"}


def test_stops_on_max_steps(tools):
    queries = (f"vpn attempt {i}" for i in itertools.count())
    llm = FakeLLM(handler=lambda req: call("search_runbooks", query=next(queries)))
    result = AgentRuntime(llm, tools, budget=Budget(max_steps=3),
                          config=LoopConfig(max_identical_calls=5, max_no_progress_steps=10)).run("loop forever")
    assert result.stop_reason is TerminationReason.MAX_STEPS
    assert result.state.usage.steps == 3
    assert len(llm.requests) == 3
    assert result.stop_reason.is_budget


def test_detects_repeated_identical_action(tools, counter):
    llm = FakeLLM(handler=lambda req: call("search_runbooks", query="vpn"))
    result = AgentRuntime(llm, tools, budget=Budget(max_steps=10)).run("find the vpn runbook")
    assert result.stop_reason is TerminationReason.REPEATED_ACTION
    assert counter.count("search_runbooks") == 2              # max_identical_calls default
    second = result.events_of(ToolResult)[1]
    assert "already made" in second.notice
    assert result.events_of(ToolCallDenied)[-1].error_class is ErrorClass.VALIDATION


def test_detects_no_progress(tools):
    words = (f"zzz{i}" for i in itertools.count())
    llm = FakeLLM(handler=lambda req: call("search_runbooks", query=next(words)))
    result = AgentRuntime(llm, tools, budget=Budget(max_steps=20)).run("search for nonsense")
    assert result.stop_reason is TerminationReason.NO_PROGRESS
    # the first "no results" is new information; the next three repeat it
    assert result.state.usage.steps == 4


def test_denies_unauthorized_tool_and_hides_it(tools, counter):
    llm = FakeLLM(responses=[
        call("send_reply", ticket_id="TCK-1", body="done"),
        call("delete_database", cluster="pg-retail-prod"),
        call("search_runbooks", query="password reset"),
        "Verify identity with the manager first [it-password-reset-runbook].",
    ])
    policy = DefaultPolicy(allowed={"search_runbooks"})
    result = AgentRuntime(llm, tools, policy=policy).run("Reset a password for a store manager")

    assert result.ok
    assert counter.count("send_reply") == 0
    denied = result.events_of(ToolCallDenied)
    assert [d.tool for d in denied] == ["send_reply", "delete_database"]
    assert denied[0].error_class is ErrorClass.VALIDATION   # not visible, so it is unknown to this run
    assert {t.name for t in llm.requests[0].tools} == {"search_runbooks"}
    assert "DENIED" in llm.requests[1].messages[-1].text


def test_policy_rule_denies_by_argument(tools, counter):
    from agentkit import PolicyDecision

    def tenant_scope(tool, args, principal):
        if tool.name == "send_reply" and not args["ticket_id"].startswith(principal["tenant"]):
            return PolicyDecision(allowed=False, reason="ticket belongs to another tenant")
        return None

    llm = FakeLLM(responses=[call("send_reply", ticket_id="logistics-7", body="hi"), "Cannot reply to that ticket."])
    rt = AgentRuntime(llm, tools, policy=DefaultPolicy(rules=[tenant_scope]), principal={"tenant": "retail"})
    result = rt.run("reply to logistics-7")
    assert result.ok and counter.count("send_reply") == 0
    assert result.events_of(ToolCallDenied)[0].error_class is ErrorClass.PERMISSION


def test_approval_pause_then_resume_executes_once(tools, counter):
    store = InMemoryEventStore()
    llm = FakeLLM(responses=[call("send_reply", ticket_id="TCK-9", body="Your VPN is fixed."), "Reply sent to TCK-9."])
    rt = AgentRuntime(llm, tools, store=store)
    paused = rt.run("Tell the requester of TCK-9 the VPN is fixed", run_id="run-approval")

    assert paused.stop_reason is TerminationReason.APPROVAL_REQUIRED
    assert paused.state.pending_approval is not None and counter.count("send_reply") == 0

    resumed = rt.resume("run-approval", approve=True, reason="checked by on-call")
    assert resumed.ok
    assert counter.count("send_reply") == 1
    assert counter.keys == ["run-approval:1.0"]
    approvals = resumed.events_of(ToolCallApproved)
    assert approvals[-1].by == "human"
    # one continuous log, no gaps
    assert [e.seq for e in store.load("run-approval")] == list(range(len(store.load("run-approval"))))


def test_approval_rejected_goes_back_to_model(tools, counter):
    llm = FakeLLM(responses=[call("send_reply", ticket_id="TCK-9", body="x"), "I did not send the reply."])
    rt = AgentRuntime(llm, tools)
    rt.run("reply", run_id="run-reject")
    result = rt.resume("run-reject", approve=False, reason="wrong wording")
    assert result.ok and counter.count("send_reply") == 0
    assert "wrong wording" in llm.requests[-1].messages[-1].text


def test_inline_approver(tools, counter):
    llm = FakeLLM(responses=[call("send_reply", ticket_id="TCK-1", body="ok"), "sent"])
    result = AgentRuntime(llm, tools, approver=lambda rec, state: rec.arguments["ticket_id"] == "TCK-1").run("reply")
    assert result.ok and counter.count("send_reply") == 1


def test_verifier_rejects_premature_final_answer_and_loop_continues(tools):
    llm = FakeLLM(responses=[
        "Just restart the laptop.",                                     # premature: no evidence, no citation
        call("search_runbooks", query="vpn login"),
        ANSWER,
    ])
    dod = DefinitionOfDone(tool_was_called("search_runbooks"), citations_grounded(1))
    result = AgentRuntime(llm, tools, dod=dod).run("VPN login loops")

    assert result.ok and result.final_answer == ANSWER
    rejected = [n for n in result.events_of(Note) if n.kind == "dod_rejected"]
    assert len(rejected) == 1 and rejected[0].error_class is ErrorClass.SEMANTIC
    assert result.state.dod_rejections == 1
    # the model was told why
    assert "search_runbooks" in llm.requests[1].messages[-1].text
    final = result.events_of(FinalAnswer)[0]
    assert all(c["passed"] for c in final.checks)


def test_verification_failed_after_repeated_rejections(tools):
    llm = FakeLLM(handler=lambda req: "trust me")
    dod = DefinitionOfDone(citations_grounded(1))
    result = AgentRuntime(llm, tools, dod=dod, budget=Budget(max_steps=10),
                          config=LoopConfig(max_no_progress_steps=10)).run("answer")
    assert result.stop_reason is TerminationReason.VERIFICATION_FAILED
    assert result.state.dod_rejections == 3


def test_ungrounded_citation_is_rejected(tools):
    llm = FakeLLM(responses=[
        call("search_runbooks", query="vpn"),
        "See [it-made-up-runbook].",
        ANSWER,
    ])
    result = AgentRuntime(llm, tools, dod=DefinitionOfDone(citations_grounded(1))).run("vpn")
    notes = [n for n in result.events_of(Note) if n.kind == "dod_rejected"]
    assert result.ok and "it-made-up-runbook" in notes[0].text


def test_budget_exceeded_cost_tokens_tool_calls_deadline(tools):
    pricing = PricingTable({"fake-model": {"input_per_1m": 1_000_000.0, "output_per_1m": 0.0}})  # illustrative
    llm = FakeLLM(handler=lambda req: call("search_runbooks", query=f"vpn {len(req.messages)}"))
    r1 = AgentRuntime(llm, tools, budget=Budget(max_steps=50, max_cost_usd=100.0), pricing=pricing).run("x")
    assert r1.stop_reason is TerminationReason.MAX_COST and r1.state.usage.cost_usd >= 100.0

    r2 = AgentRuntime(FakeLLM(responses=["never called"]), tools, budget=Budget(max_tokens=50)).run("x")
    assert r2.stop_reason is TerminationReason.MAX_TOKENS and r2.state.usage.steps == 0   # pre-flight refusal

    r3 = AgentRuntime(llm, tools, budget=Budget(max_steps=50, max_tool_calls=2)).run("x")
    assert r3.stop_reason is TerminationReason.MAX_TOOL_CALLS and r3.state.usage.tool_calls == 2

    ticks = itertools.count(0, 10)                      # each clock read advances 10 s
    r4 = AgentRuntime(llm, tools, budget=Budget(max_steps=50, deadline_s=45), clock=lambda: float(next(ticks))).run("x")
    assert r4.stop_reason is TerminationReason.DEADLINE


def test_transient_error_retried_validation_repaired_fatal_stops(counter):
    attempts = {"n": 0}

    def flaky(service: str) -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise TransientToolError("status backend timed out")
        return f"{service}: degraded"

    def broken(service: str) -> str:
        return {"a": 1}["missing"]  # a bug: KeyError

    schema = {"type": "object", "properties": {"service": {"type": "string", "enum": ["pos", "tracking"]}},
              "required": ["service"], "additionalProperties": False}
    tools = [FunctionTool("get_service_status", "Status of a service.", schema, flaky),
             FunctionTool("get_metrics", "Metrics.", schema, broken)]
    llm = FakeLLM(responses=[
        call("get_service_status", service="checkout"),     # invalid enum: validation, model repairs
        call("get_service_status", service="pos"),          # transient once, retried, succeeds
        call("get_metrics", service="pos"),                 # fatal bug: run stops
    ])
    result = AgentRuntime(llm, tools).run("Is POS healthy?")
    denied = result.events_of(ToolCallDenied)
    assert denied[0].error_class is ErrorClass.VALIDATION and "must be one of" in denied[0].reason
    ok = result.events_of(ToolResult)[0]
    assert ok.ok and ok.attempts == 2
    assert result.stop_reason is TerminationReason.FATAL_ERROR
    assert result.events_of(ToolResult)[-1].error_class is ErrorClass.FATAL


def test_permission_error_from_tool_is_not_retried():
    calls = {"n": 0}

    def guarded(path: str) -> str:
        calls["n"] += 1
        raise PermissionError("not allowed to read HR records")

    schema = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
    llm = FakeLLM(responses=[call("read", path="/hr/salaries"), "I cannot access HR records."])
    result = AgentRuntime(llm, [FunctionTool("read", "Read a file.", schema, guarded)]).run("salaries?")
    assert result.ok and calls["n"] == 1
    assert result.events_of(ToolResult)[0].error_class is ErrorClass.PERMISSION


def test_observation_truncation_is_recorded():
    big = "x" * 10_000 + "ROOT CAUSE: disk full"
    tool = FunctionTool("tail_log", "Tail a log.", {"type": "object", "properties": {}}, lambda: big)
    llm = FakeLLM(responses=[call("tail_log"), "disk full"])
    result = AgentRuntime(llm, [tool], config=LoopConfig(max_observation_chars=500)).run("why down?")
    obs = result.events_of(ToolResult)[0]
    assert obs.truncated and obs.original_chars == len(big) and len(obs.content) <= 500
    assert obs.content.endswith("disk full")        # head-and-tail keeps the end of a log


def test_model_error_stops_cleanly(tools):
    llm = FakeLLM(responses=[ProviderUnavailableError("upstream 503")])
    result = AgentRuntime(llm, tools).run("x")
    assert result.stop_reason is TerminationReason.MODEL_ERROR and "transient" in result.detail


def test_tracing_spans_per_step(tools):
    tracer = InMemoryTracer()
    llm = FakeLLM(responses=[call("search_runbooks", query="vpn"), ANSWER])
    AgentRuntime(llm, tools, tracer=tracer).run("vpn")
    assert len(tracer.find("agent.run")) == 1
    steps = tracer.find("agent.step")
    assert [s.attributes["decision"] for s in steps] == ["tool_calls", "final"]
    assert tracer.find("agent.tool")[0].attributes["tool"] == "search_runbooks"
    assert tracer.find("agent.run")[0].attributes["stop_reason"] == "completed"


def test_events_are_immutable(tools):
    result = AgentRuntime(FakeLLM(responses=["hi"]), tools).run("hello")
    with pytest.raises(Exception):
        result.events[0].goal = "changed"  # type: ignore[misc]
    assert isinstance(result.events_of(ModelDecision)[0], ModelDecision)
    assert isinstance(result.events[-1], Stopped)
