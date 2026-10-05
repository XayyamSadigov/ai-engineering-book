# path: book/projects/examples/ch38/test_ch38_interrupts.py
"""Approvals with expiry and escalation, timers, and webhooks, all without a waiting process."""
from __future__ import annotations

from typing import Any

import pytest

from aie_core.llm.providers import FakeLLM
from aie_core.llm.types import ToolCall
from agentkit import AgentRuntime, AgentStatus, FunctionTool, SideEffect, ToolCallDenied, ToolResult

from durable import Database, DurableRunner
from fakes import FakeClock
from interrupts import EscalationPolicy, InterruptKind, InterruptManager, InterruptStatus


def tc(name: str, i: int = 0, **args: Any) -> list[ToolCall]:
    return [ToolCall(id=f"{name}-{i}", name=name, arguments=args)]


class World:
    def __init__(self, responses: list[Any]) -> None:
        self.clock = FakeClock()
        self.db = Database()
        self.sent: list[dict[str, Any]] = []
        self.notified: list[tuple[str, str | None]] = []
        llm = FakeLLM(responses=responses)

        def send_reply(ticket_id: str, body: str) -> str:
            self.sent.append({"ticket_id": ticket_id, "body": body})
            return f"reply sent on {ticket_id}"

        send = FunctionTool("send_reply", "Send a reply to the customer.",
                            {"type": "object", "properties": {"ticket_id": {"type": "string"},
                                                              "body": {"type": "string"}},
                             "required": ["ticket_id", "body"], "additionalProperties": False},
                            fn=send_reply, side_effect=SideEffect.EXTERNAL, idempotent=False)
        holder: dict[str, InterruptManager] = {}
        self.runner = DurableRunner(self.db, lambda store: AgentRuntime(
            llm, [send, *holder["m"].wait_tools()], store=store, clock=self.clock), owner="w1", clock=self.clock)
        self.mgr = InterruptManager(
            self.db, self.runner, clock=self.clock,
            policies={"send_reply": EscalationPolicy(chain=["support-lead", "duty-manager"],
                                                     escalate_after_s=3600, expire_after_s=7200)},
            notify=lambda it, what: self.notified.append((what, it.assignee)))
        holder["m"] = self.mgr

    def start(self, goal: str, run_id: str = "r1"):
        result = self.runner.start(goal, run_id=run_id)
        return result, self.mgr.after_run(result)


REPLY = tc("send_reply", ticket_id="TCK-9", body="Your refund was approved.")


def test_approval_pause_holds_no_process_and_resumes_on_decision():
    w = World([REPLY, "Reply sent to TCK-9."])
    result, it = w.start("Reply to TCK-9 that the refund is approved.")
    assert result.status is AgentStatus.AWAITING_APPROVAL and it.kind is InterruptKind.APPROVAL
    assert w.runner.leases.holder("r1") is None            # nothing is running while we wait
    assert w.mgr.pending("support-lead")[0].id == it.id and w.sent == []
    final = w.mgr.decide(it.id, approve=True, by="support-lead", reason="checked order")
    assert final.ok and len(w.sent) == 1
    with pytest.raises(ValueError):                        # a second click on the same request
        w.mgr.decide(it.id, approve=True, by="support-lead")


def test_escalation_moves_the_task_and_limits_who_may_decide():
    w = World([REPLY, "Sent."])
    _, it = w.start("Reply to TCK-9.")
    w.clock.advance(3601)
    w.mgr.tick()
    escalated = w.mgr.get(it.id)
    assert escalated.level == 1 and escalated.assignee == "duty-manager"
    assert ("escalated", "duty-manager") in w.notified
    with pytest.raises(PermissionError):
        w.mgr.decide(it.id, approve=True, by="random-agent-user")
    assert w.mgr.decide(it.id, approve=True, by="duty-manager").ok


def test_unanswered_approval_expires_as_a_denial():
    w = World([REPLY, "I could not send the reply: the approval expired. A human should follow up on TCK-9."])
    _, it = w.start("Reply to TCK-9.")
    w.clock.advance(7201)
    [result] = w.mgr.tick()
    assert w.mgr.get(it.id).status is InterruptStatus.EXPIRED
    denied = result.events_of(ToolCallDenied)
    assert denied and "expired" in denied[0].reason
    assert result.ok and w.sent == []                      # silence never becomes consent


def test_timer_wait_survives_without_a_process():
    w = World([tc("wait_until", seconds=3600, reason="carrier SLA"), "Waited one hour; escalating to the carrier."])
    result, it = w.start("Check back on PKG-77 in an hour.")
    assert it.kind is InterruptKind.TIMER and w.runner.store.status("r1") == "awaiting_approval"
    w.clock.advance(100)
    assert w.mgr.tick() == []
    w.clock.advance(3500)
    [done] = w.mgr.tick()
    assert done.ok
    assert "timer fired" in done.events_of(ToolResult)[0].content
    assert done.state.usage.elapsed_s < 60                 # the hour of waiting did not count as run time


def test_webhook_is_deduplicated_and_resumes_once():
    w = World([tc("wait_for_event", event_type="delivery_scan", correlation_id="PKG-77", timeout_s=86400),
               "PKG-77 was delivered at the Baku hub."])
    _, it = w.start("Tell me when PKG-77 is scanned at the hub.")
    assert it.kind is InterruptKind.EVENT
    payload = {"hub": "Baku", "status": "delivered"}
    first = w.mgr.deliver(event_id="evt-1", event_type="delivery_scan", correlation_id="PKG-77", payload=payload)
    again = w.mgr.deliver(event_id="evt-1", event_type="delivery_scan", correlation_id="PKG-77", payload=payload)
    assert first is not None and first.ok and again is None
    assert "Baku" in first.events_of(ToolResult)[0].content
    w.clock.advance(90000)
    assert w.mgr.tick() == []                              # the timeout timer was cancelled


def test_event_that_arrives_before_the_wait_is_not_lost():
    w = World([tc("wait_for_event", event_type="vendor_reply", correlation_id="PO-5", timeout_s=3600), "Vendor said yes."])
    assert w.mgr.deliver(event_id="evt-9", event_type="vendor_reply", correlation_id="PO-5",
                         payload={"answer": "yes"}) is None
    result, it = w.start("Wait for the vendor's answer on PO-5.")
    assert it.status is InterruptStatus.RESOLVED
    assert w.runner.store.status("r1") == "completed"


def test_event_wait_times_out():
    w = World([tc("wait_for_event", event_type="vendor_reply", correlation_id="PO-6", timeout_s=600),
               "No vendor reply within 10 minutes; escalated to purchasing."])
    w.start("Wait for the vendor reply on PO-6.")
    w.clock.advance(601)
    [result] = w.mgr.tick()
    assert "timed out" in result.events_of(ToolResult)[0].content and result.ok
