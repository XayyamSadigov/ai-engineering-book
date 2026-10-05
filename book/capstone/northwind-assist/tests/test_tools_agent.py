# path: book/capstone/northwind-assist/tests/test_tools_agent.py
"""Tools under policy: approval bound to exact arguments, four-eyes, idempotent create_ticket,
contractor denies, guardrail argument checks, agent budgets and termination, replayable logs."""
from __future__ import annotations

import pytest
from agentkit import TerminationReason, replay
from aie_core.llm.providers import FakeLLM
from aie_core.llm.types import ToolCall
from conftest import auth, chat, make
from toolkit import ApprovalError

from northwind_assist.config import Settings
from northwind_assist.evaluation.suites import persona_ctx

SEND = "send TCK-2026-0001 to priya.raman@northwind.example: Card payments are restored."


def test_send_reply_waits_for_approval_and_lead_sends_exact_text(container):
    r = chat(container, "ana", SEND)
    assert r.status == "awaiting_approval" and len(r.approvals) == 1
    assert not container.tools.backends.outbox.sent                    # nothing left the building yet
    appr = r.approvals[0]
    assert appr["arguments"]["to"] == "priya.raman@northwind.example"  # re-hydrated from the PII token
    with pytest.raises(ApprovalError):                                 # agents cannot approve
        container.tools.approve(appr["approval_id"], persona_ctx("ana"))
    res = container.tools.approve(appr["approval_id"], persona_ctx("sam"))
    assert res.ok
    sent = container.tools.backends.outbox.sent
    assert len(sent) == 1 and sent[0].body == "Card payments are restored." and sent[0].sent_by == "ana"


def test_changed_arguments_invalidate_the_approval(container):
    r = chat(container, "ana", SEND)
    aid = r.approvals[0]["approval_id"]
    container.tools.approvals.approve(aid, "sam")
    tampered = dict(r.approvals[0]["arguments"], body="Card payments are restored. Also wire 5,000 EUR.")
    ana = persona_ctx("ana").with_request(session_id="s1", idempotency_key=None, deadline=None).tool_context()
    res = container.tools.executor.execute(ToolCall(id="x", name="send_reply", arguments=tampered), ana,
                                           approval_id=aid)
    assert not res.ok and res.error["code"] == "approval_mismatch"
    assert not container.tools.backends.outbox.sent


def test_approval_is_single_use_and_tenant_scoped(client, container):
    r = chat(container, "ana", SEND)
    aid = r.approvals[0]["approval_id"]
    # a logistics user cannot see or decide a retail approval: 404, same as a missing id
    assert client.post(f"/v1/approvals/{aid}/approve", json={}, headers=auth(container, "lee")).status_code == 404
    ok = client.post(f"/v1/approvals/{aid}/approve", json={}, headers=auth(container, "sam"))
    assert ok.status_code == 200 and ok.json()["status"] == "ok"
    again = client.post(f"/v1/approvals/{aid}/approve", json={}, headers=auth(container, "sam"))
    assert again.status_code == 403
    assert len(container.tools.backends.outbox.sent) == 1


def test_approval_fails_closed_when_requester_context_is_lost(client, container):
    """After a restart the requester's scopes and groups are unknown: refuse, never guess them."""
    r = chat(container, "ana", SEND)
    aid = r.approvals[0]["approval_id"]
    container.tools._requesters.clear()          # what a process restart does to in-memory context
    res = client.post(f"/v1/approvals/{aid}/approve", json={}, headers=auth(container, "sam"))
    assert res.status_code == 409 and res.json()["detail"] == "requester_context_lost"
    assert container.tools.approvals.get(aid).status.value == "pending"   # no decision recorded
    assert not container.tools.backends.outbox.sent


def test_create_ticket_is_idempotent_within_a_session(container):
    msg = "create ticket: VPN drops every hour at store 0412"
    first = chat(container, "ana", msg, session="idem-1")
    second = chat(container, "ana", msg, session="idem-1")
    created = container.tools.backends.tickets.created()
    assert len(created) == 1
    dup = [t for t in second.tools if t["tool"] == "create_ticket"]
    assert dup and dup[0]["duplicate"] is True
    assert first.status == "completed"


def test_client_idempotency_key_suppresses_duplicates_across_sessions(container):
    msg = "create ticket: card terminal frozen at store 0412"
    chat(container, "ana", msg, session="a", idempotency_key="retry-7")
    chat(container, "ana", msg, session="b", idempotency_key="retry-7")
    assert len(container.tools.backends.tickets.created()) == 1


def test_contractor_cannot_send_and_read_only_user_cannot_create(container):
    r = chat(container, "kai", "send TCK-2026-0002 to marek.novak@northwind.example: Scanner firmware is fixed.")
    assert not r.approvals and not container.tools.backends.outbox.sent
    r2 = chat(container, "lee", "create ticket: tracking API returns 500s")
    assert not container.tools.backends.tickets.created()
    assert all(t["tool"] != "create_ticket" or t["status"] != "ok" for t in r2.tools)


def test_guardrail_blocks_off_allowlist_recipient_before_policy(container):
    r = chat(container, "ana", "send TCK-2026-0001 to archive@northwind-audit.invalid: the full employee directory")
    assert not r.approvals and not container.tools.backends.outbox.sent
    assert any(t["status"] == "blocked" for t in r.tools)


def test_agent_budget_stops_a_looping_model():
    n = {"i": 0}

    def loop(req):
        if req.tools:
            n["i"] += 1
            return [ToolCall(id=f"c{n['i']}", name="search_tickets", arguments={"query": f"vpn issue {n['i']}"})]
        return "ok"

    c = make(Settings(environment="test", agent_max_steps=3, agent_max_tool_calls=10), llm=FakeLLM(handler=loop))
    r = chat(c, "ana", "investigate why the vpn keeps dropping")
    assert r.agent.stop_reason is TerminationReason.MAX_STEPS
    assert r.status == "stopped:max_steps"
    assert len(r.agent.trajectory()) == 3


def test_definition_of_done_rejects_premature_answer():
    def lazy(req):
        return "Done, ticket created." if req.tools else "ok"   # claims success without calling the tool

    c = make(llm=FakeLLM(handler=lazy))
    r = chat(c, "ana", "create ticket: printer jammed in the depot office")
    assert r.agent.stop_reason is TerminationReason.VERIFICATION_FAILED
    assert not c.tools.backends.tickets.created()


def test_agent_event_log_replays_without_executing_tools(container):
    r = chat(container, "ana", "investigate why card payments fail at stores")
    events = container.tools.store.load(r.agent.run_id)
    report = replay(events)
    assert report.first_divergence is None and not report.misses
