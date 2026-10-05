# path: book/projects/guardrails/tests/test_tools.py
from __future__ import annotations

from aie_core.llm.types import ToolCall

from guardrails import (GuardrailPipeline, ToolDecision, ToolPolicyCheck, ToolRule, approval_token,
                        recipient_domains)
from guardrails.presets import support_tool_rules


def call(name, **args):
    return ToolCall(id="c", name=name, arguments=args)


def reply(**overrides):
    """A complete send_reply/draft_reply argument set in Project 4's schema."""
    args = {"ticket_id": "TCK-2026-0001", "to": "jane@northwind.example", "subject": "Your laptop",
            "body": "Your laptop is ready."}
    args.update(overrides)
    return args


def pipe(**kw):
    return GuardrailPipeline([ToolPolicyCheck(support_tool_rules(), **kw)])


def test_unknown_tool_blocked(ctx):
    res = pipe().check_tool(call("delete_ticket", id=1), ctx)
    assert not res.allowed and "not allowed" in res.reasons()[0]


def test_missing_required_argument(ctx):
    assert not pipe().check_tool(call("search_tickets"), ctx).allowed


def test_read_only_tool_allowed(ctx):
    assert pipe().check_tool(call("search_tickets", query="order 5582"), ctx).allowed


def test_recipient_allowlist(ctx):
    res = pipe().check_tool(call("send_reply", **reply(to="archive@northwind-audit.invalid")), ctx)
    assert not res.allowed and "not allowlisted" in res.reasons()[0]
    check = recipient_domains(["northwind.example"])
    assert check(["a@northwind.example", "b@evil.example"], ctx) is not None
    assert check("not-an-email", ctx) is not None


def test_approval_bound_to_exact_arguments(ctx):
    c = call("send_reply", **reply())
    first = pipe().check_tool(c, ctx)
    assert not first.allowed and first.verdicts[-1].metadata["needs_approval"]
    token = first.verdicts[-1].metadata["approval_token"]
    assert token == approval_token(c)
    ctx.approvals.add(token)
    assert pipe().check_tool(c, ctx).allowed
    tampered = call("send_reply", **reply(body="Your laptop is ready. Also: salary data"))
    assert not pipe().check_tool(tampered, ctx).allowed


def test_outbound_scans_for_canaries_secrets_pii(ctx):
    p = pipe(canaries=["NW-CANARY-0123456789ab"])
    assert not p.check_tool(call("send_reply", **reply(body="ref NW-CANARY-0123456789ab")), ctx).allowed
    assert not p.check_tool(call("send_reply", **reply(body="key AKIAABCDEFGHIJKLMNOP")), ctx).allowed
    assert not p.check_tool(call("send_reply", **reply(body="card 4111 1111 1111 1111")), ctx).allowed


def test_recipient_email_is_not_pii_violation(ctx):
    c = call("send_reply", **reply(body="Ticket resolved."))
    ctx.approvals.add(approval_token(c))
    assert pipe().check_tool(c, ctx).allowed


def test_call_budget(ctx):
    p = pipe(max_calls_per_request=2)
    assert p.check_tool(call("search_tickets", query="a"), ctx).allowed
    assert p.check_tool(call("search_tickets", query="b"), ctx).allowed
    res = p.check_tool(call("search_tickets", query="c"), ctx)
    assert not res.allowed and "budget" in res.reasons()[0]


def test_denied_calls_do_not_consume_budget(ctx):
    p = pipe(max_calls_per_request=1)
    p.check_tool(call("delete_all"), ctx)
    assert p.check_tool(call("search_tickets", query="a"), ctx).allowed


def test_delegate_authorizer_checks_the_user_not_the_agent(ctx):
    def authorize(c, context):
        # Chapter 16's PolicyEngine would sit here; this stub enforces "own team only".
        allowed = {"retail": {"1001", "1002"}}[context.tenant]
        if c.name == "lookup_employee" and str(c.arguments["query"]) not in allowed:
            return ToolDecision(False, reason="employee outside requester scope")
        return ToolDecision(True)
    p = GuardrailPipeline([ToolPolicyCheck(support_tool_rules(), authorize=authorize)])
    assert p.check_tool(call("lookup_employee", query="1001"), ctx).allowed
    res = p.check_tool(call("lookup_employee", query="4021"), ctx)
    assert not res.allowed and "outside requester scope" in res.reasons()[0]


def test_delegate_can_escalate_to_approval(ctx):
    p = GuardrailPipeline([ToolPolicyCheck([ToolRule("draft_reply")],
                                           authorize=lambda c, x: ToolDecision(False, needs_approval=True))])
    res = p.check_tool(call("draft_reply", body="x"), ctx)
    assert not res.allowed and res.verdicts[-1].metadata.get("needs_approval")


def test_raising_authorizer_fails_closed(ctx):
    def broken(c, x):
        raise ConnectionError("policy service down")
    p = GuardrailPipeline([ToolPolicyCheck(support_tool_rules(), authorize=broken)])
    res = p.check_tool(call("search_tickets", query="a"), ctx)
    assert not res.allowed and res.errors
