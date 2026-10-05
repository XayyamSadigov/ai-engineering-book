# path: book/projects/toolkit/tests/test_policy.py
from __future__ import annotations

from toolkit import PolicyEngine, SideEffect, Tool, ToolContext, Verdict, recipient_allowlist

from .conftest import EchoArgs, SendArgs


def _send_tool(**kw) -> Tool:
    return Tool(name="send", description="send", args_model=SendArgs, handler=lambda a, e: None,
                side_effect=SideEffect.EXTERNAL, required_permission="mail:send", idempotent=False, **kw)


def test_missing_scope_denies(ctx):
    tool = _send_tool()
    no_scope = ctx.model_copy(update={"scopes": frozenset()})
    d = PolicyEngine().evaluate(tool, SendArgs(to="a@northwind.example", body="hi"), no_scope)
    assert d.verdict == Verdict.DENY and d.rules == ["permission"]


def test_external_side_effect_needs_approval(ctx):
    d = PolicyEngine().evaluate(_send_tool(), SendArgs(to="a@northwind.example", body="hi"), ctx)
    assert d.verdict == Verdict.NEEDS_APPROVAL
    assert "side_effect_requires_approval" in d.rules


def test_deny_rule_beats_approval(ctx):
    p = PolicyEngine()
    p.add_rule("send", recipient_allowlist("to", ["northwind.example"]), rule_id="recipient_allowlist")
    d = p.evaluate(_send_tool(), SendArgs(to="attacker@evil.example", body="data"), ctx)
    assert d.verdict == Verdict.DENY and d.rules == ["recipient_allowlist"]
    assert "attacker@evil.example" in d.reasons[0]


def test_escalating_rule_turns_allow_into_approval(ctx):
    p = PolicyEngine(approval_for=())
    tool = Tool(name="echo", description="e", args_model=EchoArgs, handler=lambda a, e: a)
    p.add_rule("echo", lambda a, c: "long text" if len(a.text) > 10 else None, rule_id="long",
               on_violation=Verdict.NEEDS_APPROVAL)
    assert p.evaluate(tool, EchoArgs(text="short"), ctx).allowed
    assert p.evaluate(tool, EchoArgs(text="a much longer text"), ctx).needs_approval


def test_rate_limit_per_user_with_retry_after(ctx):
    now = [0.0]
    p = PolicyEngine(clock=lambda: now[0])
    tool = Tool(name="echo", description="e", args_model=EchoArgs, handler=lambda a, e: a)
    p.set_rate_limit("echo", 2, 60)
    assert p.evaluate(tool, EchoArgs(text="a"), ctx).allowed
    assert p.evaluate(tool, EchoArgs(text="b"), ctx).allowed
    d = p.evaluate(tool, EchoArgs(text="c"), ctx)
    assert d.verdict == Verdict.DENY and d.rules == ["rate_limit"] and d.retry_after_s == 60
    other = ctx.model_copy(update={"user_id": "u2"})
    assert p.evaluate(tool, EchoArgs(text="d"), other).allowed  # separate bucket
    now[0] = 61
    assert p.evaluate(tool, EchoArgs(text="e"), ctx).allowed


def test_denied_calls_do_not_consume_rate_budget(ctx):
    p = PolicyEngine(clock=lambda: 0.0)
    p.add_rule("send", recipient_allowlist("to", ["northwind.example"]), rule_id="allow")
    p.set_rate_limit("send", 1, 60)
    tool = _send_tool()
    for _ in range(3):
        assert p.evaluate(tool, SendArgs(to="x@evil.example", body="b"), ctx).verdict == Verdict.DENY
    assert p.evaluate(tool, SendArgs(to="x@northwind.example", body="b"), ctx).needs_approval


def test_group_deny_hides_and_denies(ctx):
    p = PolicyEngine()
    p.deny_tool_for_group("contractor", "send")
    contractor = ctx.model_copy(update={"groups": frozenset({"contractor"})})
    assert not p.visible(_send_tool(), contractor)
    assert p.evaluate(_send_tool(), SendArgs(to="a@northwind.example", body="b"), contractor).rules == ["group_deny"]
