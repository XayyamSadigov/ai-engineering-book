# path: book/projects/toolkit/tests/test_executor.py
from __future__ import annotations

import time

from aie_core.llm.types import ToolCall
from pydantic import BaseModel

from toolkit import (ErrorCategory, InMemoryIdempotencyStore, PolicyEngine, SideEffect, SQLiteIdempotencyStore,
                     Tool, ToolError, ToolExecutor, ToolRegistry, recipient_allowlist, truncate_payload)

from .conftest import Counter, EchoArgs, SendArgs


def call(name: str, **args) -> ToolCall:
    return ToolCall(id=f"c-{name}", name=name, arguments=args)


def test_unknown_tool_and_invalid_args_are_machine_readable(harness, ctx):
    ex = harness["ex"]
    r = ex.execute(call("nope"), ctx)
    assert r.error["category"] == "not_found" and "echo" in r.error["details"]["available"]
    r = ex.execute(call("echo", text=""), ctx)
    assert r.status == "error" and r.error["category"] == "validation"
    assert r.error["details"]["errors"][0]["field"] == "text"
    r = ex.execute(call("echo", text="hi", bcc="x@evil.example"), ctx)  # invented key: rejected, not dropped
    assert r.error["details"]["errors"] == [{"field": "bcc", "problem": "unknown field", "type": "extra_forbidden"}]
    assert harness["echo"].calls == 0


def test_read_tool_runs_and_audits(harness, ctx):
    r = harness["ex"].execute(call("echo", text="hello"), ctx)
    assert r.ok and r.data == {"echo": "hello", "n": 1}
    assert '"ok": true' in r.content
    assert harness["audit"].types() == ["tool.proposed", "tool.executed"]
    ev = harness["audit"].events[-1]
    assert ev.args_hash == r.args_hash and ev.arguments is None and ev.tool_fingerprint


def test_permission_denied_is_not_retried(harness, ctx):
    no_scope = ctx.model_copy(update={"scopes": frozenset()})
    r = harness["ex"].execute(call("echo", text="hi"), no_scope)
    assert r.error["category"] == "permission" and r.error["retryable"] is False
    assert harness["echo"].calls == 0
    assert "tool.denied" in harness["audit"].types()


def test_transient_errors_are_retried_then_succeed(harness, ctx):
    echo = harness["echo"]
    echo.fail_first, echo.fail_with = 2, ToolError.transient("upstream_503", "busy")
    r = harness["ex"].execute(call("echo", text="hi"), ctx)
    assert r.ok and r.attempts == 3
    assert harness["audit"].types().count("tool.retry") == 2


def test_transient_errors_exhaust_budget(harness, ctx):
    echo = harness["echo"]
    echo.fail_first, echo.fail_with = 10, ToolError.transient("upstream_503", "busy")
    r = harness["ex"].execute(call("echo", text="hi"), ctx)
    assert r.error["category"] == "transient" and r.attempts == 3


def test_validation_error_from_handler_is_not_retried(harness, ctx):
    echo = harness["echo"]
    echo.fail_first, echo.fail_with = 10, ToolError.not_found("no_such_ticket", "TCK-9 not found")
    r = harness["ex"].execute(call("echo", text="hi"), ctx)
    assert r.error["code"] == "no_such_ticket" and r.attempts == 1


def test_handler_bug_is_fatal_without_leaking_message(harness, ctx):
    def boom(a, e):
        raise KeyError("secret-internal-name")

    harness["reg"].register(Tool(name="boom", description="b", args_model=EchoArgs, handler=boom))
    r = harness["ex"].execute(call("boom", text="x"), ctx)
    assert r.error["category"] == "fatal" and "secret-internal-name" not in r.content


def test_approval_flow_binds_to_arguments(harness, ctx):
    ex, approvals, send = harness["ex"], harness["approvals"], harness["send"]
    r = ex.execute(call("send", to="a@northwind.example", body="hi"), ctx)
    assert r.status == "pending_approval" and send.calls == 0
    approvals.approve(r.approval_id, "lead1")
    # Same approval, different body: refused.
    bad = ex.execute(call("send", to="a@northwind.example", body="CHANGED"), ctx, approval_id=r.approval_id)
    assert bad.error["code"] == "approval_mismatch" and send.calls == 0
    ok = ex.execute_approved(r.approval_id, ctx)
    assert ok.ok and send.calls == 1
    # Approval is single use, and the idempotency record answers a replay.
    again = ex.execute_approved(r.approval_id, ctx)
    assert again.ok and again.duplicate and send.calls == 1


def test_rejected_approval_is_permission_error(harness, ctx):
    ex, approvals = harness["ex"], harness["approvals"]
    r = ex.execute(call("send", to="a@northwind.example", body="hi"), ctx)
    approvals.reject(r.approval_id, "lead1")
    out = ex.execute_approved(r.approval_id, ctx)
    assert out.error["code"] == "approval_rejected"
    assert "tool.approval_rejected" in harness["audit"].types()


def _writer_executor(store, handler, *, timeout_s=5.0, reconcile=None):
    class TicketArgs(BaseModel):
        title: str

    reg = ToolRegistry([Tool(name="create", description="Create a ticket.", args_model=TicketArgs, handler=handler,
                             side_effect=SideEffect.REVERSIBLE_WRITE, idempotent=False, timeout_s=timeout_s,
                             reconcile=reconcile)])
    return ToolExecutor(reg, PolicyEngine(), idempotency=store, sleep=lambda s: None)


def test_duplicate_side_effect_suppressed_in_memory_and_sqlite(ctx, tmp_path):
    for store in (InMemoryIdempotencyStore(), SQLiteIdempotencyStore(tmp_path / "idem.db")):
        created: list[str] = []
        ex = _writer_executor(store, lambda a, e: created.append(a.title) or {"id": f"T{len(created)}"})
        first = ex.execute(call("create", title="VPN down"), ctx)
        second = ex.execute(call("create", title="VPN down"), ctx)
        third = ex.execute(call("create", title="Printer jam"), ctx)
        assert first.data == {"id": "T1"} and second.duplicate and second.data == {"id": "T1"}
        assert third.data == {"id": "T2"} and created == ["VPN down", "Printer jam"]
        ex.close()


def test_store_outage_fails_closed_for_writes(ctx):
    class DownStore(InMemoryIdempotencyStore):
        def get(self, key):
            raise ConnectionError("idempotency store unreachable")

    created: list[str] = []
    ex = _writer_executor(DownStore(), lambda a, e: created.append(a.title) or {"id": "T1"})
    result = ex.execute(call("create", title="VPN down"), ctx)
    assert result.status == "error" and result.error["code"] == "idempotency_unavailable"
    assert result.error["category"] == "transient" and created == [], "no write without duplicate protection"


def test_explicit_key_reused_for_different_args_is_rejected(ctx):
    ex = _writer_executor(InMemoryIdempotencyStore(), lambda a, e: {"id": "T1"})
    ex.execute(call("create", title="A"), ctx, idempotency_key="k1")
    r = ex.execute(call("create", title="B"), ctx, idempotency_key="k1")
    assert r.error["code"] == "idempotency_key_reused"


def test_timeout_on_write_is_outcome_unknown_and_not_retried(ctx):
    calls = []

    def slow(a, e):
        calls.append(1)
        time.sleep(0.5)
        return {"id": "T1"}

    store = InMemoryIdempotencyStore()
    ex = _writer_executor(store, slow, timeout_s=0.05)
    r = ex.execute(call("create", title="A"), ctx)
    assert r.error["code"] == "outcome_unknown" and r.error["category"] == ErrorCategory.FATAL.value
    assert len(calls) == 1
    again = ex.execute(call("create", title="A"), ctx)
    assert again.error["code"] == "outcome_unknown" and len(calls) == 1
    ex.close()


def test_reconcile_resolves_unknown_outcome(ctx):
    store = InMemoryIdempotencyStore()
    side_effects: list[str] = []

    def slow(a, e):
        side_effects.append(e.idempotency_key)
        time.sleep(0.3)
        return {"id": "T1"}

    def reconcile(a, e):  # ask the system of record by idempotency key
        return {"id": "T1", "reconciled": True} if e.idempotency_key in side_effects else None

    ex = _writer_executor(store, slow, timeout_s=0.05, reconcile=reconcile)
    assert ex.execute(call("create", title="A"), ctx).error["code"] == "outcome_unknown"
    r = ex.execute(call("create", title="A"), ctx)
    assert r.ok and r.duplicate and r.data["reconciled"] and len(side_effects) == 1
    ex.close()


def test_timeout_on_read_is_retried():
    attempts = []

    def flaky(a, e):
        attempts.append(1)
        if len(attempts) == 1:
            time.sleep(0.3)
        return "ok"

    reg = ToolRegistry([Tool(name="r", description="read", args_model=EchoArgs, handler=flaky, timeout_s=0.05)])
    ex = ToolExecutor(reg, sleep=lambda s: None)
    from toolkit import ToolContext
    r = ex.execute(call("r", text="x"), ToolContext(user_id="u", tenant="t"))
    assert r.ok and r.attempts == 2
    ex.close()


def test_policy_rate_limit_maps_to_transient_with_retry_after(harness, ctx):
    harness["policy"].set_rate_limit("echo", 1, 30)
    assert harness["ex"].execute(call("echo", text="a"), ctx).ok
    r = harness["ex"].execute(call("echo", text="b"), ctx)
    assert r.error["code"] == "rate_limited" and r.error["retry_after_s"] > 0


def test_allowlist_blocks_before_approval_is_created(harness, ctx):
    harness["policy"].add_rule("send", recipient_allowlist("to", ["northwind.example"]), rule_id="recipient_allowlist")
    r = harness["ex"].execute(call("send", to="attacker@evil.example", body="dump"), ctx)
    assert r.error["category"] == "permission" and harness["approvals"].pending() == []


def test_truncation_is_structural_and_announced():
    rows = [{"id": i, "text": "x" * 50} for i in range(100)]
    shown, truncated = truncate_payload({"query": "vpn", "results": rows}, 1000)
    assert truncated and shown["_truncation"]["total"] == 100 and 0 < shown["_truncation"]["returned"] < 100
    shown, truncated = truncate_payload("y" * 5000, 300)
    assert truncated and shown["omitted_chars"] > 0
    assert truncate_payload({"a": 1}, 100) == ({"a": 1}, False)


def test_result_limit_applies_to_model_content_not_data(ctx):
    reg = ToolRegistry([Tool(name="big", description="big", args_model=EchoArgs, max_result_chars=500,
                             handler=lambda a, e: [{"n": i, "pad": "z" * 40} for i in range(200)])])
    ex = ToolExecutor(reg)
    r = ex.execute(call("big", text="x"), ctx)
    assert r.truncated and len(r.content) < 700 and len(r.data) == 200
    ex.close()


def test_bound_tools_for_agent_runtimes(harness, ctx):
    class AgentCtx:  # what Chapter 19's runtime passes
        call_id = "step3-call1"
        idempotency_key = "run42:step3:send"

    reader = ctx.model_copy(update={"scopes": frozenset({"tickets:read"})})
    assert [b.name for b in harness["ex"].bind(reader)] == ["echo"]
    bound = {b.name: b for b in harness["ex"].bind(ctx)}
    assert bound["send"].side_effect == "irreversible" and bound["echo"].side_effect == "read"
    assert bound["echo"].spec.name == "echo"
    r = bound["echo"].execute({"text": "hi"}, AgentCtx())
    assert r.ok and r.call_id == "step3-call1"
    pending = bound["send"].execute({"to": "a@northwind.example", "body": "x"}, AgentCtx())
    assert pending.status == "pending_approval" and harness["send"].calls == 0
