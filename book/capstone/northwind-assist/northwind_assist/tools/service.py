# path: book/capstone/northwind-assist/northwind_assist/tools/service.py
"""Tools and agents: Project 4's tools under toolkit's executor, driven by agentkit's loop.

Who decides what:
  * toolkit ToolExecutor (Ch 16): schema validation, policy (permissions, recipient allowlist,
    contractor denies, rate limits), idempotency, approval bound to the argument hash, audit;
  * guardrails tool stage (Ch 27): argument constraints, canaries and PII in outbound arguments;
  * agentkit AgentRuntime (Ch 19): the loop, budgets, Definition of Done, event log, replay.

The seam is `GuardedTool`: agentkit's `executor_tools` adapts each toolkit tool (with
`idempotency="content"`, so toolkit's content-bound key deduplicates the same action across runs
of one session, or a client Idempotency-Key when the request carries one), and the guardrail tool
stage sits in front of it.

The model only ever sees PII tokens (the input guardrail tokenized them) and a token-tolerant
schema. `guardrails.guard_tool_call` re-hydrates the e-mail argument inside the tool boundary,
and toolkit validates the real address against the original schema, so a send to the address
the user typed works while the model never handled the raw value.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any

from agentkit import (
    AgentRuntime,
    Budget,
    DefinitionOfDone,
    InMemoryEventStore,
    JsonlEventStore,
    LoopConfig,
    RunResult,
    ToolOutput,
    executor_tools,
    non_empty,
    tool_was_called,
)
from agentkit import SideEffect as AgentSideEffect
from agentkit import ToolContext as AgentToolContext
from agentkit.errors import ErrorClass
from aie_core.llm.gateway import PricingTable
from aie_core.llm.types import ToolCall
from guardrails import GuardContext, token_tolerant_schema
from support_assistant.config import AssistantSettings
from support_assistant.domain.directory import Directory
from support_assistant.domain.services import DraftStore, Outbox, StatusBoard
from support_assistant.domain.tickets import TicketStore, load_shared_tickets
from support_assistant.tools import Backends, build_policy, build_registry, summarize_for_approval
from toolkit import (
    ApprovalError,
    ApprovalManager,
    ApprovalRequest,
    AuditSink,
    InMemoryAuditLog,
    JsonlAuditLog,
    SQLiteIdempotencyStore,
    ToolContext,
    ToolExecutor,
    ToolResult,
    args_hash,
)

from .. import _paths
from ..config import Settings
from ..domain.context import RequestContext
from ..security.guards import Guards


@dataclass
class ToolActivity:
    """What the chat stream reports about tools during one request."""

    calls: list[dict[str, Any]] = field(default_factory=list)
    approvals: list[dict[str, Any]] = field(default_factory=list)


def _envelope(out: ToolOutput) -> dict[str, Any]:
    """toolkit's result envelope as the model read it ({ok, result | error, status, note})."""
    try:
        data = json.loads(out.content)
    except (TypeError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


class GuardedTool:
    """An agentkit tool from `executor_tools`, behind the guardrail tool stage.

    The model sees a token-tolerant schema; `guard_tool_call` re-hydrates the e-mail token and
    checks the re-hydrated call; the executor (policy, idempotency, approval, audit) runs exactly
    that call. Nothing here re-implements a check that a package already performs.
    """

    def __init__(self, inner: Any, layer: ToolLayer, tctx: ToolContext, gctx: GuardContext,
                 activity: ToolActivity) -> None:
        self._inner, self._layer, self._tctx, self._gctx, self._activity = inner, layer, tctx, gctx, activity
        self.name = inner.name
        raw = inner.spec
        self.spec = raw.model_copy(update={"parameters": token_tolerant_schema(raw.parameters)})
        self.side_effect = inner.side_effect
        self.idempotent = inner.idempotent
        self.requires_approval = False       # the executor owns approvals ...
        self.approval_by_executor = True     # ... so agentkit's DefaultPolicy does not gate twice

    def execute(self, arguments: dict[str, Any], ctx: AgentToolContext) -> ToolOutput:
        call, verdict = self._layer.guards.tool(ToolCall(id=ctx.call_id, name=self.name, arguments=arguments),
                                                self._gctx)
        if not verdict.allowed:
            self._activity.calls.append({"tool": self.name, "status": "blocked", "reason": verdict.reasons})
            return ToolOutput.failure(f"blocked by guardrail: {'; '.join(verdict.reasons)}", ErrorClass.PERMISSION)
        out = self._inner.execute(call.arguments, ctx)
        self._layer.note_output(self.name, out, self._tctx, self._activity)
        return out


def idempotency_policy(header_key: str | None) -> Any:
    """agentkit `executor_tools(idempotency=...)`: with a client Idempotency-Key, writes are keyed by
    it plus the action; without one, "content" lets toolkit derive its session-scoped content key."""
    if not header_key:
        return "content"

    def key(tool_name: str, arguments: dict[str, Any], ctx: Any) -> str:
        return f"{header_key}:{tool_name}:{args_hash(tool_name, arguments)[:16]}"

    return key


AGENT_DOD = {
    "create_ticket": DefinitionOfDone(non_empty(10), tool_was_called("create_ticket"),
                                      description="a ticket was created or an existing one was named"),
    "default": DefinitionOfDone(non_empty(10), description="the answer states what was done or found"),
}


class ToolLayer:
    def __init__(self, settings: Settings, guards: Guards, *, tracer: Any = None, pricing: PricingTable | None = None,
                 clock: Any = None) -> None:
        self.settings = settings
        self.guards = guards
        self.tracer = tracer
        self.pricing = pricing
        p4 = AssistantSettings(shared_data_dir=_paths.SHARED_DATA, allowed_recipient_domains=settings.allowed_mail_domains,
                               approval_ttl_s=settings.approval_ttl_s, four_eyes=settings.four_eyes)
        self.backends = Backends(directory=Directory(),
                                 tickets=TicketStore(load_shared_tickets(p4.shared_data_dir, p4.extra_tickets_path)),
                                 status=StatusBoard(), drafts=DraftStore(), outbox=Outbox())
        self.registry = build_registry(self.backends)
        self.policy = build_policy(p4)
        extra = {"clock": clock} if clock is not None else {}
        self.approvals = ApprovalManager(ttl_s=settings.approval_ttl_s, allow_self_approval=not settings.four_eyes,
                                         **extra)
        self.audit: AuditSink = JsonlAuditLog(settings.audit_log_path) if settings.audit_log_path else InMemoryAuditLog()
        self.idempotency = SQLiteIdempotencyStore(settings.idempotency_db)
        self.executor = ToolExecutor(self.registry, self.policy, approvals=self.approvals, idempotency=self.idempotency,
                                     audit=self.audit, tracer=tracer, summarize=summarize_for_approval,
                                     sleep=lambda s: None)
        self.store = JsonlEventStore(settings.event_dir) if settings.event_dir else InMemoryEventStore()
        self._requesters: dict[str, ToolContext] = {}   # approval id -> requester's context
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ tool calls
    def note_output(self, tool: str, out: ToolOutput, tctx: ToolContext, activity: ToolActivity) -> None:
        env = _envelope(out)
        status = env.get("status") or ("ok" if out.ok else "error")
        approval_id = env.get("approval_id")
        activity.calls.append({"tool": tool, "status": status, "duplicate": bool(env.get("note")) and out.ok,
                               "error": (env.get("error") or {}).get("code"), "approval_id": approval_id})
        if status == "pending_approval" and approval_id:
            with self._lock:
                self._requesters[approval_id] = tctx
            req = self.approvals.get(approval_id)
            if req is not None:
                activity.approvals.append(self.view(req))

    def tools_for(self, ctx: RequestContext, gctx: GuardContext, activity: ToolActivity, *,
                  allow_side_effects: bool) -> list[GuardedTool]:
        tctx = ctx.tool_context()
        inner = executor_tools(self.executor, tctx, idempotency=idempotency_policy(ctx.idempotency_key))
        return [GuardedTool(t, self, tctx, gctx, activity) for t in inner
                if allow_side_effects or t.side_effect is AgentSideEffect.READ]

    # ------------------------------------------------------------------ agent runs
    def run_agent(self, ctx: RequestContext, goal: str, *, llm: Any, system_prompt: str, gctx: GuardContext,
                  allow_side_effects: bool, task: str, run_id: str) -> tuple[RunResult, ToolActivity]:
        activity = ToolActivity()
        tools = self.tools_for(ctx, gctx, activity, allow_side_effects=allow_side_effects)
        s = self.settings
        remaining = ctx.deadline.remaining() if ctx.deadline is not None else s.agent_deadline_s
        budget = Budget(max_steps=s.agent_max_steps, max_tool_calls=s.agent_max_tool_calls,
                        max_cost_usd=s.agent_max_cost_usd, deadline_s=max(0.5, min(s.agent_deadline_s, remaining)))
        runtime = AgentRuntime(llm, tools, system_prompt=system_prompt, budget=budget,
                               dod=AGENT_DOD.get(task, AGENT_DOD["default"]), store=self.store, tracer=self.tracer,
                               pricing=self.pricing, config=LoopConfig(max_tokens_per_call=512),
                               principal={"user_id": ctx.user_id, "tenant": ctx.tenant})
        result = runtime.run(goal, run_id=run_id, metadata={"task_id": task, "agent_version": "capstone-agent@1",
                                                            "tenant": ctx.tenant, "request_id": ctx.request_id})
        return result, activity

    # ------------------------------------------------------------------ approvals
    @staticmethod
    def view(req: ApprovalRequest) -> dict[str, Any]:
        return {"approval_id": req.id, "tool": req.tool_name, "arguments": req.arguments, "args_hash": req.args_hash,
                "summary": req.summary, "requested_by": req.user_id, "status": req.status.value,
                "expires_at": req.expires_at}

    def pending(self, ctx: RequestContext) -> list[dict[str, Any]]:
        if ctx.has_scope("replies:approve"):
            items = self.approvals.pending(tenant=ctx.tenant)
        else:
            items = self.approvals.pending(tenant=ctx.tenant, user_id=ctx.user_id)
        return [self.view(i) for i in items]

    def _authorized(self, approval_id: str, ctx: RequestContext, *, deciding: bool) -> ApprovalRequest:
        req = self.approvals.get(approval_id)
        if req is None or req.tenant != ctx.tenant:   # foreign and missing look identical: 404
            raise ApprovalError("approval_not_found", f"no approval {approval_id}")
        if deciding and not ctx.has_scope("replies:approve"):
            raise ApprovalError("not_an_approver", "only a lead may approve actions")
        return req

    def approve(self, approval_id: str, ctx: RequestContext, note: str | None = None) -> ToolResult:
        """Record the decision, then run exactly the stored arguments as the requester."""
        self._authorized(approval_id, ctx, deciding=True)
        with self._lock:
            requester = self._requesters.get(approval_id)
        if requester is None:
            # The requester's scopes and groups are not recoverable from the approval record, and
            # inventing them would let policy pass for a user who lost a role or is a contractor.
            # Fail closed before recording a decision; durable approvals must persist the requester
            # context with the approval (Chapter 39, practical exercise P1).
            raise ApprovalError("requester_context_lost",
                                "the requester's context is gone (restart); ask them to resubmit")
        self.approvals.approve(approval_id, ctx.user_id, note)
        return self.executor.execute_approved(approval_id, requester)

    def reject(self, approval_id: str, ctx: RequestContext, note: str | None = None) -> ApprovalRequest:
        self._authorized(approval_id, ctx, deciding=True)   # four-eyes applies to rejections too
        return self.approvals.reject(approval_id, ctx.user_id, note)

    def schema_versions(self) -> dict[str, str]:
        import hashlib  # noqa: PLC0415

        return {t.name: hashlib.sha256(json.dumps(t.spec().model_dump(mode="json"), sort_keys=True).encode())
                .hexdigest()[:12] for t in self.registry.select()}


__all__ = ["ToolLayer", "GuardedTool", "ToolActivity", "AGENT_DOD", "idempotency_policy"]
