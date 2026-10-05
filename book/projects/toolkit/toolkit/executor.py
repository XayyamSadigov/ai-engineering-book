# path: book/projects/toolkit/toolkit/executor.py
"""The only code path that runs a tool.

execute(call, ctx) performs, in order:
  1. lookup            unknown tool                       -> not_found
  2. validation        pydantic on the raw model arguments -> validation (model repairs)
  3. policy            permission, constraints, rate limit -> permission / transient
  4. idempotency peek  same action already done?           -> recorded result, no side effect
  5. approval          bound to tool + args hash           -> pending_approval or permission
  6. reservation       atomic begin on the idempotency key
  7. run               timeout, bounded retries for transient errors only
  8. record            store result, consume approval, truncate for the model, audit
Every step emits an audit event, so the trail explains every outcome.
"""
from __future__ import annotations

import json
import random
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass
from typing import Any, Literal

from aie_core.llm.types import Message, ToolCall
from aie_core.observability import NoopTracer, Tracer
from pydantic import BaseModel, ValidationError

from .approval import ApprovalError, ApprovalManager
from .audit import AuditEvent, AuditSink, NullAuditLog
from .errors import ErrorCategory, ToolError
from .idempotency import IdempotencyRecord, IdempotencyStore
from .policy import Decision, PolicyEngine, ToolContext, Verdict
from .registry import Tool, ToolRegistry, args_hash


@dataclass(frozen=True)
class ExecutionContext:
    """What a handler receives besides its arguments."""

    tool_context: ToolContext
    tool_name: str
    call_id: str
    args_hash: str
    attempt: int
    deadline: float                     # time.monotonic() value; pass the remainder downstream
    idempotency_key: str | None = None  # forward to downstream APIs that accept one
    approval_id: str | None = None

    @property
    def user_id(self) -> str:
        return self.tool_context.user_id

    @property
    def tenant(self) -> str:
        return self.tool_context.tenant

    def remaining_s(self) -> float:
        return max(0.0, self.deadline - time.monotonic())


ResultStatus = Literal["ok", "error", "pending_approval"]


class ToolResult(BaseModel):
    call_id: str
    tool_name: str
    status: ResultStatus
    content: str                         # exactly what the model will read
    data: Any = None                     # full, untruncated result for the application
    error: dict[str, Any] | None = None
    approval_id: str | None = None
    args_hash: str | None = None
    truncated: bool = False
    duplicate: bool = False              # served from the idempotency store
    attempts: int = 0
    latency_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def error_category(self) -> str | None:
        return (self.error or {}).get("category")

    @property
    def error_class(self) -> str | None:  # name read by Chapter 19's agentkit adapter
        return self.error_category

    def to_message(self) -> Message:
        return Message.tool(self.call_id, self.content)


def _unknown_fields(model: type[BaseModel], raw: dict[str, Any]) -> list[dict[str, str]]:
    """The schema we publish is closed, so a key the model invented is an error, not noise.
    (A silently dropped `bcc` is worse than a rejected one.)"""
    known = set(model.model_fields) | {f.alias for f in model.model_fields.values() if f.alias}
    return [{"field": k, "problem": "unknown field", "type": "extra_forbidden"} for k in raw if k not in known]


# ----------------------------------------------------------------------- truncation
def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def truncate_payload(data: Any, limit: int) -> tuple[Any, bool]:
    """Shrink a JSON-able result to at most `limit` characters, structurally if possible.

    Lists lose trailing items (with a note saying how many were kept), dicts lose items
    from their largest list field, and anything else is cut as text. The model is always
    told that truncation happened, so it can narrow its query instead of assuming it saw
    everything.
    """
    if len(_dumps(data)) <= limit:
        return data, False

    def fit_list(items: list[Any], wrap: Callable[[list[Any]], Any]) -> Any | None:
        lo, hi, best = 0, len(items), None
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = wrap(items[:mid])
            if len(_dumps(candidate)) <= limit:
                best, lo = candidate, mid + 1
            else:
                hi = mid - 1
        return best

    if isinstance(data, list):
        note = lambda kept: {"items": kept, "truncated": True, "returned": len(kept), "total": len(data)}  # noqa: E731
        fitted = fit_list(data, note)
        if fitted is not None and fitted["returned"] > 0:
            return fitted, True
    if isinstance(data, dict):
        lists = [(k, v) for k, v in data.items() if isinstance(v, list)]
        if lists:
            key, items = max(lists, key=lambda kv: len(_dumps(kv[1])))

            def wrap(kept: list[Any]) -> dict[str, Any]:
                return {**data, key: kept,
                        "_truncation": {"field": key, "returned": len(kept), "total": len(items)}}

            fitted = fit_list(items, wrap)
            if fitted is not None:
                return fitted, True
    text = _dumps(data)
    keep = max(0, limit - 120)
    return {"truncated_text": text[:keep], "truncated": True, "omitted_chars": len(text) - keep}, True


# ------------------------------------------------------------------- agent adapter
_COARSE_SIDE_EFFECT = {"read": "read", "reversible_write": "write", "irreversible": "irreversible",
                       "external": "irreversible"}


class BoundTool:
    """One tool bound to one principal, shaped for agent runtimes (Chapter 19 `adapt_tool`).

    The runtime sees `name`, `spec`, `side_effect`, `requires_approval`, `idempotent`, and
    calls `execute(arguments, ctx)`. Every call still goes through `ToolExecutor.execute`,
    so policy, approvals, idempotency, and audit cannot be bypassed by the runtime.
    """

    def __init__(self, executor: "ToolExecutor", tool: Tool, principal: ToolContext) -> None:
        self._executor, self._tool, self._principal = executor, tool, principal
        self.name = tool.name
        self.spec = tool.spec()
        self.side_effect = _COARSE_SIDE_EFFECT[tool.side_effect.value]
        self.side_effect_class = tool.side_effect
        self.requires_approval = tool.requires_approval
        self.idempotent = tool.idempotent

    def execute(self, arguments: dict[str, Any], ctx: Any = None) -> "ToolResult":
        call_id = getattr(ctx, "call_id", None) or f"{self.name}-{id(arguments)}"
        key = getattr(ctx, "idempotency_key", None)
        approval_id = getattr(ctx, "approval_id", None)
        return self._executor.execute(ToolCall(id=call_id, name=self.name, arguments=arguments),
                                      self._principal, approval_id=approval_id, idempotency_key=key)


# ------------------------------------------------------------------------- executor
class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        policy: PolicyEngine | None = None,
        *,
        approvals: ApprovalManager | None = None,
        idempotency: IdempotencyStore | None = None,
        audit: AuditSink | None = None,
        tracer: Tracer | None = None,
        max_attempts: int = 3,
        base_delay_s: float = 0.25,
        max_delay_s: float = 4.0,
        idempotency_ttl_s: float = 24 * 3600,
        include_arguments_in_audit: bool = False,
        summarize: Callable[[Tool, dict[str, Any]], str] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_workers: int = 8,
    ) -> None:
        self.registry = registry
        self.policy = policy or PolicyEngine()
        self.approvals = approvals
        self.idempotency = idempotency
        self.audit = audit or NullAuditLog()
        self.tracer = tracer or NoopTracer()
        self.max_attempts = max_attempts
        self.base_delay_s = base_delay_s
        self.max_delay_s = max_delay_s
        self.idempotency_ttl_s = idempotency_ttl_s
        self.include_arguments_in_audit = include_arguments_in_audit
        self.summarize = summarize
        self._sleep = sleep
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tool")
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ public API
    def visible_specs(self, ctx: ToolContext, **filters: Any) -> list[Any]:
        """ToolSpecs this principal may see for this task (discovery, not authorization)."""
        return self.registry.specs(ctx, visible=self.policy.visible, **filters)

    def execute(self, call: ToolCall, ctx: ToolContext, *, approval_id: str | None = None,
                idempotency_key: str | None = None) -> ToolResult:
        start = time.monotonic()
        with self.tracer.span("tool.execute", tool=call.name, call_id=call.id, user_id=ctx.user_id) as span:
            result = self._execute(call, ctx, approval_id, idempotency_key, start)
            result.latency_ms = round((time.monotonic() - start) * 1000, 3)
            span.set_attribute("tool.status", result.status)
            span.set_attribute("tool.duplicate", result.duplicate)
            span.set_attribute("tool.attempts", result.attempts)
            if result.error:
                span.set_attribute("tool.error_category", result.error["category"])
            return result

    def bind(self, ctx: ToolContext, **filters: Any) -> list[BoundTool]:
        """Tools this principal may see, each bound to `ctx`, for an agent runtime."""
        return [BoundTool(self, t, ctx) for t in self.registry.select(ctx, visible=self.policy.visible, **filters)]

    def execute_approved(self, approval_id: str, ctx: ToolContext) -> ToolResult:
        """Run exactly the action a human approved, with the stored arguments."""
        if self.approvals is None:
            raise RuntimeError("executor has no ApprovalManager")
        item = self.approvals.get(approval_id)
        if item is None:
            err = ToolError.not_found("approval_not_found", f"no approval {approval_id}")
            return self._error(ToolCall(id=approval_id, name="?", arguments={}), err, None)
        call = ToolCall(id=f"approved-{approval_id}", name=item.tool_name, arguments=item.arguments)
        return self.execute(call, ctx, approval_id=approval_id)

    def default_idempotency_key(self, tool: Tool, h: str, ctx: ToolContext) -> str:
        """Same principal + same session + same tool + same normalized args = same action."""
        return f"{tool.name}:{ctx.tenant}:{ctx.session_id or ctx.user_id}:{h[:32]}"

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------- pipeline
    def _execute(self, call: ToolCall, ctx: ToolContext, approval_id: str | None,
                 explicit_key: str | None, start: float) -> ToolResult:
        # 1. lookup
        if call.name not in self.registry:
            err = ToolError.not_found("unknown_tool", f"no tool named '{call.name}'",
                                      details={"available": self.registry.names()})
            self._emit("tool.invalid", call.name, ctx, call_id=call.id, error=err)
            return self._error(call, err, None)
        tool = self.registry.get(call.name)

        # 2. validation, outside the model, before anything else sees the arguments
        try:
            args = tool.args_model.model_validate(call.arguments)
            problems = _unknown_fields(tool.args_model, call.arguments)
        except ValidationError as exc:
            problems = [{"field": ".".join(str(p) for p in e["loc"]) or "(root)", "problem": e["msg"],
                         "type": e["type"]} for e in exc.errors()]
        if problems:
            err = ToolError.validation("invalid_arguments", "arguments do not match the tool schema",
                                       details={"errors": problems})
            self._emit("tool.invalid", tool.name, ctx, call_id=call.id, error=err, tool=tool)
            return self._error(call, err, None)
        normalized = args.model_dump(mode="json")
        h = args_hash(tool.name, normalized)
        self._emit("tool.proposed", tool.name, ctx, call_id=call.id, args_hash=h, tool=tool,
                   arguments=normalized, approval_id=approval_id)

        # 3. policy
        decision = self.policy.evaluate(tool, args, ctx)
        if decision.verdict == Verdict.DENY:
            if "rate_limit" in decision.rules:
                err = ToolError.transient("rate_limited", "; ".join(decision.reasons),
                                          retry_after_s=decision.retry_after_s)
            else:
                err = ToolError.permission("policy_denied", "; ".join(decision.reasons),
                                           details={"rules": decision.rules})
            self._emit("tool.denied", tool.name, ctx, call_id=call.id, args_hash=h, tool=tool,
                       decision=decision, error=err)
            return self._error(call, err, h)

        # 4. idempotency peek: a completed identical action is answered from the record
        key = None
        if self.idempotency is not None and not tool.idempotent:
            key = explicit_key or self.default_idempotency_key(tool, h, ctx)
            try:
                existing = self.idempotency.get(key)
            except Exception as exc:  # noqa: BLE001 - store outage: fail closed, nothing has run
                return self._store_unavailable(call, ctx, tool, h, exc)
            if existing is not None:
                handled = self._handle_existing(existing, tool, args, call, ctx, h, key, approval_id)
                if handled is not None:
                    return handled

        # 5. approval bound to tool + args hash
        if decision.verdict == Verdict.NEEDS_APPROVAL:
            gate = self._approval_gate(tool, normalized, call, ctx, h, decision, approval_id)
            if gate is not None:
                return gate

        # 6. reservation
        if key is not None:
            assert self.idempotency is not None
            try:
                existing = self.idempotency.begin(key, tool.name, h, self.idempotency_ttl_s)
            except Exception as exc:  # noqa: BLE001
                return self._store_unavailable(call, ctx, tool, h, exc)
            if existing is not None:
                handled = self._handle_existing(existing, tool, args, call, ctx, h, key, approval_id)
                if handled is not None:
                    return handled
                if self.idempotency.begin(key, tool.name, h, self.idempotency_ttl_s) is not None:
                    err = ToolError.transient("in_progress", "an identical action is in progress", retry_after_s=1.0)
                    return self._error(call, err, h)

        # 7. run
        return self._run(tool, args, call, ctx, h, key, approval_id, start)

    def _approval_gate(self, tool: Tool, normalized: dict[str, Any], call: ToolCall, ctx: ToolContext,
                       h: str, decision: Decision, approval_id: str | None) -> ToolResult | None:
        if self.approvals is None:
            err = ToolError.permission("approval_unavailable", "this action needs approval and none is configured")
            self._emit("tool.denied", tool.name, ctx, call_id=call.id, args_hash=h, tool=tool,
                       decision=decision, error=err)
            return self._error(call, err, h)
        if approval_id is None:
            summary = self.summarize(tool, normalized) if self.summarize else ""
            req = self.approvals.request(tool.name, normalized, ctx, reasons=decision.reasons, summary=summary)
            self._emit("tool.approval_requested", tool.name, ctx, call_id=call.id, args_hash=h, tool=tool,
                       decision=decision, approval_id=req.id)
            content = _dumps({"ok": False, "status": "pending_approval", "approval_id": req.id,
                              "message": "A human must approve this exact action. It has NOT been performed. "
                                         "Tell the user it is awaiting approval; do not call it again."})
            return ToolResult(call_id=call.id, tool_name=tool.name, status="pending_approval",
                              content=content, approval_id=req.id, args_hash=h)
        try:
            self.approvals.verify(approval_id, tool.name, h, ctx)
        except ApprovalError as exc:
            err = ToolError.permission(exc.code, exc.message)
            event = "tool.approval_rejected" if exc.code == "approval_rejected" else "tool.denied"
            self._emit(event, tool.name, ctx, call_id=call.id, args_hash=h, tool=tool, error=err,
                       approval_id=approval_id)
            return self._error(call, err, h)
        return None

    def _handle_existing(self, rec: IdempotencyRecord, tool: Tool, args: BaseModel, call: ToolCall,
                         ctx: ToolContext, h: str, key: str, approval_id: str | None) -> ToolResult | None:
        """Decide what an existing idempotency record means. None = proceed to execute."""
        assert self.idempotency is not None
        if rec.args_hash != h:
            err = ToolError.validation("idempotency_key_reused",
                                       "this idempotency key was used for a different action")
            return self._error(call, err, h)
        if rec.status == "succeeded":
            self._emit("tool.duplicate_suppressed", tool.name, ctx, call_id=call.id, args_hash=h, tool=tool,
                       idempotency_key=key)
            return self._ok(call, tool, rec.result, h, attempts=0, duplicate=True)
        if rec.status == "in_progress":
            return self._error(call, ToolError.transient("in_progress", "an identical action is in progress",
                                                         retry_after_s=1.0), h)
        # unknown: ask the system of record, never guess
        if tool.reconcile is not None:
            ex = ExecutionContext(ctx, tool.name, call.id, h, 0, time.monotonic() + tool.timeout_s, key, approval_id)
            found = tool.reconcile(args, ex)
            if found is not None:
                value = _jsonable(found)
                self.idempotency.complete(key, value)
                self._emit("tool.duplicate_suppressed", tool.name, ctx, call_id=call.id, args_hash=h,
                           tool=tool, idempotency_key=key, reasons=["reconciled"])
                return self._ok(call, tool, value, h, attempts=0, duplicate=True)
            self.idempotency.release(key)
            return None
        err = ToolError.fatal("outcome_unknown",
                              "a previous attempt may or may not have taken effect; a human must reconcile it")
        self._emit("tool.failed", tool.name, ctx, call_id=call.id, args_hash=h, tool=tool, error=err,
                   idempotency_key=key)
        return self._error(call, err, h)

    def _run(self, tool: Tool, args: BaseModel, call: ToolCall, ctx: ToolContext, h: str,
             key: str | None, approval_id: str | None, start: float) -> ToolResult:
        attempt = 0
        while True:
            attempt += 1
            ex = ExecutionContext(ctx, tool.name, call.id, h, attempt, time.monotonic() + tool.timeout_s,
                                  key, approval_id)
            t0 = time.monotonic()
            try:
                future = self._pool.submit(tool.handler, args, ex)
                value = future.result(timeout=tool.timeout_s)
            except ToolError as err:
                if err.retryable and attempt < self.max_attempts:
                    self._emit("tool.retry", tool.name, ctx, call_id=call.id, args_hash=h, tool=tool,
                               error=err, attempt=attempt)
                    self._sleep(self._backoff(attempt, err.retry_after_s))
                    continue
                # A classified error means the handler knows the effect did not happen.
                if key is not None and self.idempotency is not None:
                    self.idempotency.release(key)
                self._emit("tool.failed", tool.name, ctx, call_id=call.id, args_hash=h, tool=tool, error=err,
                           attempt=attempt, latency_ms=(time.monotonic() - t0) * 1000)
                return self._error(call, err, h, attempts=attempt)
            except FuturesTimeout:
                future.cancel()  # no effect if already running; the thread is abandoned
                if tool.idempotent and attempt < self.max_attempts:
                    err = ToolError.transient("timeout", f"timed out after {tool.timeout_s}s")
                    self._emit("tool.retry", tool.name, ctx, call_id=call.id, args_hash=h, tool=tool,
                               error=err, attempt=attempt)
                    self._sleep(self._backoff(attempt, None))
                    continue
                if tool.idempotent:
                    err = ToolError.transient("timeout", f"timed out after {tool.timeout_s}s on every attempt")
                else:
                    if key is not None and self.idempotency is not None:
                        self.idempotency.mark_unknown(key)
                    err = ToolError.fatal("outcome_unknown",
                                          f"timed out after {tool.timeout_s}s; the action may have happened. "
                                          "Do not retry; a human will reconcile.")
                self._emit("tool.failed", tool.name, ctx, call_id=call.id, args_hash=h, tool=tool, error=err,
                           attempt=attempt, idempotency_key=key)
                return self._error(call, err, h, attempts=attempt)
            except Exception as exc:  # a bug in the handler: outcome unknown for writes
                if key is not None and self.idempotency is not None:
                    self.idempotency.mark_unknown(key)
                err = ToolError.fatal("handler_error", f"tool failed unexpectedly ({type(exc).__name__})")
                self._emit("tool.failed", tool.name, ctx, call_id=call.id, args_hash=h, tool=tool, error=err,
                           attempt=attempt, reasons=[repr(exc)[:500]])
                return self._error(call, err, h, attempts=attempt)

            value = _jsonable(value)
            if key is not None and self.idempotency is not None:
                self.idempotency.complete(key, value)
            if approval_id is not None and self.approvals is not None:
                self.approvals.consume(approval_id)
            self._emit("tool.executed", tool.name, ctx, call_id=call.id, args_hash=h, tool=tool, attempt=attempt,
                       latency_ms=(time.monotonic() - start) * 1000, approval_id=approval_id, idempotency_key=key)
            return self._ok(call, tool, value, h, attempts=attempt)

    # ------------------------------------------------------------------- helpers
    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(self.max_delay_s, retry_after)
        delay = min(self.max_delay_s, self.base_delay_s * 2 ** (attempt - 1))
        return random.uniform(0, delay)  # full jitter

    def _ok(self, call: ToolCall, tool: Tool, value: Any, h: str, *, attempts: int, duplicate: bool = False) -> ToolResult:
        shown, truncated = truncate_payload(value, tool.max_result_chars)
        envelope: dict[str, Any] = {"ok": True, "result": shown}
        if duplicate:
            envelope["note"] = "already performed earlier; returning the recorded result"
        return ToolResult(call_id=call.id, tool_name=tool.name, status="ok", content=_dumps(envelope), data=value,
                          args_hash=h, truncated=truncated, duplicate=duplicate, attempts=attempts)

    def _store_unavailable(self, call: ToolCall, ctx: ToolContext, tool: Tool, h: str,
                           exc: Exception) -> ToolResult:
        """Degraded mode: without duplicate suppression a write is not attempted at all.

        Reads never reach this path (they are idempotent), so they keep working.
        The action did not happen, so the error is transient and safe to retry later.
        """
        err = ToolError.transient("idempotency_unavailable",
                                  "duplicate protection is unavailable; the action was not attempted",
                                  retry_after_s=5.0)
        self._emit("tool.failed", tool.name, ctx, call_id=call.id, args_hash=h, tool=tool, error=err,
                   reasons=[type(exc).__name__])
        return self._error(call, err, h)

    def _error(self, call: ToolCall, err: ToolError, h: str | None, *, attempts: int = 0) -> ToolResult:
        return ToolResult(call_id=call.id, tool_name=call.name, status="error",
                          content=_dumps({"ok": False, "error": err.to_dict()}), error=err.to_dict(),
                          args_hash=h, attempts=attempts)

    def _emit(self, event_type: str, tool_name: str, ctx: ToolContext, *, tool: Tool | None = None,
              decision: Decision | None = None, error: ToolError | None = None,
              arguments: dict[str, Any] | None = None, reasons: list[str] | None = None, **fields: Any) -> None:
        event = AuditEvent(
            event_type=event_type, tool_name=tool_name, user_id=ctx.user_id, tenant=ctx.tenant,
            session_id=ctx.session_id, request_id=ctx.request_id,
            tool_fingerprint=tool.schema_fingerprint() if tool else None,
            policy_version=self.policy.version,
            verdict=decision.verdict.value if decision else None,
            rules=decision.rules if decision else [],
            reasons=(reasons or []) + (decision.reasons if decision else []),
            error_category=error.category.value if error else None,
            error_code=error.code if error else None,
            arguments=arguments if self.include_arguments_in_audit else None,
            **{k: v for k, v in fields.items() if v is not None},
        )
        self.audit.emit(event)


__all__ = ["ExecutionContext", "ToolResult", "ToolExecutor", "BoundTool", "truncate_payload", "ErrorCategory", "ToolError"]
