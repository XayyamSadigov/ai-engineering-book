# path: book/projects/toolkit/toolkit/policy.py
"""Who may run which tool with which arguments, decided by code from trusted state.

The model proposes; the policy engine disposes. Every input here comes from the
authenticated request (`ToolContext`) or from configuration, never from model output,
except the validated arguments being judged.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .registry import SideEffect, Tool, args_hash


class ToolContext(BaseModel):
    """The authenticated principal a tool call acts on behalf of.

    Built by your API layer from the session or token, never from the conversation.
    `session_id` scopes default idempotency keys; `attributes` carries extra trusted facts
    (for example the user's team) that argument constraints may consult.
    """

    model_config = ConfigDict(frozen=True)

    user_id: str
    tenant: str
    groups: frozenset[str] = frozenset()
    scopes: frozenset[str] = frozenset()
    session_id: str | None = None
    request_id: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes

    def in_group(self, *groups: str) -> bool:
        return bool(self.groups & set(groups))


class Verdict(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    NEEDS_APPROVAL = "needs_approval"


class Decision(BaseModel):
    """The policy outcome for one concrete proposed call, with reasons a human can read."""

    verdict: Verdict
    tool_name: str
    args_hash: str
    reasons: list[str] = Field(default_factory=list)
    rules: list[str] = Field(default_factory=list)  # machine-readable rule ids that fired
    retry_after_s: float | None = None

    @property
    def allowed(self) -> bool:
        return self.verdict == Verdict.ALLOW

    @property
    def needs_approval(self) -> bool:
        return self.verdict == Verdict.NEEDS_APPROVAL


# A constraint inspects validated args plus trusted context and returns a violation
# message, or None when the call is acceptable.
Constraint = Callable[[BaseModel, ToolContext], str | None]


@dataclass(frozen=True)
class ArgumentRule:
    rule_id: str
    tool_name: str
    check: Constraint
    on_violation: Verdict = Verdict.DENY  # DENY, or NEEDS_APPROVAL to escalate instead


@dataclass(frozen=True)
class RateLimit:
    max_calls: int
    per_seconds: float
    key: str = "user"  # "user" | "tenant" | "global"


class PolicyEngine:
    """Evaluates permission, argument constraints, rate limits, and approval requirements.

    Order matters and is fixed: permission, then argument rules, then rate limit, then
    approval. A call that is denied never consumes rate-limit budget and never creates an
    approval request, so an attacker cannot flood the approval queue with forbidden calls.
    """

    def __init__(
        self,
        *,
        approval_for: Iterable[SideEffect] = (SideEffect.IRREVERSIBLE, SideEffect.EXTERNAL),
        tenant_scoped_tools: Iterable[str] = (),
        clock: Callable[[], float] = time.monotonic,
        version: str = "1",
    ) -> None:
        self.approval_for = frozenset(approval_for)
        self.version = version
        self._rules: dict[str, list[ArgumentRule]] = defaultdict(list)
        self._limits: dict[str, RateLimit] = {}
        self._calls: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._denied_tools: dict[str, set[str]] = defaultdict(set)  # group -> tools
        self._clock = clock
        self._lock = threading.Lock()
        self._tenant_scoped = frozenset(tenant_scoped_tools)

    # ---------------------------------------------------------------- configuration
    def add_rule(self, tool_name: str, check: Constraint, *, rule_id: str,
                 on_violation: Verdict = Verdict.DENY) -> None:
        if on_violation == Verdict.ALLOW:
            raise ValueError("a violated rule cannot allow")
        self._rules[tool_name].append(ArgumentRule(rule_id, tool_name, check, on_violation))

    def set_rate_limit(self, tool_name: str, max_calls: int, per_seconds: float, key: str = "user") -> None:
        if key not in ("user", "tenant", "global"):
            raise ValueError("rate limit key must be user, tenant, or global")
        self._limits[tool_name] = RateLimit(max_calls, per_seconds, key)

    def deny_tool_for_group(self, group: str, tool_name: str) -> None:
        """Explicit deny that overrides scopes, e.g. contractors never send external mail."""
        self._denied_tools[group].add(tool_name)

    # ---------------------------------------------------------------------- queries
    def visible(self, tool: Tool, ctx: ToolContext) -> bool:
        """Discovery filter: hide tools this principal could never be allowed to call."""
        if tool.required_permission and not ctx.has_scope(tool.required_permission):
            return False
        return not any(tool.name in self._denied_tools[g] for g in ctx.groups)

    def evaluate(self, tool: Tool, args: BaseModel, ctx: ToolContext, *, consume: bool = True,
                 check_rate: bool = True) -> Decision:
        """`check_rate=False` is for executing an approved call: its budget was spent at proposal."""
        h = args_hash(tool.name, args.model_dump(mode="json"))

        def decide(verdict: Verdict, reasons: list[str], rules: list[str], retry: float | None = None) -> Decision:
            return Decision(verdict=verdict, tool_name=tool.name, args_hash=h, reasons=reasons,
                            rules=rules, retry_after_s=retry)

        # 1. Permission: scope and explicit group denies.
        if tool.required_permission and not ctx.has_scope(tool.required_permission):
            return decide(Verdict.DENY, [f"missing permission '{tool.required_permission}'"], ["permission"])
        for g in ctx.groups:
            if tool.name in self._denied_tools[g]:
                return decide(Verdict.DENY, [f"tool '{tool.name}' is denied for group '{g}'"], ["group_deny"])

        # 2. Tenant scoping: a tool marked tenant-scoped must target the caller's tenant.
        if tool.name in self._tenant_scoped:
            target = getattr(args, "tenant", None)
            if target is not None and target != ctx.tenant:
                return decide(Verdict.DENY, [f"tenant '{target}' is outside caller tenant '{ctx.tenant}'"],
                              ["tenant_scope"])

        # 3. Argument rules. Any DENY wins; escalations accumulate.
        escalations: list[ArgumentRule] = []
        escalation_reasons: list[str] = []
        for rule in self._rules.get(tool.name, []):
            problem = rule.check(args, ctx)
            if problem is None:
                continue
            if rule.on_violation == Verdict.DENY:
                return decide(Verdict.DENY, [problem], [rule.rule_id])
            escalations.append(rule)
            escalation_reasons.append(problem)

        # 4. Rate limit, only for calls that would otherwise proceed.
        limit = self._limits.get(tool.name)
        if limit is not None and check_rate:
            retry = self._check_rate(tool.name, limit, ctx, consume=consume)
            if retry is not None:
                return decide(Verdict.DENY, [f"rate limit {limit.max_calls}/{limit.per_seconds:g}s exceeded"],
                              ["rate_limit"], retry)

        # 5. Approval: by side-effect class, by tool flag, or by an escalating rule.
        reasons, rules = list(escalation_reasons), [r.rule_id for r in escalations]
        if tool.requires_approval:
            reasons.append(f"tool '{tool.name}' always requires approval")
            rules.append("tool_requires_approval")
        elif tool.side_effect in self.approval_for:
            reasons.append(f"side effect '{tool.side_effect.value}' requires approval")
            rules.append("side_effect_requires_approval")
        if rules:
            return decide(Verdict.NEEDS_APPROVAL, reasons, rules)
        return decide(Verdict.ALLOW, [], [])

    # alias used by framework adapters in Chapter 23
    check = evaluate

    # -------------------------------------------------------------------- internals
    def _check_rate(self, tool_name: str, limit: RateLimit, ctx: ToolContext, *, consume: bool) -> float | None:
        subject = {"user": ctx.user_id, "tenant": ctx.tenant, "global": "*"}[limit.key]
        now = self._clock()
        with self._lock:
            window = self._calls[(tool_name, subject)]
            while window and now - window[0] >= limit.per_seconds:
                window.popleft()
            if len(window) >= limit.max_calls:
                return max(0.0, limit.per_seconds - (now - window[0]))
            if consume:
                window.append(now)
        return None


# Backwards-compatible name used in Chapters 23 and the roadmap.
ToolPolicy = PolicyEngine


# ------------------------------------------------------------------ reusable constraints
def recipient_allowlist(field_name: str, allowed_domains: Iterable[str],
                        allowed_addresses: Iterable[str] = ()) -> Constraint:
    """Deny any address in `field_name` (str or list of str) outside the allowlist."""
    domains = {d.lower().lstrip("@") for d in allowed_domains}
    addresses = {a.lower() for a in allowed_addresses}

    def check(args: BaseModel, ctx: ToolContext) -> str | None:
        value = getattr(args, field_name)
        recipients = [value] if isinstance(value, str) else list(value)
        for r in recipients:
            r = r.strip().lower()
            domain = r.rsplit("@", 1)[-1] if "@" in r else ""
            if r not in addresses and domain not in domains:
                return f"recipient '{r}' is not on the allowlist"
        return None

    return check


def max_value(field_name: str, limit: float) -> Constraint:
    def check(args: BaseModel, ctx: ToolContext) -> str | None:
        v = getattr(args, field_name)
        return f"{field_name}={v} exceeds limit {limit}" if v is not None and v > limit else None

    return check


__all__ = [
    "ToolContext", "Verdict", "Decision", "Constraint", "ArgumentRule", "RateLimit",
    "PolicyEngine", "ToolPolicy", "recipient_allowlist", "max_value",
]
