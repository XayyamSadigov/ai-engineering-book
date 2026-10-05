# path: book/projects/guardrails/guardrails/tools.py
"""TOOL-stage guardrails: the last check between a model's proposed call and an effect.

Chapter 16 owns the full tool layer (registry, `PolicyEngine`, idempotency, sandbox, audit).
This module provides the guardrail-pipeline view of it, so tool decisions show up in the same
verdict stream and traces as input and output checks:

* an allowlist of tools for the current task (least agency),
* argument constraints (recipient domains, lengths, enumerations, patterns),
* outbound-content scanning: arguments to tools that leave the system must not carry secrets,
  canaries or (optionally) PII,
* approval bound to the exact arguments via `approval_token`,
* a per-request call budget (runaway loops, denial of wallet),
* an optional delegate `authorize(call, ctx)` that forwards to a real policy engine such as
  Chapter 16's `toolkit.policy.PolicyEngine`.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol

from aie_core.llm.types import ToolCall

from .pii import TOKEN_RE, RehydratePolicy, detect_pii
from .pipeline import (BaseCheck, FailMode, Finding, GuardContext, GuardrailPipeline, PipelineResult, Stage,
                       Subject, Verdict)
from .secrets import detect_secrets


def approval_token(call: ToolCall) -> str:
    """Hash of tool name plus canonical arguments. An approval for one set of arguments cannot be
    replayed for another (a changed recipient or body produces a different token)."""
    canonical = json.dumps({"name": call.name, "arguments": call.arguments}, sort_keys=True,
                           separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ArgConstraint(Protocol):
    def __call__(self, value: Any, ctx: GuardContext) -> str | None: ...


def recipient_domains(allowed: Iterable[str]) -> ArgConstraint:
    allowed_set = {a.lower() for a in allowed}

    def check(value: Any, ctx: GuardContext) -> str | None:
        recipients = value if isinstance(value, list) else [value]
        for r in recipients:
            if not isinstance(r, str) or "@" not in r:
                return "recipient is not an email address"
            domain = r.rsplit("@", 1)[1].lower().rstrip(".")
            if domain not in allowed_set:
                return f"recipient domain '{domain}' not allowlisted"
        return None

    return check


def max_length(n: int) -> ArgConstraint:
    def check(value: Any, ctx: GuardContext) -> str | None:
        return f"argument longer than {n} chars" if isinstance(value, str) and len(value) > n else None

    return check


def one_of(*values: Any) -> ArgConstraint:
    def check(value: Any, ctx: GuardContext) -> str | None:
        return None if value in values else "value not in allowed set"

    return check


def matches(pattern: str) -> ArgConstraint:
    rx = re.compile(pattern)

    def check(value: Any, ctx: GuardContext) -> str | None:
        return None if isinstance(value, str) and rx.fullmatch(value) else f"value does not match {pattern}"

    return check


@dataclass(frozen=True)
class ToolRule:
    name: str
    outbound: bool = False                     # data leaves the system (email, HTTP, webhook)
    requires_approval: bool = False
    constraints: dict[str, ArgConstraint] = field(default_factory=dict)
    required_args: tuple[str, ...] = ()
    # Arguments whose PII tokens are replaced by real values at the tool boundary, before the checks
    # below run and before execution (see `guard_tool_call`). The model only ever sees tokens.
    rehydrate_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolDecision:
    allowed: bool
    needs_approval: bool = False
    reason: str = ""


Authorizer = Callable[[ToolCall, GuardContext], ToolDecision]


class ToolPolicyCheck(BaseCheck):
    """Fail-closed by construction: an unknown tool, a missing argument, a constraint that raises,
    or a delegate that raises all end in BLOCK."""

    name = "tool_policy"
    stages = frozenset({Stage.TOOL})
    fail_mode = FailMode.CLOSED

    def __init__(self, rules: Iterable[ToolRule], *, max_calls_per_request: int = 8,
                 canaries: Iterable[str] = (), block_pii_outbound: bool = True,
                 authorize: Authorizer | None = None) -> None:
        self.rules = {r.name: r for r in rules}
        self.max_calls = max_calls_per_request
        self.canaries = tuple(canaries)
        self.block_pii_outbound = block_pii_outbound
        self.authorize = authorize

    def evaluate(self, subject: Subject, ctx: GuardContext) -> Verdict:
        call = subject.tool_call
        if call is None:
            return Verdict.block("no tool call in subject")
        rule = self.rules.get(call.name)
        if rule is None:
            return Verdict.block(f"tool '{call.name}' not allowed for this task")

        # Budget is consumed only by calls that are otherwise permitted, see the end of the method.
        used = int(ctx.state.get("tool_calls", 0))
        if used >= self.max_calls:
            return Verdict.block(f"tool call budget exhausted ({self.max_calls})")

        for arg in rule.required_args:
            if arg not in call.arguments:
                return Verdict.block(f"missing required argument '{arg}'")
        for arg, constraint in rule.constraints.items():
            if arg in call.arguments:
                problem = constraint(call.arguments[arg], ctx)
                if problem:
                    return Verdict.block(f"{call.name}.{arg}: {problem}", findings=[Finding("constraint", detail=arg)])

        if rule.outbound:
            blob = json.dumps(call.arguments, ensure_ascii=False)
            if any(c in blob for c in self.canaries):
                return Verdict.block("canary marker in outbound arguments", findings=[Finding("canary")])
            secrets_found = detect_secrets(blob)
            if secrets_found:
                return Verdict.block("possible secret in outbound arguments",
                                     findings=[Finding(s.kind) for s in secrets_found])
            if self.block_pii_outbound:
                pii = [p for p in detect_pii(blob) if p.kind != "email" or p.value not in _recipients(call)]
                if pii:
                    return Verdict.block("PII in outbound arguments", findings=[Finding(p.kind) for p in pii])

        needs_approval = rule.requires_approval
        if self.authorize is not None:
            decision = self.authorize(call, ctx)
            if not decision.allowed and not decision.needs_approval:
                return Verdict.block(f"policy engine denied: {decision.reason}")
            needs_approval = needs_approval or decision.needs_approval

        token = approval_token(call)
        if needs_approval and token not in ctx.approvals:
            return Verdict.block("approval required for these exact arguments", approval_token=token,
                                 needs_approval=True)

        ctx.state["tool_calls"] = used + 1
        return Verdict.allow("tool call permitted", approval_token=token)


def _recipients(call: ToolCall) -> set[str]:
    out: set[str] = set()
    for key in ("to", "cc", "bcc", "recipient", "recipients"):
        v = call.arguments.get(key)
        if isinstance(v, str):
            out.add(v)
        elif isinstance(v, list):
            out.update(x for x in v if isinstance(x, str))
    return out


# --------------------------------------------------------------------------- PII tokens at the tool boundary
# The input guardrail tokenizes personal data (`<PII:email:3f2a9c1b07>`) so the model provider never
# sees it. A tool, however, needs the real value: a token is not an e-mail address, so it fails the
# tool's own schema pattern and the recipient-domain constraint, and every legitimate send is blocked.
# The fix keeps both properties: the model works on tokens, and the tool boundary swaps tokens back
# for values, per argument and per PII kind, under a policy, *before* the guardrail checks and the
# approval token are computed. Checks therefore judge, and humans approve, what will really execute.


def rehydrate_kinds(*kinds: str) -> RehydratePolicy:
    """Policy that allows re-hydration of the given PII kinds only (for example `"email"`)."""
    allowed = frozenset(kinds)
    return lambda ctx, kind: kind in allowed


def _rehydrate_value(value: Any, ctx: GuardContext, policy: RehydratePolicy) -> Any:
    if isinstance(value, str):
        return ctx.vault.rehydrate(value, ctx, policy) if ctx.vault is not None else value
    if isinstance(value, list):
        return [_rehydrate_value(v, ctx, policy) for v in value]
    if isinstance(value, dict):
        return {k: _rehydrate_value(v, ctx, policy) for k, v in value.items()}
    return value


def rehydrate_arguments(call: ToolCall, ctx: GuardContext, policy: RehydratePolicy,
                        args: Iterable[str] | None = None) -> ToolCall:
    """A copy of `call` with PII tokens in `args` (all arguments when None) replaced by vault values.

    Only tokens from this request's vault, for the caller's tenant, and of kinds the policy allows are
    replaced; invented or foreign tokens stay tokens and then fail validation, which fails closed.
    """
    if ctx.vault is None:
        return call
    names = set(call.arguments) if args is None else set(args)
    new_args = {k: (_rehydrate_value(v, ctx, policy) if k in names else v) for k, v in call.arguments.items()}
    return call.model_copy(update={"arguments": new_args})


def guard_tool_call(pipeline: GuardrailPipeline, call: ToolCall, ctx: GuardContext,
                    policy: RehydratePolicy | None = None) -> tuple[ToolCall, PipelineResult]:
    """The tool boundary: re-hydrate the rule's `rehydrate_args`, then run the TOOL stage.

    Returns the call to execute (re-hydrated) and the pipeline result for it. Execute the returned
    call, never the model's original, and only when `result.allowed`.
    """
    policy = policy or rehydrate_kinds("email")
    names: set[str] = set()
    for check in pipeline.checks_for(Stage.TOOL):
        if isinstance(check, ToolPolicyCheck) and (rule := check.rules.get(call.name)) is not None:
            names.update(rule.rehydrate_args)
    prepared = rehydrate_arguments(call, ctx, policy, names) if names else call
    return prepared, pipeline.check_tool(prepared, ctx)


def token_tolerant_schema(parameters: dict[str, Any], kinds: Iterable[str] = ("email",)) -> dict[str, Any]:
    """Model-facing copy of a tool's JSON Schema whose string patterns also accept PII tokens.

    Give this schema to the model (so strict structured-output modes let it pass a token), and keep
    validating the re-hydrated arguments against the original schema inside the tool.
    """
    token = "<PII:(?:" + "|".join(sorted(set(kinds))) + "):[0-9a-f]{10}>"
    out = json.loads(json.dumps(parameters))

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            pattern = node.get("pattern")
            if node.get("type") == "string" and isinstance(pattern, str):
                inner = pattern.removeprefix("^").removesuffix("$")
                node["pattern"] = f"^(?:{token}|(?:{inner}))$"
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(out)
    return out


def contains_pii_token(value: Any) -> bool:
    return bool(TOKEN_RE.search(json.dumps(value, ensure_ascii=False)))


__all__ = [
    "approval_token", "ArgConstraint", "recipient_domains", "max_length", "one_of", "matches", "ToolRule",
    "ToolDecision", "Authorizer", "ToolPolicyCheck", "rehydrate_kinds", "rehydrate_arguments", "guard_tool_call",
    "token_tolerant_schema", "contains_pii_token",
]
