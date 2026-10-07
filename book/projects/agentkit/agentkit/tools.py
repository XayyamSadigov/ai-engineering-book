# path: book/projects/agentkit/agentkit/tools.py
"""The minimal tool contract the runtime needs, an adapter for richer tool objects,
argument validation, and the policy hook.

Chapter 16 builds the full tool layer (registry, sandbox, idempotency store). agentkit does
not import it: anything with `name`, `spec`, and `execute` is adapted by `adapt_tool`.
"""
from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Literal, Protocol, runtime_checkable

from pydantic import BaseModel

from aie_core.llm.types import ToolSpec

from .errors import ErrorClass


class SideEffect(str, Enum):
    READ = "read"                    # no state change anywhere; safe to retry and to run automatically
    WRITE = "write"                  # reversible change (a draft, a ticket comment)
    IRREVERSIBLE = "irreversible"    # destructive (refund, delete, access change)
    EXTERNAL = "external"            # leaves our boundary (email, webhook, third-party API)


_SIDE_EFFECT_ALIASES = {"reversible_write": "write"}   # Chapter 16's toolkit spelling


def coerce_side_effect(raw: Any) -> SideEffect:
    value = getattr(raw, "value", raw)
    return SideEffect(_SIDE_EFFECT_ALIASES.get(value, value))


_CATEGORY_TO_CLASS = {"not_found": "impossible"}         # toolkit ErrorCategory -> ErrorClass


@dataclass(frozen=True)
class ToolContext:
    """Trusted, harness-provided context. The model never writes any of these fields."""

    run_id: str
    step: int
    call_id: str
    request_id: str
    idempotency_key: str
    principal: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolOutput:
    """What a tool returns. Failures can be returned as values with a machine-readable class."""

    content: str
    ok: bool = True
    data: Any = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    error_class: ErrorClass | None = None
    error: str | None = None

    @classmethod
    def failure(cls, message: str, error_class: ErrorClass = ErrorClass.VALIDATION) -> "ToolOutput":
        return cls(content=f"ERROR ({error_class.value}): {message}", ok=False, error_class=error_class, error=message)


@runtime_checkable
class Tool(Protocol):
    name: str
    side_effect: SideEffect
    requires_approval: bool
    idempotent: bool

    @property
    def spec(self) -> ToolSpec: ...

    def execute(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutput: ...


@dataclass
class FunctionTool:
    """A tool backed by a plain function `fn(**arguments)` or `fn(ctx, **arguments)`."""

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]
    side_effect: SideEffect = SideEffect.READ
    requires_approval: bool = False
    idempotent: bool = True
    pass_context: bool = False

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, parameters=self.parameters)

    def execute(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        result = self.fn(ctx, **arguments) if self.pass_context else self.fn(**arguments)
        return normalize_output(result)


def function_tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
    *,
    side_effect: SideEffect = SideEffect.READ,
    requires_approval: bool = False,
    idempotent: bool | None = None,
    pass_context: bool = False,
) -> Callable[[Callable[..., Any]], FunctionTool]:
    """Decorator form: `@function_tool("search", "...", {...})`."""

    def wrap(fn: Callable[..., Any]) -> FunctionTool:
        return FunctionTool(
            name=name, description=description, parameters=parameters, fn=fn, side_effect=side_effect,
            requires_approval=requires_approval,
            idempotent=(side_effect is SideEffect.READ) if idempotent is None else idempotent,
            pass_context=pass_context,
        )

    return wrap


def normalize_output(result: Any) -> ToolOutput:
    """Accept the shapes real tool layers return and produce a ToolOutput."""
    if isinstance(result, ToolOutput):
        return result
    if isinstance(result, str):
        return ToolOutput(content=result)
    if hasattr(result, "ok") and hasattr(result, "content"):   # duck-typed result objects
        raw_class = getattr(result, "error_class", None)
        err = getattr(result, "error", None)
        if raw_class is None and isinstance(err, dict):          # e.g. toolkit: error={"category": ...}
            raw_class = err.get("category")
        if raw_class is None and getattr(result, "status", None) == "pending_approval":
            raw_class = ErrorClass.PERMISSION
        error_class = None
        if raw_class is not None:
            value = getattr(raw_class, "value", raw_class)
            error_class = ErrorClass(_CATEGORY_TO_CLASS.get(value, value))
        elif not result.ok:
            error_class = ErrorClass.VALIDATION
        if isinstance(err, dict):
            err = err.get("message") or json.dumps(err, default=str)
        content = result.content if isinstance(result.content, str) else json.dumps(result.content, default=str)
        return ToolOutput(content=content, ok=bool(result.ok), data=getattr(result, "data", None),
                          error_class=error_class, error=err)
    if isinstance(result, BaseModel):
        data = result.model_dump(mode="json")
        return ToolOutput(content=json.dumps(data, ensure_ascii=False), data=data)
    if isinstance(result, (dict, list)):
        return ToolOutput(content=json.dumps(result, ensure_ascii=False, default=str), data=result)
    return ToolOutput(content=str(result))


class _AdaptedTool:
    """Wraps a foreign tool object so the runtime sees the `Tool` protocol."""

    def __init__(self, obj: Any, *, key_policy: "KeyPolicy | None" = None) -> None:
        self._obj = obj
        self._key_policy = key_policy
        self.name: str = obj.name
        self.side_effect = coerce_side_effect(getattr(obj, "side_effect", SideEffect.READ))
        self.requires_approval = bool(getattr(obj, "requires_approval", False))
        self.idempotent = bool(getattr(obj, "idempotent", self.side_effect is SideEffect.READ))
        self.approval_by_executor = bool(getattr(obj, "approval_by_executor", False))
        try:
            params = [p for p in inspect.signature(obj.execute).parameters.values()
                      if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
            self._wants_ctx = len(params) >= 2
        except (TypeError, ValueError):
            self._wants_ctx = False

    @property
    def spec(self) -> ToolSpec:
        raw = self._obj.to_tool_spec() if hasattr(self._obj, "to_tool_spec") else self._obj.spec
        if callable(raw) and not isinstance(raw, (ToolSpec, dict)):   # spec() as a method
            raw = raw()
        if isinstance(raw, ToolSpec):
            return raw
        if isinstance(raw, dict):
            return ToolSpec(**raw)
        return ToolSpec.model_validate(raw, from_attributes=True)

    def execute(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        if self._key_policy is not None:
            # An empty key tells toolkit to derive its own content-bound default key.
            ctx = replace(ctx, idempotency_key=self._key_policy(self.name, arguments, ctx) or "")
        result = self._obj.execute(arguments, ctx) if self._wants_ctx else self._obj.execute(arguments)
        return normalize_output(result)


def adapt_tool(obj: Any) -> Tool:
    """Return `obj` if it already satisfies the protocol, else wrap it."""
    if isinstance(obj, FunctionTool | _AdaptedTool | _ExecutorTool):
        return obj  # type: ignore[return-value]
    for attr in ("name", "execute"):
        if not hasattr(obj, attr):
            raise TypeError(f"cannot adapt {obj!r}: missing `{attr}`")
    if not (hasattr(obj, "spec") or hasattr(obj, "to_tool_spec")):
        raise TypeError(f"cannot adapt {obj!r}: missing `spec`")
    return _AdaptedTool(obj)  # type: ignore[return-value]


class _ExecutorTool:
    """One tool of a governed executor (Chapter 16's ToolExecutor, duck-typed). Execution goes
    through the executor, so its policy, validation, idempotency store, sandbox, audit log,
    and approval manager all stay in force; agentkit's idempotency key is passed through."""

    def __init__(self, executor: Any, tool: Any, exec_ctx: Any, *, propagate_approval: bool,
                 key_policy: "KeyPolicy | None" = None) -> None:
        self._executor = executor
        self._key_policy = key_policy or _run_key
        self._tool = tool
        self._exec_ctx = exec_ctx
        self.name: str = tool.name
        self.side_effect = coerce_side_effect(getattr(tool, "side_effect", SideEffect.READ))
        self.idempotent = bool(getattr(tool, "idempotent", self.side_effect is SideEffect.READ))
        self.requires_approval = bool(getattr(tool, "requires_approval", False)) and propagate_approval
        self.approval_by_executor = not propagate_approval   # DefaultPolicy will not gate it a second time

    @property
    def spec(self) -> ToolSpec:
        raw = self._tool.spec
        return raw() if callable(raw) else raw

    def execute(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        from aie_core.llm.types import ToolCall

        call = ToolCall(id=ctx.call_id, name=self.name, arguments=arguments)
        key = self._key_policy(self.name, arguments, ctx)
        return normalize_output(self._executor.execute(call, self._exec_ctx, idempotency_key=key or None))


# --------------------------------------------------------------------------- idempotency key policy
# (tool_name, arguments, agentkit ctx) -> explicit key, or None to let the executor derive one.
KeyPolicy = Callable[[str, dict[str, Any], ToolContext], "str | None"]
IdempotencyMode = Literal["content", "run"] | KeyPolicy


def _run_key(name: str, arguments: dict[str, Any], ctx: ToolContext) -> str | None:
    return ctx.idempotency_key


def _content_key(name: str, arguments: dict[str, Any], ctx: ToolContext) -> str | None:
    return None


def _key_policy(mode: IdempotencyMode) -> KeyPolicy:
    if mode == "content":
        return _content_key
    if mode == "run":
        return _run_key
    if callable(mode):
        return mode
    raise ValueError(f"idempotency must be 'content', 'run', or a callable, got {mode!r}")


def executor_tools(executor: Any, exec_ctx: Any, *, names: set[str] | None = None,
                   propagate_approval: bool = False, idempotency: IdempotencyMode = "content",
                   **filters: Any) -> list[Tool]:
    """Wrap every tool the executor's policy lets `exec_ctx` see.

    Uses `executor.bind(ctx, **filters)` when the executor offers it (Chapter 16's toolkit
    returns BoundTool objects bound to the principal); otherwise wraps registry tools directly.
    By default the executor owns approvals: a gated call returns `pending_approval`, which the
    model reads as a non-ok observation, and DefaultPolicy does not gate it a second time.
    Set `propagate_approval=True` only for an executor without its own approval manager, so
    agentkit pauses the run instead.

    `idempotency` decides which key the executor deduplicates on:
    - "content" (default): pass no key, so the executor derives its content-bound default
      (toolkit: tool, tenant, user, session, normalized-argument hash). The same action proposed in
      two runs of one session executes once, and so does a re-execution after a crash.
    - "run": pass agentkit's `run_id:request_id`. Duplicates are suppressed only within one
      run (crash and resume); a new run repeats the action. Use it when repeating an action
      across runs is intended.
    - a callable `(tool_name, arguments, ctx) -> key | None` for anything else, for example a
      business key such as `f"ticket:{arguments['incident_id']}"`; None falls back to content.
    """
    policy_fn = _key_policy(idempotency)
    if hasattr(executor, "bind"):
        bound = executor.bind(exec_ctx, names=names, **filters) if names is not None \
            else executor.bind(exec_ctx, **filters)
        wrapped: list[Tool] = []
        for obj in bound:
            tool = _AdaptedTool(obj, key_policy=policy_fn)
            tool.side_effect = coerce_side_effect(getattr(obj, "side_effect_class", tool.side_effect))
            tool.requires_approval = tool.requires_approval and propagate_approval
            tool.approval_by_executor = not propagate_approval  # type: ignore[attr-defined]
            wrapped.append(tool)  # type: ignore[arg-type]
        return wrapped
    registry, policy = executor.registry, getattr(executor, "policy", None)
    candidates = registry.select(exec_ctx, visible=policy.visible) if policy is not None else registry.all()
    return [_ExecutorTool(executor, t, exec_ctx, propagate_approval=propagate_approval,  # type: ignore[misc]
                          key_policy=policy_fn)
            for t in candidates if names is None or t.name in names]


# --------------------------------------------------------------------------- argument validation
_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,), "integer": (int,), "number": (int, float), "boolean": (bool,),
    "array": (list,), "object": (dict,), "null": (type(None),),
}


def validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> list[str]:
    """A deliberately small JSON Schema subset: required, additionalProperties, type, enum,
    minimum/maximum, minLength/maxLength, and array item types. Returns readable errors."""
    errors: list[str] = []
    if not isinstance(arguments, dict):
        return ["arguments must be a JSON object"]
    props: dict[str, Any] = schema.get("properties", {})
    for name in schema.get("required", []):
        if name not in arguments:
            errors.append(f"missing required argument '{name}'")
    if schema.get("additionalProperties") is False:
        for name in arguments:
            if name not in props:
                errors.append(f"unexpected argument '{name}'")
    for name, value in arguments.items():
        if name in props:
            errors.extend(_check_value(name, props[name], value))
    return errors


def _check_value(path: str, spec: dict[str, Any], value: Any) -> list[str]:
    errs: list[str] = []
    expected = spec.get("type")
    if expected:
        allowed = expected if isinstance(expected, list) else [expected]
        types = tuple(t for a in allowed for t in _JSON_TYPES.get(a, (object,)))
        bad_bool = isinstance(value, bool) and "boolean" not in allowed
        if not isinstance(value, types) or bad_bool:
            return [f"'{path}' must be {' or '.join(allowed)}, got {type(value).__name__}"]
    if "enum" in spec and value not in spec["enum"]:
        errs.append(f"'{path}' must be one of {spec['enum']}, got {value!r}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in spec and value < spec["minimum"]:
            errs.append(f"'{path}' must be >= {spec['minimum']}")
        if "maximum" in spec and value > spec["maximum"]:
            errs.append(f"'{path}' must be <= {spec['maximum']}")
    if isinstance(value, str):
        if "minLength" in spec and len(value) < spec["minLength"]:
            errs.append(f"'{path}' must have at least {spec['minLength']} characters")
        if "maxLength" in spec and len(value) > spec["maxLength"]:
            errs.append(f"'{path}' must have at most {spec['maxLength']} characters")
    if isinstance(value, list) and isinstance(spec.get("items"), dict):
        for i, item in enumerate(value):
            errs.extend(_check_value(f"{path}[{i}]", spec["items"], item))
    return errs


# --------------------------------------------------------------------------- policy
class PolicyDecision(BaseModel):
    allowed: bool
    requires_approval: bool = False
    reason: str = ""


class ToolPolicy(Protocol):
    """Decides, outside the model, what the model may see and do."""

    def visible(self, tool: Tool, principal: dict[str, Any]) -> bool: ...

    def check(self, tool: Tool, arguments: dict[str, Any], principal: dict[str, Any]) -> PolicyDecision: ...


class DefaultPolicy:
    """Allow-list plus approval rules.

    - `allowed`: tool names this run may use (None means every registered tool).
    - `require_approval`: extra tool names that need a human, on top of tools that declare it.
    - `irreversible_needs_approval`: any IRREVERSIBLE tool needs approval (default True).
    - `rules`: extra callables `(tool, arguments, principal) -> PolicyDecision | None`; the first
      decision that denies or requires approval wins. A rule cannot waive the approval checks
      below. Use them for argument-level checks such as tenant scope.
    """

    def __init__(
        self,
        allowed: set[str] | None = None,
        require_approval: set[str] | None = None,
        *,
        irreversible_needs_approval: bool = True,
        rules: list[Callable[[Tool, dict[str, Any], dict[str, Any]], PolicyDecision | None]] | None = None,
    ) -> None:
        self.allowed = allowed
        self.require_approval = require_approval or set()
        self.irreversible_needs_approval = irreversible_needs_approval
        self.rules = rules or []

    def visible(self, tool: Tool, principal: dict[str, Any]) -> bool:
        return self.allowed is None or tool.name in self.allowed

    def check(self, tool: Tool, arguments: dict[str, Any], principal: dict[str, Any]) -> PolicyDecision:
        if not self.visible(tool, principal):
            return PolicyDecision(allowed=False, reason=f"tool '{tool.name}' is not allowed for this run")
        for rule in self.rules:
            decision = rule(tool, arguments, principal)
            if decision is not None and (not decision.allowed or decision.requires_approval):
                return decision
        if getattr(tool, "approval_by_executor", False):
            return PolicyDecision(allowed=True, reason="approval delegated to the tool executor")
        needs = (
            tool.requires_approval
            or tool.name in self.require_approval
            or (self.irreversible_needs_approval
                and tool.side_effect in (SideEffect.IRREVERSIBLE, SideEffect.EXTERNAL))
        )
        return PolicyDecision(allowed=True, requires_approval=needs, reason="approval required" if needs else "")


__all__ = [
    "SideEffect", "coerce_side_effect", "executor_tools", "IdempotencyMode", "ToolContext", "ToolOutput", "Tool", "FunctionTool", "function_tool", "normalize_output",
    "adapt_tool", "validate_arguments", "PolicyDecision", "ToolPolicy", "DefaultPolicy",
]
