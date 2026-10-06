# path: book/projects/toolkit/toolkit/registry.py
"""Tool definitions and the registry that decides which tools a model may see.

A `Tool` is the contract between three parties: the model (which reads the name,
description, and JSON Schema), the policy engine (which reads the side-effect class,
required permission, and approval flag), and the executor (which reads the timeout,
idempotency behavior, and result limit). Keeping all of it on one frozen object means a
tool cannot be registered without answering every question the harness will ask.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from aie_core.llm.types import ToolSpec
from pydantic import BaseModel

if TYPE_CHECKING:  # pragma: no cover
    from .executor import ExecutionContext
    from .policy import ToolContext


class SideEffect(str, Enum):
    """What happens in the world when the tool runs. Ordered from least to most risky."""

    READ = "read"                          # no state change anywhere
    REVERSIBLE_WRITE = "reversible_write"  # changes our state; can be undone (draft, ticket)
    IRREVERSIBLE = "irreversible"          # cannot be undone (delete, payment, access change)
    EXTERNAL = "external"                  # leaves our boundary (email, webhook, third-party API)


Handler = Callable[[Any, "ExecutionContext"], Any]


@dataclass(frozen=True)
class Tool:
    """One callable capability plus everything the harness needs to govern it.

    `idempotent=True` means running the handler twice with the same arguments has the
    same effect as running it once (reads, upserts). Non-idempotent tools get duplicate
    suppression through an `IdempotencyStore` and are never blindly retried on timeout.
    """

    name: str
    description: str
    args_model: type[BaseModel]
    handler: Handler
    side_effect: SideEffect = SideEffect.READ
    required_permission: str | None = None
    timeout_s: float = 10.0
    idempotent: bool = True
    requires_approval: bool = False
    max_result_chars: int = 4000
    tags: frozenset[str] = field(default_factory=frozenset)
    version: str = "1"
    # Optional: given args, context, and idempotency key, report whether a previous
    # attempt with an unknown outcome actually took effect (returns the result or None).
    reconcile: Callable[[Any, "ExecutionContext"], Any | None] | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.name):
            raise ValueError(f"tool name must be [A-Za-z0-9_-]{{1,64}}: {self.name!r}")
        if not self.description.strip():
            raise ValueError(f"tool {self.name!r} needs a description; the model selects tools by it")
        if self.idempotent and self.reconcile is not None:
            raise ValueError("reconcile only makes sense for non-idempotent tools")

    @property
    def mutates(self) -> bool:
        return self.side_effect != SideEffect.READ

    def json_schema(self) -> dict[str, Any]:
        """The args model as a provider-friendly JSON Schema object."""
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        # Closed objects: unknown keys are a model mistake we want reported, not ignored.
        schema.setdefault("additionalProperties", False)
        return schema

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, parameters=self.json_schema())

    def schema_fingerprint(self) -> str:
        """Hash of what the model sees. Log it: a description change is a behavior change."""
        payload = canonical_json({"name": self.name, "description": self.description,
                                  "parameters": self.json_schema(), "version": self.version})
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def canonical_json(value: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, UTF-8 preserved."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def args_hash(tool_name: str, arguments: dict[str, Any]) -> str:
    """Identity of a concrete proposed action. Approvals and idempotency keys bind to this.

    Callers must hash the *validated, normalized* arguments (``model_dump(mode="json")``),
    so that two spellings pydantic normalizes to the same value hash identically.
    """
    return hashlib.sha256(canonical_json({"tool": tool_name, "args": arguments}).encode()).hexdigest()


class ToolRegistry:
    """Holds tools and answers 'which tools may this user see for this task?'."""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools:
            self.register(t)

    # ------------------------------------------------------------------ registration
    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool
        return tool

    def tool(
        self,
        args_model: type[BaseModel],
        *,
        name: str | None = None,
        description: str | None = None,
        **options: Any,
    ) -> Callable[[Handler], Handler]:
        """Decorator form. The function docstring becomes the description if none is given."""

        def decorate(fn: Handler) -> Handler:
            desc = description or (fn.__doc__ or "").strip()
            tags = frozenset(options.pop("tags", ()))
            self.register(Tool(name=name or fn.__name__, description=desc, args_model=args_model,
                               handler=fn, tags=tags, **options))
            return fn

        return decorate

    # ----------------------------------------------------------------------- lookup
    def get(self, name: str) -> Tool:
        return self._tools[name]

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def all(self) -> list[Tool]:
        return [self._tools[n] for n in self.names()]

    # -------------------------------------------------------------------- filtering
    def select(
        self,
        ctx: "ToolContext | None" = None,
        *,
        names: Iterable[str] | None = None,
        task_tags: Iterable[str] | None = None,
        visible: Callable[[Tool, "ToolContext"], bool] | None = None,
    ) -> list[Tool]:
        """Tools for one user and one task.

        - ``names``: explicit allowlist for the task (wins over tags).
        - ``task_tags``: keep tools sharing at least one tag with the task.
        - ``ctx``: drop tools whose ``required_permission`` the user lacks.
        - ``visible``: extra predicate, usually ``PolicyEngine.visible``.

        Discovery is not authorization: the executor re-checks everything at call time.
        Filtering here only shrinks the menu so the model chooses better and an injected
        instruction has fewer tools to aim at.
        """
        tools = self.all()
        if names is not None:
            wanted = set(names)
            tools = [t for t in tools if t.name in wanted]
        elif task_tags is not None:
            tags = set(task_tags)
            tools = [t for t in tools if t.tags & tags]
        if ctx is not None:
            tools = [t for t in tools if t.required_permission is None or ctx.has_scope(t.required_permission)]
            if visible is not None:
                tools = [t for t in tools if visible(t, ctx)]
        return tools

    def specs(self, ctx: "ToolContext | None" = None, **filters: Any) -> list[ToolSpec]:
        return [t.spec() for t in self.select(ctx, **filters)]


__all__ = ["SideEffect", "Tool", "Handler", "ToolRegistry", "canonical_json", "args_hash"]
