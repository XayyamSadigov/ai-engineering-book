# path: book/projects/examples/ch20/patterns/script.py
"""Scripted models for offline tests: a FakeLLM handler that dispatches on the role tag.

A real model reads the whole prompt; a scripted one only needs to read enough to behave
plausibly for the role it is playing. Keeping each role's behavior in its own small function
makes a multi-agent test as readable as a single-agent one.
"""
from __future__ import annotations

import json
import re
from itertools import count
from typing import Any, Callable

from aie_core.llm.types import CompletionRequest, Role, ToolCall

from .common import role_of
from .fixtures import METRICS, STATUS

Handler = Callable[[CompletionRequest], Any]
_ids = count()


def tc(name: str, **arguments: Any) -> list[ToolCall]:
    return [ToolCall(id=f"call_{next(_ids)}", name=name, arguments=arguments)]


def goal_of(req: CompletionRequest) -> str:
    return next((m.text for m in req.messages if m.role is Role.USER), "")


def tool_texts(req: CompletionRequest) -> list[str]:
    return [m.text for m in req.messages if m.role is Role.TOOL]


def harness_notes(req: CompletionRequest) -> list[str]:
    return [m.text for m in req.messages if m.role is Role.USER and m.text.startswith("[harness:")]


def citations(text: str) -> list[str]:
    return re.findall(r"\[([A-Za-z0-9][A-Za-z0-9_.:/-]*)\]", text)


def default_arguments(tool: str, goal: str) -> dict[str, Any]:
    """Derive plausible arguments for a fixture tool from the task text."""
    low = goal.lower()
    if tool == "get_service_status":
        return {"service": next((s for s in STATUS if s in low), "trackline")}
    if tool == "query_metrics":
        return {"metric": next((m for m in METRICS if m.split(".")[0] in low), "trackline.p95_ms")}
    words = [w for w in re.findall(r"[a-z]+", low) if len(w) > 4][:6]
    return {"query": " ".join(words) or "incident"}


def one_tool_then_report(req: CompletionRequest) -> Any:
    """Executor-style behavior: call the first available tool once, then report what it returned."""
    seen = tool_texts(req)
    if not seen:
        name = req.tools[0].name if req.tools else ""
        return tc(name, **default_arguments(name, goal_of(req)))
    cited = sorted(set(c for t in seen for c in citations(t)))
    first = seen[-1].splitlines()[0][:200]
    return f"Finding: {first} " + " ".join(f"[{c}]" for c in cited)


class Script:
    """FakeLLM(handler=Script(planner=..., executor=...)). Lookup: exact role, then the prefix before ':'."""

    def __init__(self, default: Handler | None = None, **handlers: Handler) -> None:
        self.handlers = {k.replace("__", ":"): v for k, v in handlers.items()}
        self.default = default
        self.seen_roles: list[str] = []

    def __call__(self, req: CompletionRequest) -> Any:
        role = role_of(req)
        self.seen_roles.append(role)
        handler = self.handlers.get(role) or self.handlers.get(role.split(":")[0]) or self.default
        if handler is None:
            raise KeyError(f"no scripted behavior for role {role!r}")
        out = handler(req)
        return json.dumps(out) if isinstance(out, dict) else out


__all__ = ["tc", "goal_of", "tool_texts", "harness_notes", "citations", "default_arguments", "one_tool_then_report",
           "Script"]
