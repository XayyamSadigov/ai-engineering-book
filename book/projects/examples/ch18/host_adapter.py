# path: book/projects/examples/ch18/host_adapter.py
"""Host side: turn tools discovered over MCP into aie_core ToolSpecs, and route the
model's tool calls back to the right server, with authorization that does not depend
on what the server advertised.

Three separate questions, three separate mechanisms:

  1. Is this tool what we reviewed?      fingerprint of (name, description, inputSchema)
                                          compared with a pinned lockfile -> else quarantine
  2. May this principal use this tool?   ToolGrant in the lockfile: required groups
  3. Are these arguments acceptable?     validation against the pinned schema, in code

Discovery answers none of them. A server can advertise anything; only tools that pass
(1) are offered, only tools that pass (2) for the current principal are offered to that
principal, and every call is re-checked at execution time because the model can name
tools it was never offered.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from aie_core import CompletionRequest, Completion, Message, Role, ToolCall, ToolSpec
from aie_core.llm.client import LLMClient
from aie_core.observability import NoopTracer, Tracer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from mcp_client import McpClientError, StdioMcpClient  # noqa: E402
from schema_check import validate_arguments  # noqa: E402

SEP = "__"  # exposed name = "<server_id>__<tool>", so two servers can both have "search"


# --------------------------------------------------------------------------- identity
class Principal(BaseModel):
    """The end user on whose behalf the model acts. Authorization is about this, not the agent."""

    user_id: str
    tenant: Literal["retail", "logistics"]
    groups: frozenset[str] = frozenset({"all"})


# --------------------------------------------------------------------------- lockfile
def tool_fingerprint(tool: dict[str, Any]) -> str:
    """Hash over everything the model will read or rely on. A one-character change to the
    description, which is prompt text, changes the hash."""
    canonical = json.dumps(
        {"name": tool.get("name"), "description": tool.get("description"), "inputSchema": tool.get("inputSchema")},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ToolGrant(BaseModel):
    server_id: str
    tool: str
    fingerprint: str
    required_groups: frozenset[str] = frozenset({"all"})  # any one group grants access
    side_effect: Literal["read", "write"] = "read"
    max_result_chars: int = 4000


class ToolLock(BaseModel):
    """Reviewed, versioned configuration: the host's source of truth for tools.
    Generated from a trusted run, reviewed like a dependency lockfile, then committed."""

    grants: list[ToolGrant] = Field(default_factory=list)

    def get(self, server_id: str, tool: str) -> ToolGrant | None:
        return next((g for g in self.grants if g.server_id == server_id and g.tool == tool), None)

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ToolLock":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


def pin_reviewed_tools(
    server_id: str, discovered: Iterable[dict[str, Any]], approvals: dict[str, frozenset[str]]
) -> ToolLock:
    """Create grants for the tools a reviewer approved (tool name -> required groups).
    Anything discovered but not approved gets no grant and will never be offered."""
    grants = [
        ToolGrant(server_id=server_id, tool=t["name"], fingerprint=tool_fingerprint(t), required_groups=approvals[t["name"]])
        for t in discovered
        if t["name"] in approvals
    ]
    return ToolLock(grants=grants)


# --------------------------------------------------------------------------- decisions
class DiscoveryReport(BaseModel):
    server_id: str
    verified: list[str] = Field(default_factory=list)
    quarantined: dict[str, str] = Field(default_factory=dict)  # tool -> reason
    ignored: list[str] = Field(default_factory=list)  # discovered, no grant


class Decision(BaseModel):
    allowed: bool
    reason: str
    server_id: str | None = None
    tool: str | None = None


# --------------------------------------------------------------------------- adapter
class McpToolAdapter:
    """Binds one connected server to the host's lockfile."""

    def __init__(self, server_id: str, client: StdioMcpClient, lock: ToolLock, tracer: Tracer | None = None) -> None:
        self.server_id = server_id
        self.client = client
        self.lock = lock
        self.tracer = tracer or NoopTracer()
        self._verified: dict[str, dict[str, Any]] = {}  # tool name -> descriptor that matched its pin

    def discover(self) -> DiscoveryReport:
        """List tools and verify each against the lockfile. Call again on reconnect or on
        a list-changed notification; a tool that changes after review is quarantined."""
        report = DiscoveryReport(server_id=self.server_id)
        with self.tracer.span("mcp.discover", server=self.server_id) as span:
            self._verified = {}
            for tool in self.client.list_tools():
                name = tool.get("name", "")
                grant = self.lock.get(self.server_id, name)
                if grant is None:
                    report.ignored.append(name)
                elif tool_fingerprint(tool) != grant.fingerprint:
                    report.quarantined[name] = "fingerprint mismatch: description or schema changed since review"
                else:
                    self._verified[name] = tool
                    report.verified.append(name)
            span.set_attribute("verified", len(report.verified))
            span.set_attribute("quarantined", sorted(report.quarantined))
            span.set_attribute("ignored", sorted(report.ignored))
        return report

    def tool_specs(self, principal: Principal) -> list[ToolSpec]:
        """Least-privilege exposure: verified tools this principal is granted, nothing else."""
        specs = []
        for name, tool in self._verified.items():
            grant = self.lock.get(self.server_id, name)
            if grant and principal.groups & grant.required_groups:
                specs.append(
                    ToolSpec(name=f"{self.server_id}{SEP}{name}", description=tool["description"], parameters=tool["inputSchema"])
                )
        return specs

    def authorize(self, principal: Principal, tool: str, arguments: dict[str, Any]) -> Decision:
        """Runs at call time, from the lockfile and the principal. Being listed by the
        server, or offered to the model earlier, grants nothing."""
        base = {"server_id": self.server_id, "tool": tool}
        grant = self.lock.get(self.server_id, tool)
        if grant is None:
            return Decision(allowed=False, reason="no grant for this tool", **base)
        if not principal.groups & grant.required_groups:
            return Decision(allowed=False, reason="principal lacks a required group", **base)
        if tool not in self._verified:
            return Decision(allowed=False, reason="tool not verified in this session (quarantined or not discovered)", **base)
        problems = validate_arguments(self._verified[tool]["inputSchema"], arguments)
        if problems:
            return Decision(allowed=False, reason="invalid arguments: " + "; ".join(problems), **base)
        return Decision(allowed=True, reason="ok", **base)

    def execute(self, tool: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        """Call the server and flatten the result to text. Returns (text, is_error)."""
        grant = self.lock.get(self.server_id, tool)
        limit = grant.max_result_chars if grant else 4000
        try:
            result = self.client.call_tool(tool, arguments)
        except McpClientError as exc:
            return f"tool unavailable: {exc.message}", True
        text = "\n".join(c.get("text", "") for c in result.get("content", []) if c.get("type") == "text")
        if len(text) > limit:
            text = text[:limit] + f"\n[truncated {len(text) - limit} chars]"
        return text, bool(result.get("isError"))


# --------------------------------------------------------------------------- host
class McpHost:
    """Owns several adapters, builds the tool list per principal, routes calls, and runs a
    small bounded tool loop. Chapter 19 replaces run_turn with the full AgentRuntime."""

    def __init__(self, adapters: list[McpToolAdapter], tracer: Tracer | None = None) -> None:
        self.adapters = {a.server_id: a for a in adapters}
        self.tracer = tracer or NoopTracer()
        self.decisions: list[Decision] = []  # audit trail for tests and logs

    def tool_specs(self, principal: Principal) -> list[ToolSpec]:
        return [spec for a in self.adapters.values() for spec in a.tool_specs(principal)]

    def route(self, principal: Principal, call: ToolCall) -> Message:
        server_id, _, tool = call.name.partition(SEP)
        adapter = self.adapters.get(server_id)
        with self.tracer.span("mcp.tool_call", exposed_name=call.name, user=principal.user_id) as span:
            if adapter is None or not tool:
                decision = Decision(allowed=False, reason="unknown tool name", tool=call.name)
            else:
                decision = adapter.authorize(principal, tool, call.arguments)
            self.decisions.append(decision)
            span.set_attribute("decision", decision.reason)
            if not decision.allowed:
                # Denials are not retried and not sent anywhere; the model sees a plain refusal.
                return Message.tool(call.id, f"DENIED: {decision.reason}")
            assert adapter is not None
            started = time.perf_counter()
            text, is_error = adapter.execute(tool, call.arguments)
            span.set_attribute("latency_ms", round((time.perf_counter() - started) * 1000, 2))
            span.set_attribute("is_error", is_error)
        # Tool output is data from another system. Mark it so downstream guardrails
        # (Chapter 27) and the system prompt can treat it as untrusted.
        wrapped = f'<tool_result server="{server_id}" tool="{tool}" error="{str(is_error).lower()}">\n{text}\n</tool_result>'
        return Message.tool(call.id, wrapped)

    def run_turn(self, llm: LLMClient, principal: Principal, messages: list[Message], max_tool_rounds: int = 3) -> Completion:
        msgs = list(messages)
        tools = self.tool_specs(principal)
        for _ in range(max_tool_rounds + 1):
            completion = llm.complete(CompletionRequest(messages=msgs, tools=tools or None))
            if not completion.tool_calls:
                return completion
            msgs.append(Message(role=Role.ASSISTANT, content=completion.text, tool_calls=completion.tool_calls))
            for call in completion.tool_calls:
                msgs.append(self.route(principal, call))
        return completion  # tool budget exhausted; the caller decides how to surface it


__all__ = [
    "Principal",
    "ToolGrant",
    "ToolLock",
    "tool_fingerprint",
    "pin_reviewed_tools",
    "DiscoveryReport",
    "Decision",
    "McpToolAdapter",
    "McpHost",
    "SEP",
]
