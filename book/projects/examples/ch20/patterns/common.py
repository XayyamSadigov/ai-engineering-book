# path: book/projects/examples/ch20/patterns/common.py
"""Shared pieces for the Chapter 20 patterns: role-tagged prompts, a metering client,
a uniform result type, and one factory for bounded agents.

Every pattern in this package is composition over agentkit's AgentRuntime. None of them
contains a loop that calls a model and executes tools; that loop exists once, in agentkit.
"""
from __future__ import annotations

import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator, Sequence

from pydantic import BaseModel

from aie_core.llm.client import LLMClient
from aie_core.llm.gateway import PricingTable
from aie_core.llm.structured import complete_structured
from aie_core.llm.types import Completion, CompletionRequest, Message, StreamEvent
from agentkit import AgentRuntime, Budget, DefinitionOfDone, EventStore, LoopConfig, RunResult

ROLE_TAG = re.compile(r"\[role:([a-z0-9_:\-]+)\]")
SEP = "."   # child run ids are "parent.child"; "." is valid in JSONL event-store file names


def role_prompt(role: str, instructions: str) -> str:
    """Prefix a system prompt with a role tag. Traces, meters, and scripted test models key on it."""
    return f"[role:{role}] {instructions}"


def role_of(req: CompletionRequest) -> str:
    for m in req.messages:
        if m.role.value == "system":
            found = ROLE_TAG.search(m.text)
            if found:
                return found.group(1)
    return "unknown"


class Meter:
    """An LLMClient wrapper that counts calls, tokens, and cost per role. Thread-safe.

    Metering is orthogonal to every pattern: wrap the client once and compare architectures
    on the same axis (calls, tokens, cost) without touching pattern code.
    """

    def __init__(self, inner: LLMClient, pricing: PricingTable | None = None) -> None:
        self.inner = inner
        self.pricing = pricing
        self.provider = getattr(inner, "provider", "unknown")
        self.supports_response_schema = getattr(inner, "supports_response_schema", False)
        self.calls: Counter[str] = Counter()
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self._lock = threading.Lock()

    def _record(self, req: CompletionRequest, c: Completion) -> Completion:
        cost = self.pricing.cost_usd(c.model, c.usage) if self.pricing else 0.0
        with self._lock:
            self.calls[role_of(req)] += 1
            self.input_tokens += c.usage.input_tokens
            self.output_tokens += c.usage.output_tokens
            self.cost_usd += cost
        return c

    def complete(self, req: CompletionRequest) -> Completion:
        return self._record(req, self.inner.complete(req))

    async def acomplete(self, req: CompletionRequest) -> Completion:
        return self._record(req, await self.inner.acomplete(req))

    def stream(self, req: CompletionRequest) -> Iterator[StreamEvent]:
        return self.inner.stream(req)

    def astream(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:
        return self.inner.astream(req)

    @property
    def total_calls(self) -> int:
        return sum(self.calls.values())

    def snapshot(self) -> dict[str, Any]:
        return {"calls": self.total_calls, "by_role": dict(self.calls),
                "tokens": self.input_tokens + self.output_tokens, "cost_usd": round(self.cost_usd, 6)}


@dataclass
class PatternResult:
    """What every pattern returns: the answer, every agent run it made, and pattern-specific data."""

    pattern: str
    ok: bool
    answer: str | None
    runs: list[RunResult] = field(default_factory=list)
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def tool_calls(self) -> int:
        return sum(r.state.usage.tool_calls for r in self.runs)

    @property
    def agent_steps(self) -> int:
        return sum(r.state.usage.steps for r in self.runs)

    def trajectory(self) -> list[str]:
        return [t for r in self.runs for t in r.trajectory()]


def make_agent(
    llm: LLMClient,
    tools: Sequence[Any],
    *,
    role: str,
    instructions: str,
    budget: Budget | None = None,
    dod: DefinitionOfDone | None = None,
    store: EventStore | None = None,
    config: LoopConfig | None = None,
    principal: dict[str, Any] | None = None,
) -> AgentRuntime:
    """One bounded agent. Every pattern builds its agents through this function."""
    return AgentRuntime(llm, list(tools), system_prompt=role_prompt(role, instructions),
                        budget=budget or Budget(max_steps=5, max_tool_calls=5), dod=dod, store=store,
                        config=config, principal=principal)


def ask(llm: LLMClient, role: str, system: str, user: str, *, max_tokens: int = 800) -> str:
    """A single model call that is not an agent: no tools, no loop."""
    req = CompletionRequest(messages=[Message.system(role_prompt(role, system)), Message.user(user)],
                            max_tokens=max_tokens, metadata={"role": role})
    return llm.complete(req).text


def ask_structured(llm: LLMClient, role: str, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
    """A single structured call, validated and repaired by aie_core."""
    req = CompletionRequest(messages=[Message.system(role_prompt(role, system)), Message.user(user)],
                            metadata={"role": role})
    out, _ = complete_structured(llm, req, schema)
    return out


def obj(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


__all__ = ["SEP", "ROLE_TAG", "role_prompt", "role_of", "Meter", "PatternResult", "make_agent", "ask", "ask_structured",
           "obj"]
