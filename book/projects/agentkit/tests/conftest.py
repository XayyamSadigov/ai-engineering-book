# path: book/projects/agentkit/tests/conftest.py
"""Shared fixtures: a tiny Northwind runbook corpus and tools that count their executions."""
from __future__ import annotations

from typing import Any

import pytest

from aie_core.llm.types import ToolCall
from agentkit import FunctionTool, SideEffect, ToolContext

RUNBOOKS = {
    "it-vpn-access-runbook": "VPN: if login loops, clear the cached profile and re-enroll the device certificate.",
    "it-database-failover-runbook": "Failover: confirm replica lag under 5 s, then run patronictl switchover.",
    "it-password-reset-runbook": "Password reset: verify identity with the manager, then reset in Northwind ID.",
}


class Counter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.keys: list[str] = []

    def count(self, name: str) -> int:
        return sum(1 for n, _ in self.calls if n == name)


@pytest.fixture
def counter() -> Counter:
    return Counter()


@pytest.fixture
def tools(counter: Counter) -> list[FunctionTool]:
    def search_runbooks(query: str) -> str:
        counter.calls.append(("search_runbooks", {"query": query}))
        hits = [f"[{doc_id}] {text}" for doc_id, text in RUNBOOKS.items()
                if any(w in text.lower() for w in query.lower().split())]
        return "\n".join(hits) if hits else "no results"

    def send_reply(ctx: ToolContext, ticket_id: str, body: str) -> dict[str, Any]:
        counter.calls.append(("send_reply", {"ticket_id": ticket_id, "body": body}))
        counter.keys.append(ctx.idempotency_key)
        return {"sent": True, "ticket_id": ticket_id}

    return [
        FunctionTool(
            name="search_runbooks", description="Search IT runbooks by keywords.",
            parameters={"type": "object", "properties": {"query": {"type": "string", "minLength": 2}},
                        "required": ["query"], "additionalProperties": False},
            fn=search_runbooks,
        ),
        FunctionTool(
            name="send_reply", description="Send a reply to the ticket requester.",
            parameters={"type": "object",
                        "properties": {"ticket_id": {"type": "string"}, "body": {"type": "string"}},
                        "required": ["ticket_id", "body"], "additionalProperties": False},
            fn=send_reply, side_effect=SideEffect.IRREVERSIBLE, idempotent=False, pass_context=True,
        ),
    ]


def call(name: str, i: int = 0, **arguments: Any) -> list[ToolCall]:
    return [ToolCall(id=f"call_{name}_{i}", name=name, arguments=arguments)]
