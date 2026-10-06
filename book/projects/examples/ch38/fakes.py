# path: book/projects/examples/ch38/fakes.py
"""Deterministic stand-ins for the outside world: a clock and Northwind's ticketing system."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentkit import FunctionTool, SideEffect, ToolContext, ToolOutput

from durable import SimulatedCrash


@dataclass
class FakeClock:
    now: float = 1_000_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@dataclass
class FakeTicketSystem:
    """The external system a side-effecting tool talks to.

    - `crash_after_commit`: the next create commits remotely, then the calling process dies
      before it sees the response (the classic lost-acknowledgment case).
    - `crash_before_commit`: the process dies before the request leaves.
    - `supports_lookup`: whether the system can answer "do you have a ticket created with
      idempotency key K?" Many real APIs can (a client reference field, an idempotency-key
      header with replay); some cannot.
    """

    supports_lookup: bool = True
    crash_after_commit: bool = False
    crash_before_commit: bool = False
    tickets: list[dict[str, Any]] = field(default_factory=list)

    def create(self, *, title: str, body: str, client_ref: str) -> dict[str, Any]:
        if self.crash_before_commit:
            self.crash_before_commit = False
            raise SimulatedCrash("died before the request was sent")
        ticket = {"id": f"TCK-{1000 + len(self.tickets)}", "title": title, "body": body, "client_ref": client_ref}
        self.tickets.append(ticket)
        if self.crash_after_commit:
            self.crash_after_commit = False
            raise SimulatedCrash("died after the remote commit, before reading the response")
        return ticket

    def find_by_client_ref(self, client_ref: str) -> dict[str, Any] | None:
        if not self.supports_lookup:
            raise NotImplementedError("this system cannot search by client reference")
        return next((t for t in self.tickets if t["client_ref"] == client_ref), None)


def ticket_tool(system: FakeTicketSystem) -> FunctionTool:
    def create_ticket(ctx: ToolContext, title: str, body: str) -> ToolOutput:
        t = system.create(title=title, body=body, client_ref=ctx.idempotency_key)
        return ToolOutput(content=f"created {t['id']}: {title}", data=t, artifacts={"ticket_id": t["id"]})

    return FunctionTool(
        "create_ticket", "Open a follow-up ticket in Northwind's ticketing system.",
        {"type": "object", "properties": {"title": {"type": "string"}, "body": {"type": "string"}},
         "required": ["title", "body"], "additionalProperties": False},
        fn=create_ticket, side_effect=SideEffect.WRITE, idempotent=False, pass_context=True,
    )


def ticket_reconciler(system: FakeTicketSystem):
    """Answers 'did this action already happen?' by asking the system of record."""

    def reconcile(key: str, arguments: dict[str, Any]) -> ToolOutput | None:
        t = system.find_by_client_ref(key)
        if t is None:
            return None
        return ToolOutput(content=f"created {t['id']}: {t['title']} (recovered after restart)", data=t,
                          artifacts={"ticket_id": t["id"]})

    return reconcile


def search_tool() -> FunctionTool:
    runbooks = {
        "it-database-failover-runbook": "Failover: confirm replica lag under 5 s, then run the switchover.",
        "trackline-latency-incident-2025": "Trackline p95 spike was caused by a missing index on scans.",
    }

    def search_runbooks(query: str) -> str:
        hits = [f"[{k}] {v}" for k, v in runbooks.items() if any(w in v.lower() for w in query.lower().split())]
        return "\n".join(hits) or "no results"

    return FunctionTool("search_runbooks", "Search runbooks and past incidents by keywords.",
                        {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"],
                         "additionalProperties": False}, fn=search_runbooks)
