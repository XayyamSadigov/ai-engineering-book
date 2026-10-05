# path: book/projects/examples/ch38/demo.py
"""Crash a run after a remote commit, recover it on another worker, and show that the ticket
exists exactly once. Offline and deterministic.

    python demo.py [path/to/runs.db]
"""
from __future__ import annotations

import sys

from aie_core.llm.providers import FakeLLM
from aie_core.llm.types import ToolCall
from agentkit import AgentRuntime
from toolkit import SQLiteIdempotencyStore

from durable import Database, DurableRunner, ReconcilingTool, SimulatedCrash
from fakes import FakeClock, FakeTicketSystem, search_tool, ticket_reconciler, ticket_tool


def main(db_path: str = ":memory:") -> None:
    clock, db, system = FakeClock(), Database(db_path), FakeTicketSystem(crash_after_commit=True)
    ledger = SQLiteIdempotencyStore(clock=clock)
    tool = ReconcilingTool(ticket_tool(system), ledger, ticket_reconciler(system))
    llm = FakeLLM(responses=[
        [ToolCall(id="c1", name="search_runbooks", arguments={"query": "trackline p95"})],
        [ToolCall(id="c2", name="create_ticket", arguments={"title": "Add scan index", "body": "See INC."})],
        "Cause: missing scan index [trackline-latency-incident-2025]; follow-up ticket opened.",
    ])

    def factory(store):
        return AgentRuntime(llm, [search_tool(), tool], store=store)

    w1 = DurableRunner(db, factory, owner="worker-1", clock=clock)
    w2 = DurableRunner(db, factory, owner="worker-2", clock=clock)
    try:
        w1.start("Trackline is slow; find the cause and open a follow-up ticket.", run_id="demo-1")
    except SimulatedCrash as exc:
        print(f"worker-1 died: {exc}")
    print("status after crash:", w1.store.status("demo-1"), "| recover now:", w2.recover())
    clock.advance(31)
    [result] = w2.recover()
    print("recovered:", result.stop_reason.value, "| tickets in system:", [t["id"] for t in system.tickets])
    print("reconciliation log:", tool.log)
    for e in w2.store.load("demo-1"):
        print(f"  {e.seq:2d} {getattr(e, 'type'):20s} {getattr(e, 'tool', '')}")


if __name__ == "__main__":
    main(*sys.argv[1:])
