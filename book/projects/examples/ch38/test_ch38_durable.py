# path: book/projects/examples/ch38/test_ch38_durable.py
"""Crash, recover, reconcile: the at-least-once window and how the harness closes it."""
from __future__ import annotations

import pytest

from aie_core.llm.providers import FakeLLM
from aie_core.llm.types import ToolCall
from agentkit import (
    AgentRuntime, DefinitionOfDone, Event, GoalSet, Resumed, TerminationReason, ToolResult, has_artifact, replay,
)
from toolkit import SQLiteIdempotencyStore

from durable import (
    CrashingStore, Database, DurableRunner, LeaseHeld, LeaseLost, LeaseManager, ReconcilingTool, SimulatedCrash,
    SqliteEventStore,
)
from fakes import FakeClock, FakeTicketSystem, search_tool, ticket_reconciler, ticket_tool

GOAL = "Trackline lookups are slow for logistics. Find the known cause and open a follow-up ticket."
ANSWER = "Cause: missing index on scans [trackline-latency-incident-2025]. Follow-up ticket opened."


def script() -> FakeLLM:
    return FakeLLM(responses=[
        [ToolCall(id="c1", name="search_runbooks", arguments={"query": "trackline p95"})],
        [ToolCall(id="c2", name="create_ticket", arguments={"title": "Add scan index", "body": "See incident."})],
        ANSWER,
    ])


def setup(system: FakeTicketSystem, *, reconcile: bool, wrap: bool = True, crash_when=None):
    clock = FakeClock()
    db = Database()
    ledger = SQLiteIdempotencyStore(clock=clock)
    llm = script()
    raw_tool = ticket_tool(system)
    tool = ReconcilingTool(raw_tool, ledger, ticket_reconciler(system) if reconcile else None) if wrap else raw_tool
    crash = {"armed": crash_when}

    def factory(store):
        if crash["armed"] is not None:            # only the first worker gets the crashing store
            store, crash["armed"] = CrashingStore(store, crash["armed"]), None
        return AgentRuntime(llm, [search_tool(), tool], store=store, dod=DefinitionOfDone(has_artifact("ticket_id")))

    w1 = DurableRunner(db, factory, owner="worker-1", lease_ttl_s=30, clock=clock)
    w2 = DurableRunner(db, factory, owner="worker-2", lease_ttl_s=30, clock=clock)
    return clock, db, ledger, tool, w1, w2


def crash_then_recover(clock, w1, w2):
    with pytest.raises(SimulatedCrash):
        w1.start(GOAL, run_id="inc-1")
    assert w1.store.status("inc-1") == "running"          # nobody wrote Stopped: it looks alive
    assert w2.recover() == []                              # lease still live: a slow worker is not a dead one
    clock.advance(31)
    recovered = w2.recover()
    assert len(recovered) == 1
    return recovered[0]


def test_naive_tool_duplicates_side_effect_after_crash():
    system = FakeTicketSystem(crash_after_commit=True)
    clock, db, _, _, w1, w2 = setup(system, reconcile=False, wrap=False)
    result = crash_then_recover(clock, w1, w2)
    assert result.ok
    assert len(system.tickets) == 2                        # at-least-once execution, visible to the customer


def test_reconciliation_finds_the_committed_ticket():
    system = FakeTicketSystem(crash_after_commit=True)
    clock, db, ledger, tool, w1, w2 = setup(system, reconcile=True)
    result = crash_then_recover(clock, w1, w2)
    assert result.ok and len(system.tickets) == 1
    assert [e["outcome"] for e in tool.log] == ["reconciled_found"]
    assert result.state.artifacts["ticket_id"] == system.tickets[0]["id"]
    assert any(isinstance(e, Resumed) and e.by == "recovery" for e in result.events)


def test_crash_before_commit_executes_exactly_once():
    system = FakeTicketSystem(crash_before_commit=True)
    clock, db, ledger, tool, w1, w2 = setup(system, reconcile=True)
    result = crash_then_recover(clock, w1, w2)
    assert result.ok and len(system.tickets) == 1
    assert [e["outcome"] for e in tool.log] == ["reconciled_absent"]


def test_crash_after_ledger_write_is_deduplicated_without_asking_the_system():
    system = FakeTicketSystem(supports_lookup=False)       # the lookup would fail if it were needed
    clock, db, ledger, tool, w1, w2 = setup(
        system, reconcile=True,
        crash_when=lambda e: isinstance(e, ToolResult) and e.tool == "create_ticket")
    result = crash_then_recover(clock, w1, w2)
    assert result.ok and len(system.tickets) == 1
    assert [e["outcome"] for e in tool.log] == ["deduplicated"]


def test_unknowable_outcome_stops_for_a_human():
    system = FakeTicketSystem(crash_after_commit=True, supports_lookup=False)
    clock, db, ledger, tool, w1, w2 = setup(system, reconcile=False)
    result = crash_then_recover(clock, w1, w2)
    assert result.stop_reason is TerminationReason.FATAL_ERROR
    assert "human must reconcile" in result.detail
    assert len(system.tickets) == 1                        # not repeated
    assert ledger.get("inc-1:2.0").status == "unknown"


def test_zombie_worker_cannot_write_after_losing_its_lease():
    clock, db = FakeClock(), Database()
    leases = LeaseManager(db, ttl_s=10, clock=clock)
    base = SqliteEventStore(db)
    old = leases.acquire("r1", "worker-1")
    zombie = base.fenced(leases, old)
    zombie.append(GoalSet(run_id="r1", seq=0, goal="g"))
    clock.advance(11)                                      # worker-1 stalls past its lease
    new = leases.acquire("r1", "worker-2")
    assert new is not None and new.fence == old.fence + 1
    with pytest.raises(LeaseLost):
        zombie.append(Resumed(run_id="r1", seq=1))
    base.fenced(leases, new).append(Resumed(run_id="r1", seq=1, by="recovery"))
    assert [type(e).__name__ for e in base.load("r1")] == ["GoalSet", "Resumed"]


def test_live_lease_blocks_second_owner_and_appends_extend_it():
    clock, db = FakeClock(), Database()
    leases = LeaseManager(db, ttl_s=10, clock=clock)
    lease = leases.acquire("r2", "worker-1")
    assert leases.acquire("r2", "worker-2") is None
    store = SqliteEventStore(db).fenced(leases, lease)
    for seq in range(3):                                   # each append is also a heartbeat
        clock.advance(6)
        store.append(Resumed(run_id="r2", seq=seq) if seq else GoalSet(run_id="r2", seq=0, goal="g"))
    assert leases.acquire("r2", "worker-2") is None        # 18 s later, still owned
    runner = DurableRunner(db, lambda s: None, owner="worker-2", lease_ttl_s=10, clock=clock)  # type: ignore[arg-type]
    with pytest.raises(LeaseHeld):
        runner.resume("r2")


def test_sqlite_store_rejects_duplicate_seq_and_survives_reopen(tmp_path):
    path = tmp_path / "runs.db"
    store = SqliteEventStore(Database(path))
    store.append(GoalSet(run_id="r3", seq=0, goal="g"))
    with pytest.raises(ValueError):
        store.append(Resumed(run_id="r3", seq=0))
    reopened = SqliteEventStore(Database(path))           # a new process
    events: list[Event] = reopened.load("r3")
    assert isinstance(events[0], GoalSet) and reopened.status("r3") == "running"


def test_recovered_run_replays_identically_without_side_effects():
    system = FakeTicketSystem(crash_after_commit=True)
    clock, db, ledger, tool, w1, w2 = setup(system, reconcile=True)
    crash_then_recover(clock, w1, w2)
    report = replay(w2.store.load("inc-1"))
    assert report.identical, report.summary()
    assert len(system.tickets) == 1
