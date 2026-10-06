# path: book/projects/examples/ch38/durable.py
"""Durable execution for agentkit runs.

Chapter 19's AgentRuntime is already event-sourced: every decision is appended to an
EventStore before state changes, and `resume()` rebuilds state from the log. This module
adds the three things a long-running, multi-worker deployment needs on top:

1. `SqliteEventStore`: the same EventStore protocol over SQLite, with a `runs` projection
   (status per run) updated in the same transaction as each append.
2. Leases with fencing: one worker owns a run at a time. Every append checks the lease and
   extends it in the same transaction, so a worker that lost its lease (a "zombie" that
   was paused by GC or a network partition) cannot write another event. A heartbeat
   thread keeps the lease alive while a slow tool runs between appends.
3. Outcome reconciliation for side-effecting tools: `ReconcilingTool` records intent in an
   idempotency ledger that lives outside the worker (Chapter 16's SQLiteIdempotencyStore)
   and, after a crash, asks the external system "did this action already happen?" before
   executing it again.

`DurableRunner` ties them together: start, resume, and recover orphaned runs.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from agentkit import (
    AgentRuntime, Budget, ErrorClass, Event, EventStore, GoalSet, Resumed, RunResult, Stopped, TerminationReason,
    Tool, ToolContext, ToolOutput, adapt_tool, event_from_json, event_to_json,
)
from toolkit import SQLiteIdempotencyStore


class SimulatedCrash(BaseException):
    """Stands in for `kill -9`. It derives from BaseException so that the runtime's
    `except Exception` around tool execution cannot catch and record it: the process
    simply stops, exactly where it was."""


class LeaseLost(RuntimeError):
    """This worker no longer owns the run; it must stop writing immediately."""


class LeaseHeld(RuntimeError):
    """Another live worker owns the run."""


# --------------------------------------------------------------------------- database
SCHEMA = """
-- PRIMARY KEY (run_id, seq) is the cross-process sequence check that JsonlEventStore
-- cannot provide (Chapter 19, exercise E4): a second writer of the same seq fails.
CREATE TABLE IF NOT EXISTS events (
    run_id TEXT NOT NULL, seq INTEGER NOT NULL, type TEXT NOT NULL, body TEXT NOT NULL, at REAL NOT NULL,
    PRIMARY KEY (run_id, seq)
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, status TEXT NOT NULL, stop_reason TEXT, updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS leases (
    run_id TEXT PRIMARY KEY, owner TEXT NOT NULL, fence INTEGER NOT NULL, expires_at REAL NOT NULL
);
"""


class Database:
    """One SQLite file shared by the event store, leases, and (in interrupts.py) the
    scheduler. `transaction()` takes the write lock up front (BEGIN IMMEDIATE) so that a
    check-then-write sequence is atomic across threads and processes."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None, timeout=10)
        if self.path != ":memory:":
            self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                yield self.conn
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            self.conn.execute("COMMIT")

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        with self._lock:
            return self.conn.execute(sql, params).fetchall()

    def executescript(self, sql: str) -> None:
        with self._lock:
            self.conn.executescript(sql)


# --------------------------------------------------------------------------- leases
@dataclass(frozen=True)
class Lease:
    run_id: str
    owner: str
    fence: int           # increases on every acquire of this run; a newer owner always has a larger fence
    expires_at: float


class LeaseManager:
    def __init__(self, db: Database, *, ttl_s: float = 30.0, clock: Callable[[], float] = time.time) -> None:
        self.db = db
        self.ttl_s = ttl_s
        self.clock = clock

    def acquire(self, run_id: str, owner: str) -> Lease | None:
        """Take the run if nobody holds a live lease. Returns None if another owner does."""
        now = self.clock()
        with self.db.transaction() as c:
            row = c.execute("SELECT owner, fence, expires_at FROM leases WHERE run_id=?", (run_id,)).fetchone()
            if row is not None and row[0] != owner and row[2] > now:
                return None
            fence = (row[1] + 1) if row else 1
            c.execute("INSERT INTO leases(run_id, owner, fence, expires_at) VALUES (?,?,?,?) "
                      "ON CONFLICT(run_id) DO UPDATE SET owner=excluded.owner, fence=excluded.fence, "
                      "expires_at=excluded.expires_at", (run_id, owner, fence, now + self.ttl_s))
        return Lease(run_id, owner, fence, now + self.ttl_s)

    def release(self, lease: Lease) -> None:
        # Expire the row instead of deleting it: the fence must keep counting up, or the next
        # owner would get fence 1 again and a zombie holding an old fence 1 would pass the check.
        with self.db.transaction() as c:
            c.execute("UPDATE leases SET expires_at=0 WHERE run_id=? AND fence=?", (lease.run_id, lease.fence))

    def renew(self, lease: Lease) -> bool:
        """Heartbeat: extend a lease this worker still holds. False means it was lost."""
        with self.db.transaction() as c:
            try:
                self.check_and_extend(c, lease)
            except LeaseLost:
                return False
        return True

    def holder(self, run_id: str) -> Lease | None:
        rows = self.db.query("SELECT owner, fence, expires_at FROM leases WHERE run_id=?", (run_id,))
        if not rows or rows[0][2] <= self.clock():
            return None
        return Lease(run_id, rows[0][0], rows[0][1], rows[0][2])

    def check_and_extend(self, c: sqlite3.Connection, lease: Lease) -> None:
        """Called inside the append transaction: the fence check and the write commit together."""
        now = self.clock()
        cur = c.execute("UPDATE leases SET expires_at=? WHERE run_id=? AND owner=? AND fence=? AND expires_at>?",
                        (now + self.ttl_s, lease.run_id, lease.owner, lease.fence, now))
        if cur.rowcount != 1:
            raise LeaseLost(f"{lease.owner} lost run {lease.run_id} (fence {lease.fence})")


# --------------------------------------------------------------------------- event store
def _status_after(event: Event) -> tuple[str, str | None] | None:
    if isinstance(event, (GoalSet, Resumed)):
        return "running", None
    if isinstance(event, Stopped):
        if event.reason is TerminationReason.COMPLETED:
            return "completed", event.reason.value
        if event.reason is TerminationReason.APPROVAL_REQUIRED:
            return "awaiting_approval", event.reason.value
        return "stopped", event.reason.value
    return None


class SqliteEventStore:
    """agentkit's EventStore protocol over SQLite. `fenced(lease)` returns a view that
    refuses to append unless `lease` is still the current one."""

    def __init__(self, db: Database, *, leases: LeaseManager | None = None, lease: Lease | None = None) -> None:
        self.db = db
        self.leases = leases
        self.lease = lease

    def fenced(self, leases: LeaseManager, lease: Lease) -> "SqliteEventStore":
        return SqliteEventStore(self.db, leases=leases, lease=lease)

    def append(self, event: Event) -> None:
        with self.db.transaction() as c:
            if self.lease is not None:
                assert self.leases is not None
                self.leases.check_and_extend(c, self.lease)
            try:
                c.execute("INSERT INTO events(run_id, seq, type, body, at) VALUES (?,?,?,?,?)",
                          (event.run_id, event.seq, getattr(event, "type"), event_to_json(event), event.at))
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"run {event.run_id}: seq {event.seq} already written") from exc
            status = _status_after(event)
            if status is not None:
                c.execute("INSERT INTO runs(run_id, status, stop_reason, updated_at) VALUES (?,?,?,?) "
                          "ON CONFLICT(run_id) DO UPDATE SET status=excluded.status, "
                          "stop_reason=excluded.stop_reason, updated_at=excluded.updated_at",
                          (event.run_id, status[0], status[1], event.at))

    def load(self, run_id: str) -> list[Event]:
        rows = self.db.query("SELECT body FROM events WHERE run_id=? ORDER BY seq", (run_id,))
        return [event_from_json(r[0]) for r in rows]

    def runs(self) -> list[str]:
        return [r[0] for r in self.db.query("SELECT run_id FROM runs ORDER BY run_id")]

    def runs_with_status(self, status: str) -> list[str]:
        return [r[0] for r in self.db.query("SELECT run_id FROM runs WHERE status=? ORDER BY run_id", (status,))]

    def status(self, run_id: str) -> str | None:
        rows = self.db.query("SELECT status FROM runs WHERE run_id=?", (run_id,))
        return rows[0][0] if rows else None


class CrashingStore:
    """Test helper: wraps a store and simulates process death just before an append that
    matches `when`. Use it to open the window between "side effect happened" and "result
    recorded", the window every durable system has to survive."""

    def __init__(self, inner: EventStore, when: Callable[[Event], bool]) -> None:
        self.inner = inner
        self.when = when
        self.armed = True

    def append(self, event: Event) -> None:
        if self.armed and self.when(event):
            self.armed = False
            raise SimulatedCrash(f"crashed before writing {getattr(event, 'type')} seq={event.seq}")
        self.inner.append(event)

    def load(self, run_id: str) -> list[Event]:
        return self.inner.load(run_id)

    def runs(self) -> list[str]:
        return self.inner.runs()


# --------------------------------------------------------------------------- outcome reconciliation
Reconciler = Callable[[str, dict[str, Any]], "ToolOutput | None"]


def _args_hash(arguments: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(arguments, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _output_to_json(out: ToolOutput) -> dict[str, Any]:
    return {"content": out.content, "ok": out.ok, "data": out.data, "artifacts": out.artifacts,
            "error_class": out.error_class.value if out.error_class else None, "error": out.error}


def _output_from_json(raw: dict[str, Any]) -> ToolOutput:
    cls = raw.get("error_class")
    return ToolOutput(content=raw["content"], ok=raw["ok"], data=raw.get("data"), artifacts=raw.get("artifacts") or {},
                      error_class=ErrorClass(cls) if cls else None, error=raw.get("error"))


@dataclass
class ReconcilingTool:
    """Wraps a side-effecting tool so that, under at-least-once execution, the side effect
    happens at most once per logical call.

    The key is agentkit's `ctx.idempotency_key` ("<run_id>:<request_id>"), which is stable
    across crashes because the request id comes from the event log. Outcomes:

    - no record: reserve the key, execute, record the output.
    - succeeded: return the recorded output; do not execute (crash after the side effect
      and the ledger write, before the ToolResult event).
    - in_progress or unknown: an earlier attempt died mid-flight. Ask `reconcile(key,
      arguments)`: a ToolOutput means "it happened, here is what it produced"; None means
      "it definitely did not happen", so execute now. Without a reconciler the outcome is
      unknowable, and the call fails as FATAL so a human decides instead of the model.

    The run lease, kept alive by the runner's heartbeat while the tool runs, means one worker
    per run, so an `in_progress` record for this run's key belongs to a dead attempt. The
    exception is a worker frozen (GC, partition) for longer than the lease TTL: fencing stops
    its next event, but not a side effect it is already performing. Closing that gap needs
    the external system itself to check the fence or the idempotency key.
    """

    inner: Any
    ledger: SQLiteIdempotencyStore
    reconcile: Reconciler | None = None
    ttl_s: float = 7 * 24 * 3600.0
    log: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._tool: Tool = adapt_tool(self.inner)
        self.name = self._tool.name
        self.side_effect = self._tool.side_effect
        self.requires_approval = self._tool.requires_approval
        self.idempotent = self._tool.idempotent

    @property
    def spec(self) -> Any:
        return self._tool.spec

    def execute(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        key = ctx.idempotency_key
        existing = self.ledger.begin(key, self.name, _args_hash(arguments), self.ttl_s)
        if existing is not None:
            if existing.status == "succeeded":
                self.log.append({"key": key, "outcome": "deduplicated"})
                return _output_from_json(existing.result)
            if self.reconcile is None:
                self.ledger.mark_unknown(key)
                self.log.append({"key": key, "outcome": "unknown"})
                return ToolOutput.failure(
                    f"an earlier attempt of {self.name} may or may not have taken effect and this system "
                    "cannot check; a human must reconcile before anything else is done", ErrorClass.FATAL)
            found = self.reconcile(key, arguments)
            if found is not None:
                self.ledger.complete(key, _output_to_json(found))
                self.log.append({"key": key, "outcome": "reconciled_found"})
                return found
            self.log.append({"key": key, "outcome": "reconciled_absent"})
        out = self._tool.execute(arguments, ctx)       # exceptions propagate; the record stays in_progress
        self.ledger.complete(key, _output_to_json(out))
        return out


# --------------------------------------------------------------------------- runner
RuntimeFactory = Callable[[EventStore], AgentRuntime]


class DurableRunner:
    """Owns runs through leases. `factory(store)` builds an AgentRuntime bound to a fenced
    store; build it per call so that tools, model client, and policy come from current code
    and configuration, while state comes only from the log."""

    def __init__(self, db: Database, factory: RuntimeFactory, *, owner: str, lease_ttl_s: float = 30.0,
                 heartbeat_s: float | None = None, clock: Callable[[], float] = time.time) -> None:
        self.db = db
        self.factory = factory
        self.owner = owner
        self.store = SqliteEventStore(db)
        self.leases = LeaseManager(db, ttl_s=lease_ttl_s, clock=clock)
        self.heartbeat_s = heartbeat_s if heartbeat_s is not None else lease_ttl_s / 3

    def start(self, goal: str, *, run_id: str, metadata: dict[str, Any] | None = None) -> RunResult:
        return self._with_lease(run_id, lambda rt: rt.run(goal, run_id=run_id, metadata=metadata))

    def resume(self, run_id: str, *, approve: bool | None = None, reason: str = "",
               budget: Budget | None = None, request_id: str | None = None) -> RunResult:
        """`request_id` binds a decision to the request it was made for (Chapter 19): if the
        run is now paused on a different request, agentkit refuses instead of approving it."""
        return self._with_lease(run_id, lambda rt: rt.resume(run_id, approve=approve, reason=reason, budget=budget,
                                                             request_id=request_id))

    def recover(self) -> list[RunResult]:
        """Resume every run that is marked running but has no live lease: its worker died."""
        results = []
        for run_id in self.store.runs_with_status("running"):
            if self.leases.holder(run_id) is not None:
                continue
            try:
                results.append(self.resume(run_id))
            except LeaseHeld:
                continue                       # another recovering worker got there first
        return results

    def _with_lease(self, run_id: str, fn: Callable[[AgentRuntime], RunResult]) -> RunResult:
        lease = self.leases.acquire(run_id, self.owner)
        if lease is None:
            raise LeaseHeld(f"run {run_id} is owned by {self.leases.holder(run_id)}")
        runtime = self.factory(self.store.fenced(self.leases, lease))
        stop = threading.Event()
        beat = threading.Thread(target=self._heartbeat, args=(lease, stop), daemon=True)
        beat.start()
        try:
            result = fn(runtime)
        except SimulatedCrash:
            stop.set()
            raise                               # a dead process releases nothing; the lease must expire
        except BaseException:
            stop.set()
            self.leases.release(lease)
            raise
        stop.set()
        self.leases.release(lease)              # paused, stopped, or completed: nobody needs to hold it
        return result

    def _heartbeat(self, lease: Lease, stop: threading.Event) -> None:
        """Appends extend the lease, but a tool can run longer than the TTL between two
        appends. Renew on a timer until the call finishes or the lease is gone."""
        while not stop.wait(self.heartbeat_s):
            if not self.leases.renew(lease):
                return


__all__ = [
    "SimulatedCrash", "LeaseLost", "LeaseHeld", "Database", "Lease", "LeaseManager", "SqliteEventStore",
    "CrashingStore", "ReconcilingTool", "Reconciler", "DurableRunner", "RuntimeFactory",
]
