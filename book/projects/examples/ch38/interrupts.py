# path: book/projects/examples/ch38/interrupts.py
"""Interrupts as first-class, persisted states: human approvals with expiry and escalation,
timers, and external events (webhooks), none of which keep a process alive while waiting.

agentkit already pauses a run when a call needs approval (TerminationReason.APPROVAL_REQUIRED)
and resumes it from the log. This module gives that pause a durable record with a deadline,
an assignee, and an escalation chain, and reuses the same mechanism for waits:

    a wait is an approval granted by the clock (timer) or by the outside world (event).

The agent calls `wait_until` or `wait_for_event`; both tools declare `requires_approval`, so
the runtime pauses exactly as for a risky tool. `after_run` sees which tool is pending and
records a timer or an event subscription instead of a human task. When the timer fires or the
event arrives, the run is resumed with approve=True, the wait tool executes, and it returns the
delivered payload as its observation. Between pause and resume there is no process, no thread,
and no lease: only rows in SQLite.
"""
from __future__ import annotations

import json
import time
import uuid
from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel, Field

from agentkit import AgentStatus, FunctionTool, RunResult, SideEffect, ToolContext, ToolOutput, derive_state

from durable import Database, DurableRunner

WAIT_UNTIL = "wait_until"
WAIT_FOR_EVENT = "wait_for_event"

INTERRUPT_SCHEMA = """
CREATE TABLE IF NOT EXISTS interrupts (
    id TEXT PRIMARY KEY, run_id TEXT NOT NULL, request_id TEXT NOT NULL, kind TEXT NOT NULL,
    tool TEXT NOT NULL, arguments TEXT NOT NULL, status TEXT NOT NULL,
    assignee TEXT, level INTEGER NOT NULL DEFAULT 0, correlation_key TEXT,
    created_at REAL NOT NULL, escalate_at REAL, expires_at REAL,
    resolution TEXT, decided_by TEXT, history TEXT NOT NULL DEFAULT '[]',
    UNIQUE (run_id, request_id)
);
CREATE TABLE IF NOT EXISTS timers (
    id TEXT PRIMARY KEY, interrupt_id TEXT NOT NULL, fire_at REAL NOT NULL, purpose TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'scheduled'
);
CREATE INDEX IF NOT EXISTS timers_due ON timers(status, fire_at);
CREATE TABLE IF NOT EXISTS delivered_events (event_id TEXT PRIMARY KEY, received_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS inbox (
    correlation_key TEXT NOT NULL, event_id TEXT NOT NULL, payload TEXT NOT NULL, received_at REAL NOT NULL
);
"""


class InterruptKind(str, Enum):
    APPROVAL = "approval"
    TIMER = "timer"
    EVENT = "event"


class InterruptStatus(str, Enum):
    PENDING = "pending"
    RESOLVED = "resolved"
    EXPIRED = "expired"


class EscalationPolicy(BaseModel):
    """Who must decide, how long each level gets, and what silence means.

    Expiry always resolves to a denial: an approval nobody gave is not an approval."""

    chain: list[str] = Field(min_length=1)       # assignees in escalation order
    escalate_after_s: float = 3600.0             # per level
    expire_after_s: float = 24 * 3600.0          # total, from creation


class Interrupt(BaseModel):
    id: str
    run_id: str
    request_id: str
    kind: InterruptKind
    tool: str
    arguments: dict[str, Any]
    status: InterruptStatus
    assignee: str | None = None
    level: int = 0
    correlation_key: str | None = None
    created_at: float
    escalate_at: float | None = None
    expires_at: float | None = None
    resolution: dict[str, Any] | None = None
    decided_by: str | None = None
    history: list[dict[str, Any]] = Field(default_factory=list)


Notifier = Callable[[Interrupt, str], None]


class InterruptManager:
    def __init__(self, db: Database, runner: DurableRunner, *, policies: dict[str, EscalationPolicy] | None = None,
                 default_policy: EscalationPolicy | None = None, notify: Notifier | None = None,
                 clock: Callable[[], float] = time.time, max_wait_s: float = 30 * 24 * 3600.0) -> None:
        self.db = db
        self.runner = runner
        self.policies = policies or {}
        self.default_policy = default_policy or EscalationPolicy(chain=["support-lead"])
        self.notify = notify or (lambda it, what: None)
        self.clock = clock
        self.max_wait_s = max_wait_s
        db.executescript(INTERRUPT_SCHEMA)

    # ------------------------------------------------------------------ tools the agent calls
    def wait_tools(self) -> list[FunctionTool]:
        def wait_until(ctx: ToolContext, seconds: int, reason: str = "") -> ToolOutput:
            res = self.resolution(ctx.run_id, ctx.request_id)
            return ToolOutput(content=f"timer fired after {seconds}s at t={res.get('at')}", data=res)

        def wait_for_event(ctx: ToolContext, event_type: str, correlation_id: str, timeout_s: int) -> ToolOutput:
            res = self.resolution(ctx.run_id, ctx.request_id)
            if res.get("status") == "received":
                return ToolOutput(content=f"{event_type} received: {json.dumps(res['payload'], sort_keys=True)}",
                                  data=res)
            return ToolOutput(content=f"no {event_type} for {correlation_id} within {timeout_s}s (timed out)",
                              data=res)

        return [
            FunctionTool(WAIT_UNTIL, "Pause this task for a number of seconds, then continue. Costs nothing while "
                         "waiting.", {"type": "object", "properties": {
                             "seconds": {"type": "integer", "minimum": 1}, "reason": {"type": "string"}},
                             "required": ["seconds"], "additionalProperties": False},
                         fn=wait_until, requires_approval=True, pass_context=True),
            FunctionTool(WAIT_FOR_EVENT, "Pause until an external event arrives (for example a delivery scan or "
                         "a vendor reply) or the timeout passes.", {"type": "object", "properties": {
                             "event_type": {"type": "string"}, "correlation_id": {"type": "string"},
                             "timeout_s": {"type": "integer", "minimum": 1}},
                             "required": ["event_type", "correlation_id", "timeout_s"],
                             "additionalProperties": False},
                         fn=wait_for_event, requires_approval=True, pass_context=True, side_effect=SideEffect.READ),
        ]

    # ------------------------------------------------------------------ pause bookkeeping
    def after_run(self, result: RunResult) -> Interrupt | None:
        """Record why a run paused. Idempotent per (run_id, request_id)."""
        if result.status is not AgentStatus.AWAITING_APPROVAL:
            return None
        rec = result.state.pending_approval
        assert rec is not None
        existing = self._find(result.run_id, rec.request_id)
        if existing is not None:
            return existing
        now = self.clock()
        it = Interrupt(id=uuid.uuid4().hex[:12], run_id=result.run_id, request_id=rec.request_id, tool=rec.tool,
                       arguments=dict(rec.arguments), status=InterruptStatus.PENDING, created_at=now,
                       kind=InterruptKind.APPROVAL)
        if rec.tool == WAIT_UNTIL:
            it.kind = InterruptKind.TIMER
            self._insert(it)
            self._schedule(it.id, now + min(float(rec.arguments["seconds"]), self.max_wait_s), "fire")
        elif rec.tool == WAIT_FOR_EVENT:
            it.kind = InterruptKind.EVENT
            it.correlation_key = f"{rec.arguments['event_type']}:{rec.arguments['correlation_id']}"
            self._insert(it)
            early = self.db.query("SELECT event_id, payload FROM inbox WHERE correlation_key=? "
                                  "ORDER BY received_at LIMIT 1", (it.correlation_key,))
            if early:                                       # the event beat the wait: resolve at once
                self._consume_inbox(it.correlation_key, early[0][0])
                self._resolve_and_resume(it, {"status": "received", "payload": json.loads(early[0][1]),
                                              "event_id": early[0][0], "at": now}, by="event", approve=True)
                return self._find(result.run_id, rec.request_id)
            self._schedule(it.id, now + min(float(rec.arguments["timeout_s"]), self.max_wait_s), "timeout")
        else:
            policy = self.policies.get(rec.tool, self.default_policy)
            it.assignee = policy.chain[0]
            it.escalate_at = now + policy.escalate_after_s if len(policy.chain) > 1 else None
            it.expires_at = now + policy.expire_after_s
            it.history.append({"at": now, "event": "assigned", "to": it.assignee})
            self._insert(it)
            self.notify(it, "assigned")
        return it

    # ------------------------------------------------------------------ inputs from the world
    def decide(self, interrupt_id: str, *, approve: bool, by: str, reason: str = "") -> RunResult:
        """A human decision. Only someone on the chain up to the current level may decide, and
        only while the run is still paused on exactly this request."""
        it = self.get(interrupt_id)
        if it.kind is not InterruptKind.APPROVAL:
            raise ValueError(f"interrupt {interrupt_id} is a {it.kind.value}, not an approval")
        if it.status is not InterruptStatus.PENDING:
            raise ValueError(f"interrupt {interrupt_id} is already {it.status.value}")
        policy = self.policies.get(it.tool, self.default_policy)
        if by not in policy.chain[: it.level + 1]:
            raise PermissionError(f"{by} may not decide {it.tool} at escalation level {it.level}")
        state = derive_state(self.runner.store.load(it.run_id))
        pending = state.pending_approval
        if pending is None or pending.request_id != it.request_id:
            raise ValueError("the run is no longer paused on this request; the decision is stale")
        return self._resolve_and_resume(it, {"approved": approve, "reason": reason, "at": self.clock()},
                                        by=by, approve=approve, note=reason)

    def deliver(self, *, event_id: str, event_type: str, correlation_id: str,
                payload: dict[str, Any]) -> RunResult | None:
        """Webhook entry point. Providers deliver at least once, so `event_id` is deduplicated
        before anything else happens. Events with no waiter yet are kept in the inbox."""
        now = self.clock()
        with self.db.transaction() as c:
            fresh = c.execute("INSERT OR IGNORE INTO delivered_events(event_id, received_at) VALUES (?,?)",
                              (event_id, now)).rowcount == 1
        if not fresh:
            return None
        key = f"{event_type}:{correlation_id}"
        rows = self.db.query("SELECT id FROM interrupts WHERE kind='event' AND status='pending' AND correlation_key=? "
                             "ORDER BY created_at LIMIT 1", (key,))
        if not rows:
            with self.db.transaction() as c:
                c.execute("INSERT INTO inbox(correlation_key, event_id, payload, received_at) VALUES (?,?,?,?)",
                          (key, event_id, json.dumps(payload), now))
            return None
        it = self.get(rows[0][0])
        self._cancel_timers(it.id)
        return self._resolve_and_resume(it, {"status": "received", "payload": payload, "event_id": event_id,
                                             "at": now}, by="event", approve=True)

    def tick(self) -> list[RunResult]:
        """Run by a scheduler every few seconds (cron, a queue consumer, a k8s CronJob).
        Fires due timers, escalates stale approvals, expires dead ones. Safe to run on
        several replicas: each state change is a conditional UPDATE that only one wins."""
        now = self.clock()
        results: list[RunResult] = []
        for timer_id, interrupt_id, purpose in self.db.query(
                "SELECT id, interrupt_id, purpose FROM timers WHERE status='scheduled' AND fire_at<=? "
                "ORDER BY fire_at", (now,)):
            with self.db.transaction() as c:
                won = c.execute("UPDATE timers SET status='fired' WHERE id=? AND status='scheduled'",
                                (timer_id,)).rowcount == 1
            if not won:
                continue
            it = self.get(interrupt_id)
            if it.status is not InterruptStatus.PENDING:
                continue
            status = "fired" if purpose == "fire" else "timeout"
            results.append(self._resolve_and_resume(it, {"status": status, "at": now}, by="clock", approve=True))
        for (interrupt_id,) in self.db.query("SELECT id FROM interrupts WHERE kind='approval' AND status='pending' "
                                             "AND expires_at<=?", (now,)):
            it = self.get(interrupt_id)
            results.append(self._resolve_and_resume(
                it, {"approved": False, "reason": "expired", "at": now}, by="clock", approve=False,
                note=f"approval for {it.tool} expired unanswered; treated as denied", status=InterruptStatus.EXPIRED))
        for (interrupt_id,) in self.db.query("SELECT id FROM interrupts WHERE kind='approval' AND status='pending' "
                                             "AND escalate_at IS NOT NULL AND escalate_at<=?", (now,)):
            self._escalate(self.get(interrupt_id), now)
        return results

    # ------------------------------------------------------------------ queries
    def get(self, interrupt_id: str) -> Interrupt:
        rows = self.db.query("SELECT * FROM interrupts WHERE id=?", (interrupt_id,))
        if not rows:
            raise KeyError(interrupt_id)
        return self._row(rows[0])

    def pending(self, assignee: str | None = None) -> list[Interrupt]:
        rows = self.db.query("SELECT * FROM interrupts WHERE status='pending' ORDER BY created_at")
        items = [self._row(r) for r in rows]
        return [i for i in items if assignee is None or i.assignee == assignee]

    def resolution(self, run_id: str, request_id: str) -> dict[str, Any]:
        it = self._find(run_id, request_id)
        if it is None or it.resolution is None:
            raise RuntimeError(f"no resolution recorded for {run_id}:{request_id}")
        return it.resolution

    # ------------------------------------------------------------------ internals
    def _resolve_and_resume(self, it: Interrupt, resolution: dict[str, Any], *, by: str, approve: bool,
                            note: str = "", status: InterruptStatus = InterruptStatus.RESOLVED) -> RunResult:
        now = self.clock()
        history = it.history + [{"at": now, "event": status.value, "by": by}]
        with self.db.transaction() as c:
            won = c.execute("UPDATE interrupts SET status=?, resolution=?, decided_by=?, history=? "
                            "WHERE id=? AND status='pending'",
                            (status.value, json.dumps(resolution), by, json.dumps(history), it.id)).rowcount == 1
        if not won:
            raise ValueError(f"interrupt {it.id} was resolved concurrently")
        self._cancel_timers(it.id)
        result = self.runner.resume(it.run_id, approve=approve, reason=note or f"{it.kind.value} resolved by {by}")
        self.after_run(result)                  # the resumed run may pause again on a new request
        return result

    def _escalate(self, it: Interrupt, now: float) -> None:
        policy = self.policies.get(it.tool, self.default_policy)
        level = it.level + 1
        if level >= len(policy.chain):
            with self.db.transaction() as c:
                c.execute("UPDATE interrupts SET escalate_at=NULL WHERE id=?", (it.id,))
            return
        history = it.history + [{"at": now, "event": "escalated", "to": policy.chain[level]}]
        next_at = now + policy.escalate_after_s if level + 1 < len(policy.chain) else None
        with self.db.transaction() as c:
            won = c.execute("UPDATE interrupts SET level=?, assignee=?, escalate_at=?, history=? "
                            "WHERE id=? AND status='pending' AND level=?",
                            (level, policy.chain[level], next_at, json.dumps(history), it.id, it.level)).rowcount
        if won:
            self.notify(self.get(it.id), "escalated")

    def _schedule(self, interrupt_id: str, fire_at: float, purpose: str) -> None:
        with self.db.transaction() as c:
            c.execute("INSERT INTO timers(id, interrupt_id, fire_at, purpose) VALUES (?,?,?,?)",
                      (uuid.uuid4().hex[:12], interrupt_id, fire_at, purpose))

    def _cancel_timers(self, interrupt_id: str) -> None:
        with self.db.transaction() as c:
            c.execute("UPDATE timers SET status='cancelled' WHERE interrupt_id=? AND status='scheduled'",
                      (interrupt_id,))

    def _consume_inbox(self, key: str, event_id: str) -> None:
        with self.db.transaction() as c:
            c.execute("DELETE FROM inbox WHERE correlation_key=? AND event_id=?", (key, event_id))

    def _insert(self, it: Interrupt) -> None:
        with self.db.transaction() as c:
            c.execute("INSERT INTO interrupts(id, run_id, request_id, kind, tool, arguments, status, assignee, level, "
                      "correlation_key, created_at, escalate_at, expires_at, history) "
                      "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (it.id, it.run_id, it.request_id, it.kind.value, it.tool, json.dumps(it.arguments),
                       it.status.value, it.assignee, it.level, it.correlation_key, it.created_at, it.escalate_at,
                       it.expires_at, json.dumps(it.history)))

    def _find(self, run_id: str, request_id: str) -> Interrupt | None:
        rows = self.db.query("SELECT * FROM interrupts WHERE run_id=? AND request_id=?", (run_id, request_id))
        return self._row(rows[0]) if rows else None

    @staticmethod
    def _row(r: tuple[Any, ...]) -> Interrupt:
        (id_, run_id, request_id, kind, tool, arguments, status, assignee, level, correlation_key, created_at,
         escalate_at, expires_at, resolution, decided_by, history) = r
        return Interrupt(id=id_, run_id=run_id, request_id=request_id, kind=InterruptKind(kind), tool=tool,
                         arguments=json.loads(arguments), status=InterruptStatus(status), assignee=assignee,
                         level=level, correlation_key=correlation_key, created_at=created_at,
                         escalate_at=escalate_at, expires_at=expires_at,
                         resolution=json.loads(resolution) if resolution else None, decided_by=decided_by,
                         history=json.loads(history))


__all__ = [
    "InterruptKind", "InterruptStatus", "EscalationPolicy", "Interrupt", "InterruptManager", "WAIT_UNTIL",
    "WAIT_FOR_EVENT",
]
