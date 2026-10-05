# path: book/projects/p6-research-team/research_team/ledger.py
"""Team-level control: the global budget, the spawn cap, duplicate detection, and the
coordination log that links every child run to its parent.

Budget propagation is reserve-then-settle. Before a child starts, the supervisor reserves
its whole slice from the global pool; when the child ends, actual usage is charged and the
unused part returns to the pool. The sum of outstanding reservations plus spending can never
exceed the global limit, so parallel children cannot jointly overshoot it.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

from .contracts import BudgetSlice, TaskEnvelope


class TeamBudget(BaseModel):
    """Limits for one team run. Numbers are illustrative defaults for the Northwind corpus."""

    model_config = ConfigDict(frozen=True)

    max_tokens: int = Field(default=80_000, ge=1)
    max_cost_usd: float | None = Field(default=None, gt=0)
    deadline_s: float = Field(default=120.0, gt=0)
    max_children: int = Field(default=6, ge=1)        # spawn cap: all children over the whole run
    max_depth: int = Field(default=1, ge=0)           # 1 = workers may not spawn workers
    max_parallel: int = Field(default=4, ge=1)
    child: BudgetSlice = BudgetSlice()                # requested slice per researcher
    min_child_tokens: int = Field(default=4_000, ge=1)   # below this a child cannot finish; do not start it
    synthesis_reserve_tokens: int = Field(default=10_000, ge=0)  # held back so the answer can be written


@dataclass(frozen=True)
class Admission:
    admitted: bool
    reason: str                     # "ok", "spawn_cap", "max_depth", "duplicate", "budget", "deadline"
    granted: BudgetSlice | None = None


class BudgetLedger:
    def __init__(self, limits: TeamBudget, clock: Callable[[], float] = time.monotonic) -> None:
        self.limits = limits
        self.clock = clock
        self.started = clock()
        self._lock = threading.Lock()
        self._reserved: dict[str, int] = {}
        self._reserved_cost: dict[str, float] = {}
        self.spent_tokens = 0
        self.spent_cost = 0.0
        self.children = 0
        self._keys: dict[str, str] = {}            # objective key -> task id

    # ------------------------------------------------------------------ queries
    def elapsed(self) -> float:
        return self.clock() - self.started

    def time_left(self) -> float:
        return self.limits.deadline_s - self.elapsed()

    def available_tokens(self) -> int:
        with self._lock:
            return self._available_tokens()

    def _available_tokens(self) -> int:
        return self.limits.max_tokens - self.spent_tokens - sum(self._reserved.values())

    def _available_cost(self) -> float | None:
        if self.limits.max_cost_usd is None:
            return None
        return self.limits.max_cost_usd - self.spent_cost - sum(self._reserved_cost.values())

    # ------------------------------------------------------------------ admission
    def admit(self, env: TaskEnvelope, *, counts_as_child: bool = True) -> Admission:
        """Decide whether a task may start and reserve its slice. Order: structure, then money."""
        with self._lock:
            if counts_as_child:
                if env.depth > self.limits.max_depth:
                    return Admission(False, "max_depth")
                if self.children >= self.limits.max_children:
                    return Admission(False, "spawn_cap")
                if env.objective_key in self._keys:
                    return Admission(False, "duplicate")
            left = self.limits.deadline_s - self.elapsed()
            if left <= 1.0:
                return Admission(False, "deadline")
            tokens = min(env.budget.max_tokens, self._available_tokens())
            if tokens < self.limits.min_child_tokens:
                return Admission(False, "budget")
            cost_left = self._available_cost()
            cost = env.budget.max_cost_usd
            if cost_left is not None:
                if cost_left <= 0:
                    return Admission(False, "budget")
                cost = min(cost or cost_left, cost_left)
            deadline = min(env.budget.deadline_s or left, left)
            granted = env.budget.model_copy(update={"max_tokens": tokens, "max_cost_usd": cost,
                                                    "deadline_s": deadline})
            self._reserved[env.task_id] = tokens
            self._reserved_cost[env.task_id] = cost or 0.0
            if counts_as_child:
                self.children += 1
                self._keys[env.objective_key] = env.task_id
            return Admission(True, "ok", granted)

    def hold(self, name: str, tokens: int) -> int:
        """Reserve tokens for a later phase (synthesis) so children cannot consume them."""
        with self._lock:
            amount = max(0, min(tokens, self._available_tokens()))
            self._reserved[name] = amount
            return amount

    def release(self, name: str) -> None:
        with self._lock:
            self._reserved.pop(name, None)
            self._reserved_cost.pop(name, None)

    def settle(self, task_id: str, tokens: int, cost_usd: float) -> None:
        """Charge actual usage and return the unused part of the reservation to the pool."""
        with self._lock:
            self._reserved.pop(task_id, None)
            self._reserved_cost.pop(task_id, None)
            self.spent_tokens += tokens
            self.spent_cost += cost_usd

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"spent_tokens": self.spent_tokens, "spent_cost_usd": round(self.spent_cost, 6),
                    "reserved_tokens": sum(self._reserved.values()), "available_tokens": self._available_tokens(),
                    "children": self.children, "elapsed_s": round(self.elapsed(), 3)}


# ----------------------------------------------------------------------------- coordination log
class TeamEvent(BaseModel):
    """One coordination fact. Agent internals live in each child's agentkit log (run_id)."""

    seq: int
    ts: float
    trace_id: str
    kind: str        # team_started, task_dispatched, spawn_refused, task_finished, verified, conflict, team_finished
    task_id: str | None = None
    parent_id: str | None = None
    run_id: str | None = None
    span_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class TeamLog:
    """Append-only, thread-safe. Writes JSONL when given a path, always keeps events in memory."""

    def __init__(self, trace_id: str, path: str | Path | None = None) -> None:
        self.trace_id = trace_id
        self.path = Path(path) if path else None
        self.events: list[TeamEvent] = []
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, kind: str, **fields: Any) -> TeamEvent:
        with self._lock:
            ev = TeamEvent(seq=len(self.events), ts=time.time(), trace_id=self.trace_id, kind=kind, **fields)
            self.events.append(ev)
            if self.path:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(ev.model_dump_json() + "\n")
            return ev

    def of(self, kind: str) -> list[TeamEvent]:
        return [e for e in self.events if e.kind == kind]

    @staticmethod
    def load(path: str | Path) -> list[TeamEvent]:
        return [TeamEvent.model_validate(json.loads(line)) for line in Path(path).read_text().splitlines() if line]


__all__ = ["Admission", "BudgetLedger", "TeamBudget", "TeamEvent", "TeamLog"]
