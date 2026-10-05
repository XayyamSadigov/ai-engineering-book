# path: book/projects/examples/ch20/patterns/supervisor.py
"""Supervisor/worker: a supervisor agent whose only tools are "delegate to worker X".

Each delegation starts a bounded child AgentRuntime with its own tools, prompt, and budget.
The supervisor sees only the child's final answer, never its intermediate tokens. A task
ledger records every delegation; a shared SpawnBudget caps the number of agents across the
whole tree, so a confused supervisor cannot spawn without limit. Child run ids are derived
from the parent's ("parent.worker-n"), which is the trace propagation rule.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Sequence, Union

from aie_core.llm.client import LLMClient
from agentkit import (
    Budget, DefinitionOfDone, ErrorClass, EventStore, FunctionTool, InMemoryEventStore, RunResult, ToolContext,
    ToolOutput,
)

from .common import SEP, PatternResult, make_agent, obj


@dataclass
class Worker:
    name: str
    description: str
    tools: Sequence[Any]
    instructions: str
    budget: Budget = field(default_factory=lambda: Budget(max_steps=4, max_tool_calls=3))
    dod: DefinitionOfDone | None = None


class SpawnBudget:
    """Limits shared by every supervisor in one tree: total agents and nesting depth."""

    def __init__(self, max_agents: int = 8, max_depth: int = 2) -> None:
        self.max_agents = max_agents
        self.max_depth = max_depth
        self.spawned = 0
        self._lock = threading.Lock()

    def acquire(self, depth: int) -> str | None:
        with self._lock:
            if depth > self.max_depth:
                return f"maximum delegation depth {self.max_depth} reached"
            if self.spawned >= self.max_agents:
                return f"agent budget of {self.max_agents} exhausted"
            self.spawned += 1
            return None


@dataclass
class LedgerEntry:
    parent_run_id: str
    member: str
    task: str
    run_id: str
    ok: bool
    stop_reason: str
    answer: str | None


SUPERVISOR_INSTRUCTIONS = (
    "You coordinate specialists. Break the request into sub-tasks, delegate each to the one specialist "
    "whose description fits, and never delegate the same sub-task twice. Each sub-task must be "
    "self-contained: the specialist sees only what you write. When the results answer the request, "
    "combine them into one answer and keep every [source-id] citation."
)

Member = Union[Worker, "Supervisor"]


class Supervisor:
    def __init__(self, llm: LLMClient, members: Sequence[Member], *, name: str = "supervisor",
                 description: str = "", instructions: str = SUPERVISOR_INSTRUCTIONS,
                 budget: Budget | None = None, dod: DefinitionOfDone | None = None,
                 spawn: SpawnBudget | None = None, store: EventStore | None = None,
                 max_delegations_per_member: int = 2) -> None:
        self.llm = llm
        self.members = {m.name: m for m in members}
        self.name = name
        self.description = description
        self.instructions = instructions
        self.budget = budget or Budget(max_steps=6, max_tool_calls=5)
        self.dod = dod
        self.spawn = spawn or SpawnBudget()
        self.store = store if store is not None else InMemoryEventStore()
        self.max_delegations = max_delegations_per_member
        self.ledger: list[LedgerEntry] = []
        self.child_runs: list[RunResult] = []
        for m in members:                 # one tree, one spawn budget, one event store
            if isinstance(m, Supervisor):
                m.spawn, m.store = self.spawn, self.store

    # -------------------------------------------------------------- delegation
    def _delegate_tool(self, member: Member) -> FunctionTool:
        def delegate(ctx: ToolContext, task: str) -> ToolOutput:
            used = sum(1 for e in self.ledger if e.member == member.name and e.parent_run_id == ctx.run_id)
            if used >= self.max_delegations:
                return ToolOutput.failure(f"{member.name} already received {used} tasks in this run",
                                          ErrorClass.PERMISSION)
            depth = ctx.run_id.count(SEP) + 1
            refused = self.spawn.acquire(depth)
            if refused:
                return ToolOutput.failure(refused, ErrorClass.PERMISSION)
            child_id = f"{ctx.run_id}{SEP}{member.name}-{used + 1}"
            run = self._run_member(member, task, child_id, ctx)
            self.ledger.append(LedgerEntry(ctx.run_id, member.name, task, child_id, run.ok,
                                           run.stop_reason.value if run.stop_reason else "", run.final_answer))
            if not run.ok:
                return ToolOutput.failure(f"{member.name} stopped with {run.stop_reason.value if run.stop_reason else '?'}:"
                                          f" {run.detail}", ErrorClass.SEMANTIC)
            return ToolOutput(content=f"[{member.name} result] {run.final_answer}", data={"child_run_id": child_id})

        return FunctionTool(f"delegate_{member.name}", f"Delegate one self-contained sub-task to {member.name}: "
                            f"{member.description}", obj({"task": {"type": "string", "minLength": 10}}, ["task"]),
                            delegate, pass_context=True)

    def _run_member(self, member: Member, task: str, child_id: str, ctx: ToolContext) -> RunResult:
        meta = {"parent_run_id": ctx.run_id, "parent_request_id": ctx.request_id}
        if isinstance(member, Supervisor):
            run = member.runtime(ctx.principal).run(task, run_id=child_id, metadata=meta)
            self.ledger.extend(member.ledger)
            self.child_runs.extend(member.child_runs)
            member.ledger, member.child_runs = [], []
        else:
            runtime = make_agent(self.llm, member.tools, role=f"worker:{member.name}", instructions=member.instructions,
                                 budget=member.budget, dod=member.dod, store=self.store, principal=ctx.principal)
            run = runtime.run(task, run_id=child_id, metadata=meta)
        self.child_runs.append(run)
        return run

    # -------------------------------------------------------------------- run
    def runtime(self, principal: dict[str, Any] | None = None):
        tools = [self._delegate_tool(m) for m in self.members.values()]
        return make_agent(self.llm, tools, role=f"supervisor:{self.name}", instructions=self.instructions,
                          budget=self.budget, dod=self.dod, store=self.store, principal=principal)

    def run(self, request: str, *, run_id: str | None = None, principal: dict[str, Any] | None = None) -> PatternResult:
        root = self.runtime(principal).run(request, run_id=run_id or self.name)
        return PatternResult("supervisor", root.ok, root.final_answer, [root, *self.child_runs],
                             detail=root.stop_reason.value if root.stop_reason else "",
                             data={"ledger": [e.__dict__ for e in self.ledger], "agents_spawned": self.spawn.spawned})


__all__ = ["Worker", "SpawnBudget", "LedgerEntry", "Supervisor", "SUPERVISOR_INSTRUCTIONS"]
