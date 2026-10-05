# path: book/projects/examples/ch20/patterns/hierarchical.py
"""Hierarchical agents: supervisors whose members are other supervisors.

Nothing new is needed beyond supervisor.py: a Supervisor is a valid member of another
Supervisor. What the hierarchy adds is *shared* limits (one SpawnBudget and one event store
for the whole tree), run ids that encode the path from the root, and a ledger that rolls up
from every level. `build_tree` turns a nested description into supervisors and workers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Union

from aie_core.llm.client import LLMClient
from agentkit import Budget, EventStore

from .common import SEP, PatternResult
from .supervisor import SpawnBudget, Supervisor, Worker


@dataclass
class Team:
    name: str
    description: str
    members: Sequence[Union["Team", Worker]]
    budget: Budget = field(default_factory=lambda: Budget(max_steps=5, max_tool_calls=4))


def build_tree(llm: LLMClient, team: Team, *, spawn: SpawnBudget | None = None,
               store: EventStore | None = None) -> Supervisor:
    members = [build_tree(llm, m, spawn=spawn, store=store) if isinstance(m, Team) else m for m in team.members]
    return Supervisor(llm, members, name=team.name, description=team.description, budget=team.budget,
                      spawn=spawn, store=store)


def run_hierarchy(llm: LLMClient, team: Team, request: str, *, spawn: SpawnBudget | None = None,
                  store: EventStore | None = None) -> PatternResult:
    spawn = spawn or SpawnBudget(max_agents=10, max_depth=3)
    root = build_tree(llm, team, spawn=spawn, store=store)
    result = root.run(request)
    result.pattern = "hierarchical"
    depth = max((r.run_id.count(SEP) for r in result.runs), default=0)
    result.data["max_depth_reached"] = depth
    return result


__all__ = ["Team", "build_tree", "run_hierarchy"]
