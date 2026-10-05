# path: book/projects/examples/ch20/patterns/react.py
"""ReAct: interleave a decision, an action, and an observation until done.

AgentRuntime already *is* a ReAct loop with the safety parts built in (validation, budgets,
repeat and no-progress detection, Definition of Done). The pattern-level work is choosing the
tools, the instructions that keep the objective visible, and the stop criteria.
"""
from __future__ import annotations

from typing import Any, Sequence

from aie_core.llm.client import LLMClient
from agentkit import Budget, DefinitionOfDone, EventStore, LoopConfig

from .common import PatternResult, make_agent

REACT_INSTRUCTIONS = (
    "Work in short cycles: decide what you still need to know, call one tool, read the result. "
    "Keep the objective, what you have learned, and what is still unknown in mind at every step. "
    "Stop as soon as the evidence answers the objective; cite every fact as [source-id]."
)


def react(
    llm: LLMClient,
    goal: str,
    tools: Sequence[Any],
    *,
    budget: Budget | None = None,
    dod: DefinitionOfDone | None = None,
    store: EventStore | None = None,
    config: LoopConfig | None = None,
    principal: dict[str, Any] | None = None,
) -> PatternResult:
    runtime = make_agent(llm, tools, role="react", instructions=REACT_INSTRUCTIONS,
                         budget=budget or Budget(max_steps=6, max_tool_calls=6), dod=dod, store=store,
                         config=config, principal=principal)
    run = runtime.run(goal)
    return PatternResult("react", run.ok, run.final_answer, [run],
                         detail=run.stop_reason.value if run.stop_reason else "")


__all__ = ["REACT_INSTRUCTIONS", "react"]
