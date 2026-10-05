# path: book/projects/examples/ch20/patterns/sequential.py
"""Sequential workflows: a prompt chain whose steps may be plain model calls, agents, or code,
with a typed state dict passed between them and an optional gate after each step.

The order is fixed by the code, not chosen by a model. An agent sits inside one step, where
the work is open-ended, and the chain around it stays deterministic and testable (Chapter 17).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

from pydantic import BaseModel

from aie_core.llm.client import LLMClient
from agentkit import Budget, DefinitionOfDone, EventStore, RunResult

from .common import PatternResult, ask, ask_structured, make_agent

State = dict[str, Any]
Gate = Callable[[State], str | None]   # returns a problem or None


@dataclass
class Step:
    name: str
    kind: Literal["llm", "agent", "code"]
    fn: Callable[[State], State]
    gate: Gate | None = None


@dataclass
class ChainContext:
    runs: list[RunResult] = field(default_factory=list)


def llm_step(llm: LLMClient, name: str, system: str, render: Callable[[State], str], output: str, *,
             schema: type[BaseModel] | None = None, gate: Gate | None = None) -> Step:
    def fn(state: State) -> State:
        user = render(state)
        value = ask_structured(llm, name, system, user, schema) if schema else ask(llm, name, system, user)
        return {**state, output: value}
    return Step(name, "llm", fn, gate)


def agent_step(llm: LLMClient, ctx: ChainContext, name: str, tools: Sequence[Any], instructions: str,
               goal: Callable[[State], str], output: str, *, budget: Budget | None = None,
               dod: DefinitionOfDone | None = None, store: EventStore | None = None,
               gate: Gate | None = None) -> Step:
    def fn(state: State) -> State:
        runtime = make_agent(llm, tools, role=name, instructions=instructions, budget=budget, dod=dod, store=store)
        run = runtime.run(goal(state))
        ctx.runs.append(run)
        if not run.ok:
            raise StepFailed(name, f"agent stopped with {run.stop_reason.value if run.stop_reason else '?'}: {run.detail}")
        return {**state, output: run.final_answer}
    return Step(name, "agent", fn, gate)


def code_step(name: str, fn: Callable[[State], State], *, gate: Gate | None = None) -> Step:
    return Step(name, "code", fn, gate)


class StepFailed(Exception):
    def __init__(self, step: str, reason: str) -> None:
        super().__init__(f"{step}: {reason}")
        self.step, self.reason = step, reason


def run_chain(steps: Sequence[Step], state: State, ctx: ChainContext | None = None,
              *, answer_key: str = "answer") -> PatternResult:
    ctx = ctx or ChainContext()
    path: list[str] = []
    for step in steps:
        try:
            state = step.fn(state)
        except StepFailed as exc:
            return PatternResult("sequential", False, None, ctx.runs, detail=str(exc), data={"path": path, "state": state})
        path.append(step.name)
        problem = step.gate(state) if step.gate else None
        if problem:
            return PatternResult("sequential", False, None, ctx.runs, detail=f"gate after {step.name}: {problem}",
                                 data={"path": path, "state": state})
    answer = state.get(answer_key)
    return PatternResult("sequential", True, answer if isinstance(answer, str) else str(answer), ctx.runs,
                         detail=" -> ".join(path), data={"path": path, "state": state})


__all__ = ["State", "Gate", "Step", "ChainContext", "llm_step", "agent_step", "code_step", "StepFailed", "run_chain"]
