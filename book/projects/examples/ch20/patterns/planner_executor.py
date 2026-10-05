# path: book/projects/examples/ch20/patterns/planner_executor.py
"""Planner-executor: a planner writes a bounded, typed plan; each step runs as its own small
agent with only the tools that step names; deterministic rules detect deviation and trigger a
replan, at most `max_replans` times. A final synthesis call writes the answer from the ledger.

The plan is data, so it can be validated before anything runs, shown to a human, logged,
and diffed between runs. "Replan on every step" is not the default: a replan is a model call
that can also make things worse, so it happens only when a rule names a concrete deviation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from pydantic import BaseModel, Field

from aie_core.llm.client import LLMClient
from aie_core.llm.errors import MalformedResponseError
from agentkit import Budget, DefinitionOfDone, EventStore, RunResult, non_empty, tool_was_called

from .common import SEP, PatternResult, ask, ask_structured, make_agent


class PlanStep(BaseModel):
    id: str = Field(pattern=r"^s[0-9]+$")
    objective: str = Field(min_length=8)
    tools: list[str] = Field(min_length=1, max_length=2)
    target: str | None = None                 # the entity the step is about, e.g. a service name
    depends_on: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    steps: list[PlanStep] = Field(min_length=1, max_length=8)
    assumptions: list[str] = Field(default_factory=list)


@dataclass
class StepOutcome:
    step: PlanStep
    run: RunResult

    @property
    def ok(self) -> bool:
        return self.run.ok

    @property
    def finding(self) -> str:
        return self.run.final_answer or f"(no finding: {self.run.stop_reason.value if self.run.stop_reason else '?'})"


# A deviation rule reads one finished step and the steps still queued; it returns a reason or None.
DeviationRule = Callable[[StepOutcome, list[PlanStep]], str | None]


def step_failed(outcome: StepOutcome, remaining: list[PlanStep]) -> str | None:
    if outcome.ok:
        return None
    return f"step {outcome.step.id} did not complete ({outcome.run.stop_reason.value if outcome.run.stop_reason else '?'})"


def empty_evidence(outcome: StepOutcome, remaining: list[PlanStep]) -> str | None:
    texts = [o.content for o in outcome.run.state.observations if o.ok]
    if texts and all(t.strip() in ("no results", "[]", "") for t in texts):
        return f"step {outcome.step.id} found no evidence; the plan's assumption about where to look was wrong"
    return None


def validate_plan(plan: Plan, allowed_tools: set[str]) -> list[str]:
    errors: list[str] = []
    ids = [s.id for s in plan.steps]
    if len(set(ids)) != len(ids):
        errors.append("step ids must be unique")
    for s in plan.steps:
        unknown = sorted(set(s.tools) - allowed_tools)
        if unknown:
            errors.append(f"{s.id} uses unknown tools {unknown}; allowed: {sorted(allowed_tools)}")
        missing = [d for d in s.depends_on if d not in ids]
        if missing:
            errors.append(f"{s.id} depends on unknown steps {missing}")
    return errors


PLANNER_SYSTEM = (
    "You plan investigations. Return a JSON plan of at most {max_steps} steps. Each step has an id "
    "(s1, s2, ...), one concrete objective that a single tool call can satisfy, the tool(s) it needs "
    "from {tools}, an optional target entity, and depends_on. Do not plan the final write-up."
)
EXECUTOR_INSTRUCTIONS = ("Execute exactly one plan step. Use only the tools you are given, then report the "
                         "finding in two or three sentences, citing every fact as [source-id].")


@dataclass
class PlannerExecutor:
    llm: LLMClient
    tools: Sequence[Any]
    deviation_rules: Sequence[DeviationRule] = (step_failed, empty_evidence)
    max_replans: int = 2
    max_steps: int = 6
    step_budget: Budget = field(default_factory=lambda: Budget(max_steps=3, max_tool_calls=2))
    store: EventStore | None = None
    principal: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self._tools = {t.name: t for t in self.tools}

    # ----------------------------------------------------------------- planning
    def plan(self, goal: str, done: list[StepOutcome], deviation: str | None) -> Plan:
        system = PLANNER_SYSTEM.format(max_steps=self.max_steps, tools=sorted(self._tools))
        user = f"Goal: {goal}"
        if done:
            ledger = "\n".join(f"- {o.step.id} ({','.join(o.step.tools)}): {o.finding}" for o in done)
            user += f"\n\nCompleted steps (keep their ids, do not repeat them):\n{ledger}"
        if deviation:
            user += f"\n\nDeviation detected: {deviation}\nRevise the remaining plan."
        plan: Any = ask_structured(self.llm, "planner", system, user, Plan)
        errors = validate_plan(plan, set(self._tools))
        if errors:   # one repair round with the concrete errors, then give up
            plan = ask_structured(self.llm, "planner", system, user + "\n\nYour plan was invalid: "
                                  + "; ".join(errors), Plan)
            errors = validate_plan(plan, set(self._tools))
            if errors:
                raise MalformedResponseError("planner produced an invalid plan: " + "; ".join(errors))
        return plan

    # ---------------------------------------------------------------- execution
    def execute(self, goal: str, step: PlanStep, done: list[StepOutcome], run_id: str) -> StepOutcome:
        context = "\n".join(f"- {o.step.id}: {o.finding}" for o in done if o.step.id in step.depends_on)
        task = (f"Overall goal: {goal}\nYour step ({step.id}): {step.objective}"
                + (f"\nTarget: {step.target}" if step.target else "")
                + (f"\nFindings you depend on:\n{context}" if context else ""))
        runtime = make_agent(self.llm, [self._tools[n] for n in step.tools], role="executor",
                             instructions=EXECUTOR_INSTRUCTIONS, budget=self.step_budget,
                             dod=DefinitionOfDone(tool_was_called(step.tools[0]), non_empty(10)),
                             store=self.store, principal=self.principal)
        return StepOutcome(step, runtime.run(task, run_id=run_id, metadata={"plan_step": step.id}))

    def run(self, goal: str, *, run_id: str = "pe") -> PatternResult:
        try:
            plan = self.plan(goal, [], None)
        except MalformedResponseError as exc:
            return PatternResult("planner_executor", False, None, detail=str(exc))
        plans, done, replans = [plan], [], 0
        queue = list(plan.steps)
        while queue:
            if len(done) >= self.max_steps:
                return self._result(False, None, done, plans, f"executed {len(done)} steps; step cap reached")
            step = queue.pop(0)
            outcome = self.execute(goal, step, done, f"{run_id}{SEP}{step.id}-{len(done) + 1}")
            done.append(outcome)
            reason = next((r for rule in self.deviation_rules if (r := rule(outcome, queue))), None)
            if reason is None:
                continue
            if replans >= self.max_replans:
                return self._result(False, None, done, plans, f"replan budget exhausted after: {reason}")
            replans += 1
            try:
                plan = self.plan(goal, done, reason)
            except MalformedResponseError as exc:
                return self._result(False, None, done, plans, f"replanning failed: {exc}")
            plans.append(plan)
            finished = {o.step.id for o in done}
            queue = [s for s in plan.steps if s.id not in finished]
        findings = "\n".join(f"- {o.step.id}: {o.finding}" for o in done)
        answer = ask(self.llm, "synthesizer", "Write the final answer from the findings only. Keep every "
                     "[source-id] citation attached to the fact it supports.", f"Goal: {goal}\nFindings:\n{findings}")
        return self._result(all(o.ok for o in done[-1:]), answer, done, plans, f"{len(done)} steps, {replans} replans")

    def _result(self, ok: bool, answer: str | None, done: list[StepOutcome], plans: list[Plan],
                detail: str) -> PatternResult:
        return PatternResult("planner_executor", ok, answer, [o.run for o in done], detail=detail,
                             data={"plans": [p.model_dump() for p in plans], "replans": len(plans) - 1,
                                   "ledger": [{"step": o.step.id, "ok": o.ok, "finding": o.finding} for o in done]})


__all__ = ["PlanStep", "Plan", "StepOutcome", "DeviationRule", "step_failed", "empty_evidence", "validate_plan",
           "PlannerExecutor"]
