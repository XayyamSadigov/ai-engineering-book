# path: book/projects/examples/ch20/patterns/parallel.py
"""Parallel agents: fan out independent sub-tasks to bounded agents, fan in under an explicit
completeness policy.

The design decision lives in the fan-in. `require="all"` refuses to answer if any branch
failed; `"quorum"` needs a strict majority; `"any"` answers from whatever succeeded but says
which branches are missing. AgentRuntime is synchronous, so branches run in a thread pool;
the pool size is the concurrency limit you set to respect provider rate limits.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

from aie_core.llm.client import LLMClient
from agentkit import Budget, DefinitionOfDone, EventStore, RunResult

from .common import SEP, PatternResult, ask, make_agent


@dataclass
class Branch:
    name: str
    goal: str
    tools: Sequence[Any]
    instructions: str = "Complete this one sub-task with the tools given; cite facts as [source-id]."
    budget: Budget = field(default_factory=lambda: Budget(max_steps=4, max_tool_calls=3))
    dod: DefinitionOfDone | None = None


Merge = Callable[[list[tuple[Branch, RunResult]], list[str]], str]
Require = Literal["all", "quorum", "any"]


def _enough(ok: int, total: int, require: Require) -> bool:
    return total > 0 and {"all": ok == total, "quorum": ok * 2 > total, "any": ok >= 1}[require]


def fan_out(llm: LLMClient, request: str, branches: Sequence[Branch], *, require: Require = "quorum",
            max_workers: int = 4, merge: Merge | None = None, store: EventStore | None = None,
            run_id: str = "par", principal: dict[str, Any] | None = None) -> PatternResult:
    def run_branch(b: Branch) -> RunResult:
        runtime = make_agent(llm, b.tools, role=f"branch:{b.name}", instructions=b.instructions, budget=b.budget,
                             dod=b.dod, store=store, principal=principal)
        return runtime.run(b.goal, run_id=f"{run_id}{SEP}{b.name}", metadata={"parent_run_id": run_id})

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(run_branch, b) for b in branches]
    results: list[tuple[Branch, RunResult | None, str]] = []
    for b, f in zip(branches, futures):           # results in branch order, never completion order
        try:
            results.append((b, f.result(), ""))
        except Exception as exc:  # noqa: BLE001 - a crashed branch is a failed branch, recorded with its cause
            results.append((b, None, f"{type(exc).__name__}: {exc}"))
    succeeded = [(b, r) for b, r, _ in results if r is not None and r.ok]
    missing = [b.name for b, r, _ in results if r is None or not r.ok]
    runs = [r for _, r, _ in results if r is not None]
    status = {b.name: (r.stop_reason.value if r and r.stop_reason else err) for b, r, err in results}
    data = {"branches": status, "missing": missing, "require": require}
    if not _enough(len(succeeded), len(branches), require):
        return PatternResult("parallel", False, None, runs,
                             detail=f"{len(succeeded)}/{len(branches)} branches succeeded; require={require}", data=data)
    answer = merge(succeeded, missing) if merge else _synthesize(llm, request, succeeded, missing)
    return PatternResult("parallel", True, answer, runs, detail=f"{len(succeeded)}/{len(branches)} branches", data=data)


def _synthesize(llm: LLMClient, request: str, ok: list[tuple[Branch, RunResult]], missing: list[str]) -> str:
    parts = "\n".join(f"## {b.name}\n{r.final_answer}" for b, r in ok)
    gap = f"\nThese branches failed and must be reported as missing: {missing}" if missing else ""
    return ask(llm, "aggregator", "Merge the branch results into one answer. Keep [source-id] citations. If "
               "branches disagree, say so and show both; do not pick silently." + gap,
               f"Request: {request}\n{parts}")


__all__ = ["Branch", "Merge", "Require", "fan_out"]
