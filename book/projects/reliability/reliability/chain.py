# path: book/projects/reliability/reliability/chain.py
"""Partial-failure semantics for multi-step chains.

Chapter 17's workflow engine owns graphs, typed state, and checkpoints. This module owns
one question the engine delegates: when step k of n fails, what does the caller get?

Each step declares whether it is *critical* (the result is meaningless without it) and,
optionally, a *fallback* that produces a weaker substitute. The runner then guarantees:

- A step never starts without enough budget left (`Deadline.check`), so an expired or
  cancelled request stops spending immediately; remaining steps are recorded as SKIPPED.
- A non-critical failure is recorded and the chain continues (status PARTIAL).
- A critical failure with a working fallback continues in degraded form (PARTIAL).
- A critical failure without one stops the chain (status FAILED) but still returns the
  state accumulated so far, so the caller can show or persist what was done.
- Retries happen only for steps marked `idempotent`, and draw on the shared RetryBudget.
- No exception escapes `run_chain` except programming errors in the runner itself.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from aie_core.llm.errors import MalformedResponseError
from aie_core.llm.gateway import RetryPolicy

from .circuit import CircuitBreakerRegistry
from .clock import Clock, Sleep
from .deadline import Deadline
from .errors import DeadlineExceeded
from .malformed import Recovered
from .retry import RetryBudget, call_with_retry

StepFn = Callable[[dict[str, Any], Deadline], Any]
FallbackFn = Callable[[dict[str, Any], BaseException], Any]


class StepStatus(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"     # step succeeded through output recovery (repair, fallback model, salvage)
    FALLBACK = "fallback"     # step failed; its fallback produced a substitute
    FAILED = "failed"
    SKIPPED = "skipped"


class ChainStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass
class Step:
    name: str
    run: StepFn
    critical: bool = True
    budget_s: float | None = None
    min_s: float = 0.0                    # do not start with less than this left
    fallback: FallbackFn | None = None
    dependency: str | None = None         # circuit breaker name for non-LLM dependencies
    idempotent: bool = False              # only idempotent steps are retried here


@dataclass
class StepRecord:
    name: str
    status: StepStatus
    duration_s: float = 0.0
    error: str | None = None
    detail: str | None = None


@dataclass
class ChainResult:
    status: ChainStatus
    state: dict[str, Any]
    steps: list[StepRecord] = field(default_factory=list)

    def step(self, name: str) -> StepRecord:
        return next(s for s in self.steps if s.name == name)

    @property
    def degraded_steps(self) -> list[str]:
        return [s.name for s in self.steps if s.status is not StepStatus.OK]


def _status_of(value: Any) -> tuple[StepStatus, str | None]:
    if isinstance(value, Recovered):
        return (StepStatus.DEGRADED if value.outcome.degraded else StepStatus.OK), value.outcome.value
    return StepStatus.OK, None


def run_chain(
    steps: list[Step],
    state: dict[str, Any],
    deadline: Deadline,
    *,
    breakers: CircuitBreakerRegistry | None = None,
    retry_policy: RetryPolicy | None = None,
    retry_budget: RetryBudget | None = None,
    clock: Clock = time.monotonic,
    sleep: Sleep = time.sleep,
) -> ChainResult:
    records: list[StepRecord] = []
    failed_critical = False
    degraded = False

    for i, step in enumerate(steps):
        started = clock()
        try:
            deadline.check(step.name, need_s=step.min_s)
        except DeadlineExceeded as exc:
            records += [StepRecord(s.name, StepStatus.SKIPPED, error=str(exc)) for s in steps[i:]]
            failed_critical = failed_critical or any(s.critical for s in steps[i:])
            break

        child = deadline.child(step.budget_s, name=step.name)

        def attempt(step: Step = step, child: Deadline = child) -> Any:
            if step.dependency and breakers is not None:
                return breakers.get(step.dependency).call(step.run, state, child)
            return step.run(state, child)

        try:
            if step.idempotent:
                value = call_with_retry(attempt, policy=retry_policy, budget=retry_budget, deadline=child, sleep=sleep)
            else:
                value = attempt()
            if isinstance(value, Recovered) and not value.usable:
                raise MalformedResponseError("; ".join(value.errors) or "no usable structured output")
            status, detail = _status_of(value)
            state[step.name] = value
            records.append(StepRecord(step.name, status, clock() - started, detail=detail))
            degraded = degraded or status is not StepStatus.OK
            continue
        except Exception as exc:  # noqa: BLE001 - every dependency failure is handled by policy
            error = f"{type(exc).__name__}: {exc}"
            if step.fallback is not None:
                try:
                    state[step.name] = step.fallback(state, exc)
                    records.append(StepRecord(step.name, StepStatus.FALLBACK, clock() - started, error=error))
                    degraded = True
                    continue
                except Exception as fb_exc:  # noqa: BLE001
                    error = f"{error}; fallback {type(fb_exc).__name__}: {fb_exc}"
            state[step.name] = None
            records.append(StepRecord(step.name, StepStatus.FAILED, clock() - started, error=error))
            if step.critical:
                failed_critical = True
                records += [StepRecord(s.name, StepStatus.SKIPPED, error=f"after critical failure of {step.name}")
                            for s in steps[i + 1:]]
                break
            degraded = True

    if failed_critical:
        status = ChainStatus.FAILED
    elif degraded:
        status = ChainStatus.PARTIAL
    else:
        status = ChainStatus.COMPLETE
    return ChainResult(status=status, state=state, steps=records)


__all__ = ["Step", "StepStatus", "StepRecord", "ChainStatus", "ChainResult", "run_chain"]
