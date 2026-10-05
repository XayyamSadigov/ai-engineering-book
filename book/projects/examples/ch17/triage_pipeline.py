# path: book/projects/examples/ch17/triage_pipeline.py
"""Ticket triage as a fixed pipeline: classify -> retrieve policy -> draft ->
validate -> approve/send. Control flow is ordinary Python. There is no
engine, no graph, and therefore no way to pause: the approver must answer
synchronously, inside the request.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from triage_domain import (FakeModel, Model, Sender, TriageState, classify, draft_reply, escalate,
                           needs_human, retrieve_policy, send, validate)

MAX_DRAFTS = 2


@dataclass
class PipelineMetrics:
    step_ms: dict[str, float] = field(default_factory=dict)
    model_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return sum(self.step_ms.values())

    @property
    def orchestration_overhead_ms(self) -> float:
        """Wall time not spent waiting on the model: our code, serialization, I/O."""
        return self.total_ms - self.model_ms


def run_pipeline(state: TriageState, model: Model, approver: Callable[[TriageState], bool],
                 sender: Sender,
                 metrics: PipelineMetrics | None = None) -> TriageState:
    metrics = metrics if metrics is not None else PipelineMetrics()

    def timed(name: str, fn: Callable[[], TriageState]) -> TriageState:
        t0 = time.perf_counter()
        result = fn()
        metrics.step_ms[name] = metrics.step_ms.get(name, 0.0) + (time.perf_counter() - t0) * 1000
        return result

    state = timed("classify", lambda: classify(state, model))
    state = timed("retrieve_policy", lambda: retrieve_policy(state))

    while True:
        state = timed("draft", lambda: draft_reply(state, model))
        state = timed("validate", lambda: validate(state, model))
        assert state.report is not None
        if state.report.verdict == "ok":
            break
        if state.report.verdict == "escalate" or state.draft_attempts >= MAX_DRAFTS:
            return timed("escalate", lambda: escalate(state))

    if needs_human(state):
        decision = timed("approve", lambda: _apply_decision(state, approver(state)))
        if not decision.approved:
            return timed("escalate", lambda: escalate(state))

    if isinstance(model, FakeModel):
        metrics.model_ms = sum(c.duration_ms for c in model.calls)
    return timed("send", lambda: send(state, sender))


def _apply_decision(state: TriageState, approved: bool) -> TriageState:
    state.approved = approved
    state.log.append(f"approval: {'yes' if approved else 'no'}")
    return state
