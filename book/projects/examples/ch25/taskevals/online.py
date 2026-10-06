# path: book/projects/examples/ch25/taskevals/online.py
"""Online evaluation: join user feedback and delayed labels to traces, compute outcome metrics
per version, turn corrections into regression cases, and compare a canary with control using
a sequential decision rule that stays honest under repeated looks.

Feedback arrives late, sparsely, and from a biased subset of users. Joining it to the trace id
of the request that produced the output is what makes it attributable to a prompt, model, and
index version (Chapter 31 owns the trace schema); without the join, a thumbs-down is a mood.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timedelta
from statistics import NormalDist
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from evalkit import EvalCase


class TraceRecord(BaseModel):
    trace_id: str
    timestamp: datetime
    task: str
    tenant: str
    versions: dict[str, str]  # prompt, model, index, agent ...
    input: Any
    output: dict[str, Any]


class FeedbackEvent(BaseModel):
    trace_id: str
    timestamp: datetime
    kind: Literal["thumbs_up", "thumbs_down", "correction", "escalation", "label"]
    field: str | None = None  # for corrections and labels
    value: Any = None
    actor: str = "user"


class Outcome(BaseModel):
    trace: TraceRecord
    thumbs: Literal["up", "down"] | None = None
    corrections: dict[str, Any] = Field(default_factory=dict)
    escalated: bool = False
    labels: dict[str, Any] = Field(default_factory=dict)  # delayed ground truth, e.g. the agent's final category

    @property
    def has_feedback(self) -> bool:
        return self.thumbs is not None or bool(self.corrections) or self.escalated or bool(self.labels)


class JoinStats(BaseModel):
    traces: int
    events: int
    joined: int
    orphan_events: int  # feedback whose trace was not found: broken propagation or sampling
    late_events: int  # outside the attribution window; dropped rather than silently mixed in
    traces_with_feedback: int


def join_feedback(
    traces: Iterable[TraceRecord],
    events: Iterable[FeedbackEvent],
    *,
    window: timedelta = timedelta(days=14),
) -> tuple[list[Outcome], JoinStats]:
    by_id = {t.trace_id: Outcome(trace=t) for t in traces}
    joined = orphans = late = n_events = 0
    for ev in sorted(events, key=lambda e: e.timestamp):  # last event of a kind wins
        n_events += 1
        out = by_id.get(ev.trace_id)
        if out is None:
            orphans += 1
            continue
        if ev.timestamp - out.trace.timestamp > window or ev.timestamp < out.trace.timestamp:
            late += 1
            continue
        joined += 1
        if ev.kind in ("thumbs_up", "thumbs_down"):
            out.thumbs = "up" if ev.kind == "thumbs_up" else "down"
        elif ev.kind == "correction" and ev.field:
            out.corrections[ev.field] = ev.value
        elif ev.kind == "escalation":
            out.escalated = True
        elif ev.kind == "label" and ev.field:
            out.labels[ev.field] = ev.value
    outcomes = list(by_id.values())
    return outcomes, JoinStats(traces=len(outcomes), events=n_events, joined=joined, orphan_events=orphans,
                               late_events=late, traces_with_feedback=sum(o.has_feedback for o in outcomes))


class OutcomeMetrics(BaseModel):
    n: int
    feedback_coverage: float  # share of traces with any feedback: the denominator problem
    negative_rate: float | None  # thumbs-down share of rated traces
    correction_rate: float  # share of all traces corrected
    escalation_rate: float
    labeled_n: int
    labeled_accuracy: float | None  # output field vs delayed label, on labeled traces only


def outcome_metrics(outcomes: Sequence[Outcome], *, group_by: str = "prompt",
                    field: str = "category") -> dict[str, OutcomeMetrics]:
    groups: dict[str, list[Outcome]] = {}
    for o in outcomes:
        groups.setdefault(o.trace.versions.get(group_by, "unknown"), []).append(o)
    out = {}
    for version, rows in sorted(groups.items()):
        rated = [o for o in rows if o.thumbs is not None]
        labeled = [o for o in rows if field in o.labels]
        out[version] = OutcomeMetrics(
            n=len(rows),
            feedback_coverage=sum(o.has_feedback for o in rows) / len(rows),
            negative_rate=(sum(o.thumbs == "down" for o in rated) / len(rated)) if rated else None,
            correction_rate=sum(bool(o.corrections) for o in rows) / len(rows),
            escalation_rate=sum(o.escalated for o in rows) / len(rows),
            labeled_n=len(labeled),
            labeled_accuracy=(sum(o.trace.output.get(field) == o.labels[field] for o in labeled) / len(labeled))
            if labeled else None,
        )
    return out


def corrections_to_cases(outcomes: Iterable[Outcome], *, field: str = "category",
                         redact: Callable[[Any], Any]) -> list[EvalCase]:
    """Every human correction becomes a candidate regression case. `redact` is required and is
    applied to the production input, so raw personal data never lands in an eval set by default."""
    cases = []
    for o in outcomes:
        truth = o.labels.get(field, o.corrections.get(field))
        if truth is None or truth == o.trace.output.get(field):
            continue
        cases.append(EvalCase(
            id=f"PROD-{o.trace.trace_id}",
            input=redact(o.trace.input),
            expected={field: truth},
            tags=["origin:production-correction", f"tenant:{o.trace.tenant}", f"{field}:{truth}"],
            metadata={"trace_id": o.trace.trace_id, "versions": o.trace.versions, "group": o.trace.trace_id,
                      "review": "pending", "observed": o.trace.output.get(field)},
        ))
    return cases


# ------------------------------------------------------------------ canary comparison
class ArmCounts(BaseModel):
    n: int
    failures: int  # bad outcomes: corrections, thumbs-down, escalations, task failures
    critical: int = 0  # safety or permission violations: never averaged

    @property
    def rate(self) -> float:
        return self.failures / self.n if self.n else 0.0


Decision = Literal["continue", "rollback", "promote", "hold"]


class CanaryDecision(BaseModel):
    look: int
    decision: Decision
    reason: str
    control_rate: float
    canary_rate: float
    diff: float
    z: float | None = None
    upper_bound: float | None = None


class CanaryMonitor:
    """Sequential canary rule with a fixed number of planned looks.

    Looking at a running comparison many times with a fixed 5% test inflates the false-alarm
    rate far above 5%. This rule pays for the looks up front: with K planned looks, each look
    uses alpha/K (Bonferroni), which is conservative but simple and correct.

    - Any critical event in the canary: rollback immediately, no statistics needed.
    - Before `min_n` per arm: continue.
    - At every look: rollback if the canary's failure rate is significantly higher (one-sided).
    - At the final look: promote only if non-inferiority holds, that is the upper confidence bound
      on (canary - control) is below `margin`; otherwise hold for a human decision.
    """

    def __init__(self, *, looks: int = 5, alpha: float = 0.05, margin: float = 0.01, min_n: int = 200) -> None:
        if looks < 1:
            raise ValueError("looks must be >= 1")
        self.looks = looks
        self.alpha = alpha
        self.margin = margin
        self.min_n = min_n
        self.z_crit = NormalDist().inv_cdf(1 - alpha / looks)
        self.look = 0

    def observe(self, control: ArmCounts, canary: ArmCounts) -> CanaryDecision:
        self.look += 1
        diff = canary.rate - control.rate
        base = dict(look=self.look, control_rate=control.rate, canary_rate=canary.rate, diff=diff)
        if canary.critical > 0:
            return CanaryDecision(decision="rollback", reason=f"{canary.critical} critical event(s) in canary", **base)
        if min(control.n, canary.n) < self.min_n:
            return CanaryDecision(decision="continue", reason=f"fewer than {self.min_n} per arm", **base)
        se = math.sqrt(control.rate * (1 - control.rate) / control.n + canary.rate * (1 - canary.rate) / canary.n)
        se = max(se, 1e-9)
        z = diff / se
        upper = diff + self.z_crit * se
        if z > self.z_crit:
            return CanaryDecision(decision="rollback", reason="canary failure rate significantly higher",
                                  z=z, upper_bound=upper, **base)
        if self.look >= self.looks:
            if upper < self.margin:
                return CanaryDecision(decision="promote", reason=f"non-inferior within margin {self.margin}",
                                      z=z, upper_bound=upper, **base)
            return CanaryDecision(decision="hold", reason="inconclusive at final look: cannot rule out harm",
                                  z=z, upper_bound=upper, **base)
        return CanaryDecision(decision="continue", reason="no significant harm yet", z=z, upper_bound=upper, **base)


def simulate_false_alarms(
    *,
    p: float = 0.1,
    looks: int = 10,
    n_per_look: int = 300,
    sims: int = 400,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """A/A simulation: both arms identical. Share of runs that would wrongly roll back
    under naive peeking (z > z_{1-alpha} at any look) and under CanaryMonitor."""
    rng = np.random.default_rng(seed)
    z_naive = NormalDist().inv_cdf(1 - alpha)
    naive = bonf = 0
    for _ in range(sims):
        fails = rng.binomial(n_per_look, p, size=(looks, 2)).cumsum(axis=0)
        mon = CanaryMonitor(looks=looks, alpha=alpha, min_n=0)
        hit_naive = hit_bonf = False
        for k in range(looks):
            n = n_per_look * (k + 1)
            ctrl, can = ArmCounts(n=n, failures=int(fails[k, 0])), ArmCounts(n=n, failures=int(fails[k, 1]))
            se = math.sqrt(ctrl.rate * (1 - ctrl.rate) / n + can.rate * (1 - can.rate) / n) or 1e-9
            hit_naive |= (can.rate - ctrl.rate) / se > z_naive
            hit_bonf |= mon.observe(ctrl, can).decision == "rollback"
        naive += hit_naive
        bonf += hit_bonf
    return {"naive_peeking": naive / sims, "bonferroni_looks": bonf / sims}


__all__ = ["TraceRecord", "FeedbackEvent", "Outcome", "JoinStats", "join_feedback", "OutcomeMetrics",
           "outcome_metrics", "corrections_to_cases", "ArmCounts", "CanaryDecision", "CanaryMonitor",
           "simulate_false_alarms"]
