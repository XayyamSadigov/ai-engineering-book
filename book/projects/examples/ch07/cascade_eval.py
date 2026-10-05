# path: book/projects/examples/ch07/cascade_eval.py
"""Offline evaluation of a two-stage confidence cascade.

Run the small and the large model once on every case and record correctness, confidence,
cost and latency. Every cascade threshold can then be simulated from that table without
calling a model again, so a full threshold sweep costs one pass over the set per model.

Utility is computed end to end, per request, in one currency (illustrative USD):

    utility = value_correct * correct
              - cost_silent_error * undetected wrong answers
              - rework_cost * detected wrong answers
              - model spend (both stages, plus downstream re-runs)
              - latency_cost_per_s * latency

A wrong answer from the small model that is accepted is a *false accept* (silent quality
loss, or a retry if something downstream catches it). A correct small answer that is escalated
anyway is a *false escalation* (wasted spend and latency). Router accuracy counts both errors
equally; utility weighs each by what it actually costs, which is why the two pick different
thresholds.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

from pydantic import BaseModel

from aie_core import LLMClient, PricingTable

from selection import Scorer, percentile, run_case
from tasks import EvalCase


class CaseOutcome(BaseModel):
    case_id: str
    small_correct: bool
    small_confidence: float
    small_cost_usd: float
    small_latency_ms: float
    large_correct: bool
    large_cost_usd: float
    large_latency_ms: float


class UtilityModel(BaseModel):
    """All values illustrative. Set them with the product owner, not by the engineer alone."""

    value_correct: float = 0.0          # often zero: the baseline is "the task got done"
    cost_silent_error: float = 0.05     # a mis-triaged ticket waits in the wrong queue
    detect_rate: float = 0.0            # share of accepted wrong answers caught downstream
    rework_cost: float = 0.01           # cost of a caught error, on top of re-running large
    latency_cost_per_s: float = 0.0     # what a second of waiting is worth, if anything


class PolicyMetrics(BaseModel):
    policy: str
    threshold: float | None
    accuracy: float
    cost_per_request_usd: float
    mean_latency_ms: float
    p95_latency_ms: float
    escalation_rate: float
    false_accept_rate: float
    false_escalation_rate: float
    router_accuracy: float
    utility_per_request: float


ConfidenceFn = Callable[..., float]


def collect_outcomes(cases: Sequence[EvalCase], small: LLMClient, large: LLMClient, scorer: Scorer,
                     confidence: ConfidenceFn, pricing: PricingTable) -> list[CaseOutcome]:
    outcomes: list[CaseOutcome] = []
    for case in cases:
        # Call small directly to read its confidence; run_case scores and prices it.
        s_completion = small.complete(case.request)
        s = run_case("small", _Replay(s_completion), case, scorer, pricing)
        lg = run_case("large", large, case, scorer, pricing)
        outcomes.append(CaseOutcome(
            case_id=case.id, small_correct=s.score >= 0.5, small_confidence=confidence(s_completion),
            small_cost_usd=s.cost_usd, small_latency_ms=s.latency_ms,
            large_correct=lg.score >= 0.5 and lg.error is None, large_cost_usd=lg.cost_usd,
            large_latency_ms=lg.latency_ms,
        ))
    return outcomes


class _Replay:
    """An LLMClient that returns one prepared completion; lets run_case score it."""

    provider = "replay"

    def __init__(self, completion) -> None:
        self._c = completion

    def complete(self, req):
        return self._c


def _per_case(o: CaseOutcome, escalate: bool, u: UtilityModel) -> tuple[float, float, float, float]:
    """(expected correctness, cost, latency_ms, penalty) for one case under one decision."""
    if escalate:
        cost = o.small_cost_usd + o.large_cost_usd
        latency = o.small_latency_ms + o.large_latency_ms
        return float(o.large_correct), cost, latency, 0.0 if o.large_correct else u.cost_silent_error
    if o.small_correct:
        return 1.0, o.small_cost_usd, o.small_latency_ms, 0.0
    # False accept. A share d of these is caught downstream and redone on the large model,
    # paying rework plus a second call; the rest stays wrong and silent.
    d = u.detect_rate
    redo_penalty = u.rework_cost + (0.0 if o.large_correct else u.cost_silent_error)
    return (d * float(o.large_correct),
            o.small_cost_usd + d * o.large_cost_usd,
            o.small_latency_ms + d * o.large_latency_ms,
            d * redo_penalty + (1 - d) * u.cost_silent_error)


def simulate(outcomes: Sequence[CaseOutcome], threshold: float, u: UtilityModel) -> PolicyMetrics:
    """Accept the small answer when its confidence >= threshold, otherwise escalate."""
    n = len(outcomes)
    correct = 0.0
    cost = penalty = 0.0
    latencies: list[float] = []
    escalations = false_accepts = false_escalations = 0
    for o in outcomes:
        escalate = o.small_confidence < threshold
        ok, c, lat, pen = _per_case(o, escalate, u)
        correct += ok
        cost += c
        penalty += pen
        latencies.append(lat)
        escalations += escalate
        false_accepts += (not escalate) and (not o.small_correct)
        false_escalations += escalate and o.small_correct
    return _metrics(f"cascade@{threshold:.2f}", threshold, n, correct, cost, penalty, latencies,
                    escalations, false_accepts, false_escalations, u)


def baseline(outcomes: Sequence[CaseOutcome], which: str, u: UtilityModel) -> PolicyMetrics:
    """Single-model policies: 'small' (never escalate) or 'large' (call large only)."""
    n = len(outcomes)
    if which == "small":
        m = simulate(outcomes, threshold=0.0, u=u)
        return m.model_copy(update={"policy": "always-small", "threshold": None})
    if which != "large":
        raise ValueError(which)
    correct = sum(o.large_correct for o in outcomes)
    cost = sum(o.large_cost_usd for o in outcomes)
    penalty = sum(0.0 if o.large_correct else u.cost_silent_error for o in outcomes)
    latencies = [o.large_latency_ms for o in outcomes]
    return _metrics("always-large", None, n, correct, cost, penalty, latencies, n, 0,
                    sum(o.small_correct for o in outcomes), u)


def _metrics(policy: str, threshold: float | None, n: int, correct: float, cost: float, penalty: float,
             latencies: list[float], escalations: int, false_accepts: int, false_escalations: int,
             u: UtilityModel) -> PolicyMetrics:
    mean_latency = sum(latencies) / n
    utility = (u.value_correct * correct - penalty - cost - u.latency_cost_per_s * sum(latencies) / 1000) / n
    return PolicyMetrics(
        policy=policy, threshold=threshold, accuracy=correct / n, cost_per_request_usd=cost / n,
        mean_latency_ms=mean_latency, p95_latency_ms=percentile(latencies, 95),
        escalation_rate=escalations / n, false_accept_rate=false_accepts / n,
        false_escalation_rate=false_escalations / n,
        router_accuracy=1 - (false_accepts + false_escalations) / n, utility_per_request=utility,
    )


def sweep(outcomes: Sequence[CaseOutcome], u: UtilityModel,
          thresholds: Sequence[float] | None = None) -> list[PolicyMetrics]:
    ts = thresholds if thresholds is not None else sorted({0.0, 1.01, *(o.small_confidence for o in outcomes)})
    return [simulate(outcomes, t, u) for t in ts]


def best_by_utility(results: Sequence[PolicyMetrics]) -> PolicyMetrics:
    # Ties go to the lower threshold: same utility, fewer escalations.
    return max(results, key=lambda m: (round(m.utility_per_request, 12), -(m.threshold or 0.0)))


def best_by_router_accuracy(results: Sequence[PolicyMetrics]) -> PolicyMetrics:
    return max(results, key=lambda m: (m.router_accuracy, -(m.threshold or 0.0)))


class CalibrationBin(BaseModel):
    low: float
    high: float
    n: int
    mean_confidence: float
    accuracy: float


def calibration_table(outcomes: Sequence[CaseOutcome], bins: int = 5) -> list[CalibrationBin]:
    out: list[CalibrationBin] = []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        members = [o for o in outcomes if lo <= o.small_confidence < hi or (i == bins - 1 and o.small_confidence == 1.0)]
        if members:
            out.append(CalibrationBin(
                low=lo, high=hi, n=len(members),
                mean_confidence=sum(o.small_confidence for o in members) / len(members),
                accuracy=sum(o.small_correct for o in members) / len(members)))
    return out


def expected_calibration_error(outcomes: Sequence[CaseOutcome], bins: int = 5) -> float:
    n = len(outcomes)
    return sum(b.n / n * abs(b.mean_confidence - b.accuracy) for b in calibration_table(outcomes, bins))


def to_markdown(rows: Sequence[PolicyMetrics]) -> str:
    head = ("| policy | accuracy | $/req | mean ms | p95 ms | escalated | false accept | false escalate | router acc | utility/req |\n"
            "|---|---|---|---|---|---|---|---|---|---|")
    body = [
        f"| {m.policy} | {m.accuracy:.3f} | {m.cost_per_request_usd:.6f} | {m.mean_latency_ms:.0f} | "
        f"{m.p95_latency_ms:.0f} | {m.escalation_rate:.0%} | {m.false_accept_rate:.0%} | "
        f"{m.false_escalation_rate:.0%} | {m.router_accuracy:.3f} | {m.utility_per_request:+.5f} |"
        for m in rows
    ]
    return "\n".join([head, *body])


__all__ = [
    "CaseOutcome", "UtilityModel", "PolicyMetrics", "collect_outcomes", "simulate", "baseline",
    "sweep", "best_by_utility", "best_by_router_accuracy", "CalibrationBin", "calibration_table",
    "expected_calibration_error", "to_markdown",
]
