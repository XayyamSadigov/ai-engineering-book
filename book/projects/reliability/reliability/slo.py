# path: book/projects/reliability/reliability/slo.py
"""SLOs for AI services: indicators computed from request outcomes, error budgets, burn rates.

An AI request can fail in more ways than HTTP status shows. The indicators below are the
ones Northwind Assist commits to: availability (no 5xx, no timeout), time to first token,
completion latency, task success (from sampled evaluation or user signals), and the share
of answers served in a degraded mode. A degraded answer counts as *available*; it is
tracked separately so that hiding an outage behind degradation is visible.

Chapter 31 turns these into dashboards and alerts; this module is the arithmetic.
"""
from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, Field


class RequestOutcome(BaseModel):
    ok: bool                              # served an answer (possibly degraded), no 5xx/timeout
    ttft_s: float | None = None
    total_s: float | None = None
    task_success: bool | None = None      # None when not sampled for quality
    degraded: bool = False
    shed: bool = False                    # rejected by admission control (counts against availability)


class SLOTargets(BaseModel):
    availability: float = 0.995
    ttft_s: float = 2.0
    ttft_objective: float = 0.95          # 95% of answers start within ttft_s
    total_s: float = 8.0
    total_objective: float = 0.95
    task_success: float = 0.85
    max_degraded_share: float = 0.05


class SLIReport(BaseModel):
    requests: int
    availability: float
    ttft_compliance: float
    total_compliance: float
    task_success: float | None
    degraded_share: float
    burn_rates: dict[str, float] = Field(default_factory=dict)
    violations: list[str] = Field(default_factory=list)


def _ratio(good: int, total: int) -> float:
    return good / total if total else 1.0


def burn_rate(good_ratio: float, objective: float) -> float:
    """How fast the error budget is being spent. 1.0 spends exactly the budget over the window.

    With a 99.5% objective the budget is 0.5% bad events; observing 2% bad means burn rate 4,
    i.e., a 28-day budget gone in 7 days.
    """
    budget = 1.0 - objective
    return (1.0 - good_ratio) / budget if budget > 0 else float("inf")


def should_page(short_window_burn: float, long_window_burn: float, threshold: float = 14.4) -> bool:
    """Multi-window rule: page only if both a short and a long window burn fast.

    The long window proves it is not a blip; the short one proves it is still happening.
    14.4 over 1 h / 5 min spends 2% of a 30-day budget in an hour (a common starting point).
    """
    return short_window_burn >= threshold and long_window_burn >= threshold


def evaluate(outcomes: Iterable[RequestOutcome], targets: SLOTargets | None = None) -> SLIReport:
    t = targets or SLOTargets()
    items = list(outcomes)
    n = len(items)
    served = [o for o in items if o.ok]
    avail = _ratio(len(served), n)
    ttft = [o for o in served if o.ttft_s is not None]
    total = [o for o in served if o.total_s is not None]
    ttft_ok = _ratio(sum(1 for o in ttft if o.ttft_s is not None and o.ttft_s <= t.ttft_s), len(ttft))
    total_ok = _ratio(sum(1 for o in total if o.total_s is not None and o.total_s <= t.total_s), len(total))
    sampled = [o for o in items if o.task_success is not None]
    success = _ratio(sum(1 for o in sampled if o.task_success), len(sampled)) if sampled else None
    degraded = _ratio(sum(1 for o in served if o.degraded), len(served)) if served else 0.0

    report = SLIReport(
        requests=n, availability=round(avail, 5), ttft_compliance=round(ttft_ok, 5),
        total_compliance=round(total_ok, 5), task_success=None if success is None else round(success, 5),
        degraded_share=round(degraded, 5),
        burn_rates={
            "availability": round(burn_rate(avail, t.availability), 3),
            "ttft": round(burn_rate(ttft_ok, t.ttft_objective), 3),
            "total": round(burn_rate(total_ok, t.total_objective), 3),
        },
    )
    if avail < t.availability:
        report.violations.append("availability")
    if ttft_ok < t.ttft_objective:
        report.violations.append("ttft")
    if total_ok < t.total_objective:
        report.violations.append("total_latency")
    if success is not None and success < t.task_success:
        report.violations.append("task_success")
    if degraded > t.max_degraded_share:
        report.violations.append("degraded_share")
    return report


__all__ = ["RequestOutcome", "SLOTargets", "SLIReport", "burn_rate", "should_page", "evaluate"]
