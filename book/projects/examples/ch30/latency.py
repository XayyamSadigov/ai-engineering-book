# path: book/projects/examples/ch30/latency.py
"""Latency budgets: allocate an end-to-end SLO across stages, enforce it, and audit it from spans.

Three pieces:
- `LatencyBudget` holds per-stage budgets that must fit the end-to-end target and, for
  stages before the first visible token, the time-to-first-token target.
- `Deadline` turns the budget into per-call timeouts: a stage gets min(its budget, time left).
- `LatencyTracker` reads finished spans (live `Span` objects or JSONL dicts), groups them by
  request, and reports per-stage percentiles and budget violations.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from aie_core.observability import Span


class BudgetError(ValueError):
    """Raised when a budget cannot fit its targets."""


@dataclass(frozen=True)
class StageBudget:
    name: str
    budget_ms: float
    before_first_token: bool = False  # on the TTFT path (auth, retrieval, model prefill...)


@dataclass(frozen=True)
class LatencyBudget:
    total_ms: float
    stages: tuple[StageBudget, ...]
    ttft_ms: float | None = None
    reserve_ms: float = 0.0  # slack held back for network jitter and retries

    def __post_init__(self) -> None:
        names = [s.name for s in self.stages]
        if len(names) != len(set(names)):
            raise BudgetError(f"duplicate stage names: {names}")
        allocated = sum(s.budget_ms for s in self.stages) + self.reserve_ms
        if allocated > self.total_ms + 1e-6:
            raise BudgetError(f"stages plus reserve need {allocated:.0f} ms, total is {self.total_ms:.0f} ms")
        if self.ttft_ms is not None:
            ttft_path = sum(s.budget_ms for s in self.stages if s.before_first_token)
            if ttft_path > self.ttft_ms + 1e-6:
                raise BudgetError(f"first-token path needs {ttft_path:.0f} ms, TTFT target is {self.ttft_ms:.0f} ms")

    def __getitem__(self, name: str) -> StageBudget:
        for s in self.stages:
            if s.name == name:
                return s
        raise KeyError(name)

    @property
    def stage_names(self) -> list[str]:
        return [s.name for s in self.stages]

    @classmethod
    def allocate(
        cls,
        total_ms: float,
        shares: Mapping[str, float],
        *,
        ttft_ms: float | None = None,
        before_first_token: Iterable[str] = (),
        reserve_fraction: float = 0.1,
    ) -> "LatencyBudget":
        """Split (1 - reserve_fraction) of the total across stages in proportion to `shares`."""
        if not 0 <= reserve_fraction < 1:
            raise BudgetError("reserve_fraction must be in [0, 1)")
        weight = sum(shares.values())
        if weight <= 0:
            raise BudgetError("shares must sum to a positive number")
        usable = total_ms * (1 - reserve_fraction)
        ttft_set = set(before_first_token)
        stages = tuple(
            StageBudget(name, round(usable * share / weight, 1), name in ttft_set) for name, share in shares.items()
        )
        reserve = total_ms - sum(s.budget_ms for s in stages)
        return cls(total_ms=total_ms, stages=stages, ttft_ms=ttft_ms, reserve_ms=reserve)

    @classmethod
    def from_measurements(
        cls,
        total_ms: float,
        measured_p95_ms: Mapping[str, float],
        *,
        ttft_ms: float | None = None,
        before_first_token: Iterable[str] = (),
        reserve_fraction: float = 0.1,
    ) -> "LatencyBudget":
        """Derive budgets from measured stage p95s.

        If the measured p95s already fit, each stage gets its p95 plus a proportional share of the
        spare time. If they do not fit, raise: the fix is to make a stage faster or change the
        target, not to hand out budgets the stages cannot meet. Summing p95s overstates the
        end-to-end p95 (stages rarely all hit their tail on one request), so this errs on the safe side.
        """
        usable = total_ms * (1 - reserve_fraction)
        need = sum(measured_p95_ms.values())
        if need > usable:
            worst = max(measured_p95_ms, key=lambda k: measured_p95_ms[k])
            raise BudgetError(
                f"measured stage p95s sum to {need:.0f} ms but only {usable:.0f} ms are usable; "
                f"largest stage is {worst} at {measured_p95_ms[worst]:.0f} ms"
            )
        spare = usable - need
        ttft_set = set(before_first_token)
        stages = tuple(
            StageBudget(name, round(p95 + spare * p95 / need, 1) if need else 0.0, name in ttft_set)
            for name, p95 in measured_p95_ms.items()
        )
        reserve = total_ms - sum(s.budget_ms for s in stages)
        return cls(total_ms=total_ms, stages=stages, ttft_ms=ttft_ms, reserve_ms=reserve)


class Deadline:
    """An absolute deadline derived from the end-to-end budget; hands out per-stage timeouts."""

    def __init__(self, budget: LatencyBudget, *, clock: Callable[[], float] = time.monotonic, start: float | None = None) -> None:
        self.budget = budget
        self._clock = clock
        self.start = clock() if start is None else start
        self.expires_at = self.start + budget.total_ms / 1000

    def remaining_ms(self) -> float:
        return max(0.0, (self.expires_at - self._clock()) * 1000)

    def expired(self) -> bool:
        return self.remaining_ms() <= 0

    def timeout_for(self, stage: str) -> float:
        """Seconds to pass as a timeout to the stage: its budget, capped by what is left overall."""
        return min(self.budget[stage].budget_ms, self.remaining_ms()) / 1000


# ------------------------------------------------------------------------- measurement
def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile; q in [0, 100]. Returns nan for an empty list."""
    if not values:
        return math.nan
    ordered = sorted(values)
    rank = max(1, math.ceil(round(q * len(ordered) / 100, 9)))   # round: 99.9% of 1000 must be 999, not 1000
    return ordered[min(rank, len(ordered)) - 1]


@dataclass(frozen=True)
class StageTiming:
    request_id: str
    stage: str
    start_s: float
    end_s: float

    @property
    def duration_ms(self) -> float:
        return (self.end_s - self.start_s) * 1000


@dataclass(frozen=True)
class Violation:
    request_id: str
    stage: str  # a stage name, or "end_to_end" / "ttft"
    observed_ms: float
    budget_ms: float

    @property
    def overrun_ms(self) -> float:
        return self.observed_ms - self.budget_ms


@dataclass
class LatencyReport:
    requests: int
    stage_p50_ms: dict[str, float]
    stage_p95_ms: dict[str, float]
    stage_p99_ms: dict[str, float]
    end_to_end_p95_ms: float
    sum_of_stage_p95_ms: float
    violations: list[Violation] = field(default_factory=list)

    def violation_rate(self, stage: str) -> float:
        if self.requests == 0:
            return 0.0
        hit = {v.request_id for v in self.violations if v.stage == stage}
        return len(hit) / self.requests

    def worst_stage(self) -> str | None:
        """Stage with the most violations; where to look first."""
        counts: dict[str, int] = defaultdict(int)
        for v in self.violations:
            if v.stage not in ("end_to_end", "ttft"):
                counts[v.stage] += 1
        return max(counts, key=lambda k: counts[k]) if counts else None


def _span_fields(span: Span | Mapping[str, Any]) -> tuple[str, dict[str, Any], float, float | None]:
    if isinstance(span, Span):
        return span.name, span.attributes, span.start, span.end
    return span["name"], dict(span.get("attributes", {})), float(span["start"]), span.get("end")


class LatencyTracker:
    """Collects stage timings from spans and checks them against a `LatencyBudget`.

    A span counts toward a stage when its `stage` attribute is set, or when its name is mapped in
    `stage_for_span` (for example {"llm.complete": "generate"}). Spans need a `request_id`
    attribute; `AttributingTracer` + `bind(request_id=...)` provides it.
    A span may carry `first_token_s` (absolute time of the first streamed token) to measure TTFT.
    """

    def __init__(self, budget: LatencyBudget, stage_for_span: Mapping[str, str] | None = None) -> None:
        self.budget = budget
        self.stage_for_span = dict(stage_for_span or {})
        self._timings: dict[str, list[StageTiming]] = defaultdict(list)
        self._first_token: dict[str, float] = {}

    def record(self, span: Span | Mapping[str, Any]) -> None:
        name, attrs, start, end = _span_fields(span)
        request_id = attrs.get("request_id")
        stage = attrs.get("stage") or self.stage_for_span.get(name)
        if request_id is None or stage is None or end is None:
            return  # unattributed or unfinished spans are ignored, not guessed
        self._timings[str(request_id)].append(StageTiming(str(request_id), stage, start, float(end)))
        if "first_token_s" in attrs:   # the earliest first token wins, so a retry cannot hide an earlier stream
            first = float(attrs["first_token_s"])
            prev = self._first_token.get(str(request_id))
            self._first_token[str(request_id)] = first if prev is None else min(prev, first)

    def record_all(self, spans: Iterable[Span | Mapping[str, Any]]) -> None:
        for s in spans:
            self.record(s)

    def report(self) -> LatencyReport:
        per_stage: dict[str, list[float]] = defaultdict(list)
        e2e: list[float] = []
        violations: list[Violation] = []
        budgets = {s.name: s.budget_ms for s in self.budget.stages}
        for request_id, timings in self._timings.items():
            # A stage may run several times (retries, multiple tool calls): count its wall time once,
            # merging overlapping runs, so two parallel 500 ms calls are 500 ms, not 1000.
            stage_ms: dict[str, float] = {}
            by_stage: dict[str, list[StageTiming]] = defaultdict(list)
            for t in timings:
                by_stage[t.stage].append(t)
            for stage, runs in by_stage.items():
                total, cur_start, cur_end = 0.0, None, None
                for t in sorted(runs, key=lambda r: r.start_s):
                    if cur_end is None or t.start_s > cur_end:
                        if cur_end is not None:
                            total += cur_end - cur_start
                        cur_start, cur_end = t.start_s, t.end_s
                    else:
                        cur_end = max(cur_end, t.end_s)
                total += (cur_end - cur_start) if cur_end is not None else 0.0
                stage_ms[stage] = total * 1000
            for stage, ms in stage_ms.items():
                per_stage[stage].append(ms)
                if stage in budgets and ms > budgets[stage]:
                    violations.append(Violation(request_id, stage, ms, budgets[stage]))
            # End to end is wall clock, not the sum: parallel stages overlap.
            begin = min(t.start_s for t in timings)
            finish = max(t.end_s for t in timings)
            total = (finish - begin) * 1000
            e2e.append(total)
            if total > self.budget.total_ms:
                violations.append(Violation(request_id, "end_to_end", total, self.budget.total_ms))
            if self.budget.ttft_ms is not None and request_id in self._first_token:
                ttft = (self._first_token[request_id] - begin) * 1000
                if ttft > self.budget.ttft_ms:
                    violations.append(Violation(request_id, "ttft", ttft, self.budget.ttft_ms))
        p95 = {k: percentile(v, 95) for k, v in per_stage.items()}
        return LatencyReport(
            requests=len(self._timings),
            stage_p50_ms={k: percentile(v, 50) for k, v in per_stage.items()},
            stage_p95_ms=p95,
            stage_p99_ms={k: percentile(v, 99) for k, v in per_stage.items()},
            end_to_end_p95_ms=percentile(e2e, 95),
            sum_of_stage_p95_ms=sum(p95.values()),
            violations=violations,
        )


__all__ = [
    "BudgetError",
    "StageBudget",
    "LatencyBudget",
    "Deadline",
    "percentile",
    "StageTiming",
    "Violation",
    "LatencyReport",
    "LatencyTracker",
]
