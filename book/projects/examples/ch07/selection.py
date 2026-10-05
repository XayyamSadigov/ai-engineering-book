# path: book/projects/examples/ch07/selection.py
"""Evaluation-driven model selection.

Run one evaluation set across several candidate clients, record quality, latency, cost and
failures per candidate, and compute the Pareto front over (quality up, cost down, p95
latency down). The harness calls candidates through the LLMClient protocol, so the same code
runs against FakeLLM in tests and against ModelGateway-wrapped providers in a real bake-off.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence

from pydantic import BaseModel, Field

from aie_core import Completion, LLMClient, LLMError, PricingTable

from tasks import EvalCase

Scorer = Callable[[EvalCase, Completion], float]


class CaseResult(BaseModel):
    case_id: str
    candidate: str
    score: float                       # 0..1; 0 for errors
    latency_ms: float
    cost_usd: float
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None
    output: str = ""


class CandidateSummary(BaseModel):
    candidate: str
    n: int
    quality: float
    quality_low: float                 # Wilson 95% interval on the pass rate
    quality_high: float
    p50_latency_ms: float
    p95_latency_ms: float
    cost_per_request_usd: float
    cost_per_correct_usd: float
    error_rate: float
    pareto: bool = False


class SelectionReport(BaseModel):
    summaries: list[CandidateSummary]
    results: list[CaseResult] = Field(default_factory=list)

    def by_name(self, name: str) -> CandidateSummary:
        return next(s for s in self.summaries if s.candidate == name)

    def to_markdown(self) -> str:
        head = ("| candidate | quality (95% CI) | p50 ms | p95 ms | $/request | $/correct | errors | Pareto |\n"
                "|---|---|---|---|---|---|---|---|")
        rows = [
            f"| {s.candidate} | {s.quality:.3f} ({s.quality_low:.2f}-{s.quality_high:.2f}) | "
            f"{s.p50_latency_ms:.0f} | {s.p95_latency_ms:.0f} | {s.cost_per_request_usd:.6f} | "
            f"{s.cost_per_correct_usd:.6f} | {s.error_rate:.1%} | {'yes' if s.pareto else ''} |"
            for s in self.summaries
        ]
        return "\n".join([head, *rows])


def wilson_interval(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval: honest at small n and near 0 or 1, unlike the normal approximation."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile; q in [0, 100]."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


def run_case(name: str, client: LLMClient, case: EvalCase, scorer: Scorer,
             pricing: PricingTable, model: str | None = None) -> CaseResult:
    req = case.request.model_copy(update={"model": model}) if model else case.request
    started = time.perf_counter()
    try:
        completion = client.complete(req)
    except LLMError as exc:  # a failed call is a scored outcome, not a crash of the harness
        return CaseResult(case_id=case.id, candidate=name, score=0.0,
                          latency_ms=(time.perf_counter() - started) * 1000,
                          cost_usd=0.0, error=type(exc).__name__)
    wall_ms = (time.perf_counter() - started) * 1000
    # Providers report server-side latency; fakes report a scripted one. Prefer the reported
    # figure when present so fakes are deterministic; real runs should also log wall time.
    latency = completion.latency_ms or wall_ms
    return CaseResult(
        case_id=case.id, candidate=name, score=scorer(case, completion), latency_ms=latency,
        cost_usd=pricing.cost_usd(completion.model, completion.usage),
        input_tokens=completion.usage.input_tokens, output_tokens=completion.usage.output_tokens,
        output=completion.text[:500],
    )


def summarize(name: str, results: Sequence[CaseResult]) -> CandidateSummary:
    n = len(results)
    correct = sum(r.score for r in results)
    lo, hi = wilson_interval(correct, n)
    latencies = [r.latency_ms for r in results]
    total_cost = sum(r.cost_usd for r in results)
    return CandidateSummary(
        candidate=name, n=n, quality=correct / n if n else 0.0, quality_low=lo, quality_high=hi,
        p50_latency_ms=percentile(latencies, 50), p95_latency_ms=percentile(latencies, 95),
        cost_per_request_usd=total_cost / n if n else 0.0,
        cost_per_correct_usd=total_cost / correct if correct else math.inf,
        error_rate=sum(1 for r in results if r.error) / n if n else 0.0,
    )


def dominates(a: CandidateSummary, b: CandidateSummary) -> bool:
    """a dominates b if it is no worse on every axis and strictly better on at least one."""
    no_worse = (a.quality >= b.quality and a.cost_per_request_usd <= b.cost_per_request_usd
                and a.p95_latency_ms <= b.p95_latency_ms)
    better = (a.quality > b.quality or a.cost_per_request_usd < b.cost_per_request_usd
              or a.p95_latency_ms < b.p95_latency_ms)
    return no_worse and better


def pareto_front(summaries: Sequence[CandidateSummary]) -> list[CandidateSummary]:
    return [s for s in summaries if not any(dominates(o, s) for o in summaries if o is not s)]


def run_selection(candidates: Mapping[str, LLMClient], cases: Sequence[EvalCase], scorer: Scorer,
                  pricing: PricingTable, models: Mapping[str, str] | None = None) -> SelectionReport:
    """Evaluate every candidate on every case. `models` optionally maps candidate name to the
    pinned model id to request, for clients that serve several models."""
    results: list[CaseResult] = []
    summaries: list[CandidateSummary] = []
    for name, client in candidates.items():
        rs = [run_case(name, client, c, scorer, pricing, (models or {}).get(name)) for c in cases]
        results.extend(rs)
        summaries.append(summarize(name, rs))
    front = {s.candidate for s in pareto_front(summaries)}
    for s in summaries:
        s.pareto = s.candidate in front
    summaries.sort(key=lambda s: (-s.quality, s.cost_per_request_usd))
    return SelectionReport(summaries=summaries, results=results)


def choose(report: SelectionReport, *, min_quality: float, max_p95_ms: float,
           use_lower_bound: bool = True) -> CandidateSummary | None:
    """Cheapest candidate meeting the quality floor and latency ceiling. With use_lower_bound
    the floor applies to the lower end of the confidence interval, which refuses to pick a
    model on luck when the evaluation set is small."""
    ok = [s for s in report.summaries
          if (s.quality_low if use_lower_bound else s.quality) >= min_quality and s.p95_latency_ms <= max_p95_ms]
    return min(ok, key=lambda s: (s.cost_per_request_usd, s.p95_latency_ms), default=None)


def paired_disagreements(report: SelectionReport, a: str, b: str) -> tuple[int, int]:
    """(cases a got right and b got wrong, cases b got right and a got wrong). Paired counts
    are what a McNemar-style comparison of two candidates on the same set is built on."""
    sa = {r.case_id: r.score for r in report.results if r.candidate == a}
    sb = {r.case_id: r.score for r in report.results if r.candidate == b}
    a_only = sum(1 for k in sa if sa[k] >= 0.5 > sb.get(k, 0.0))
    b_only = sum(1 for k in sb if sb[k] >= 0.5 > sa.get(k, 0.0))
    return a_only, b_only


__all__ = [
    "Scorer", "CaseResult", "CandidateSummary", "SelectionReport", "wilson_interval", "percentile",
    "run_case", "summarize", "dominates", "pareto_front", "run_selection", "choose",
    "paired_disagreements",
]
