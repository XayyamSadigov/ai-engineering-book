# path: book/projects/examples/ch35/back_of_envelope.py
"""Back-of-the-envelope sizing for AI systems (Chapter 35, step 9 of the design method).

Every function is a one-line formula with named arguments, so a design review can be
re-run when an assumption changes. Standard library only. All prices and rates passed in
are illustrative inputs supplied by the caller; nothing here knows any vendor's numbers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


def peak_rps(daily_requests: float, active_hours: float = 8.0, peak_factor: float = 3.0) -> float:
    """Requests per second at peak.

    ``daily_requests`` spread over ``active_hours`` gives the average rate; ``peak_factor``
    is the ratio of the busiest minute to that average (measure it; 2-5 is typical for
    office-hours traffic).
    """
    if active_hours <= 0:
        raise ValueError("active_hours must be positive")
    return daily_requests / (active_hours * 3600.0) * peak_factor


def tokens_per_second(rps: float, tokens_per_request: float) -> float:
    """Token throughput the model tier must sustain: rate times tokens per request."""
    return rps * tokens_per_request


def inflight_requests(rps: float, avg_latency_s: float) -> float:
    """Little's Law: average concurrency L = arrival rate x time in system."""
    return rps * avg_latency_s


def daily_tokens(daily_requests: float, tokens_per_request: float) -> float:
    """Total tokens per day for one token class (input, output, or cached input)."""
    return daily_requests * tokens_per_request


def cost_per_day(
    input_tokens: float,
    output_tokens: float,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_input_tokens: float = 0.0,
    price_cached_per_m: float = 0.0,
    fixed_cost: float = 0.0,
) -> float:
    """Daily cost from per-million-token prices.

    ``input_tokens`` are the uncached input tokens; cached tokens are passed separately so a
    prefix-cache decision shows up as an explicit line. ``fixed_cost`` covers things billed
    per day rather than per token (self-hosted replicas, rerankers, OCR, human review).
    """
    return (
        input_tokens * price_in_per_m / 1e6
        + cached_input_tokens * price_cached_per_m / 1e6
        + output_tokens * price_out_per_m / 1e6
        + fixed_cost
    )


def vector_storage_bytes(
    chunks: int, dimensions: int, bytes_per_dim: int = 4, index_overhead: float = 1.5
) -> float:
    """Raw vectors plus graph/index overhead (HNSW roughly 1.3-2x the raw vectors)."""
    return chunks * dimensions * bytes_per_dim * index_overhead


def replicas_needed(
    demand_tokens_per_s: float, replica_tokens_per_s: float, headroom: float = 1.5
) -> int:
    """Replicas to serve ``demand_tokens_per_s`` with ``headroom`` (1.5 = 50 percent spare)."""
    if replica_tokens_per_s <= 0:
        raise ValueError("replica_tokens_per_s must be positive")
    return max(1, math.ceil(demand_tokens_per_s * headroom / replica_tokens_per_s))


@dataclass(frozen=True)
class Workload:
    """One request class: how often it happens and what each request costs in tokens."""

    name: str
    daily_requests: float
    input_tokens: float  # per request, before caching
    output_tokens: float  # per request
    avg_latency_s: float
    active_hours: float = 8.0
    peak_factor: float = 3.0
    cached_fraction: float = 0.0  # share of input tokens served from a prefix cache


@dataclass(frozen=True)
class Prices:
    """Illustrative per-million-token prices supplied by the caller."""

    input_per_m: float
    output_per_m: float
    cached_input_per_m: float = 0.0


@dataclass(frozen=True)
class Estimate:
    peak_rps: float
    peak_input_tps: float
    peak_output_tps: float
    inflight: float
    daily_input_tokens: float
    daily_cached_tokens: float
    daily_output_tokens: float
    cost_per_day: float
    cost_per_request: float
    notes: list[str] = field(default_factory=list)


def estimate(workload: Workload, prices: Prices, fixed_cost: float = 0.0) -> Estimate:
    """Run the full step-9 arithmetic for one workload. Each field is one formula above."""
    rps = peak_rps(workload.daily_requests, workload.active_hours, workload.peak_factor)
    cached = daily_tokens(workload.daily_requests, workload.input_tokens * workload.cached_fraction)
    uncached = daily_tokens(workload.daily_requests, workload.input_tokens) - cached
    out = daily_tokens(workload.daily_requests, workload.output_tokens)
    cost = cost_per_day(
        uncached,
        out,
        prices.input_per_m,
        prices.output_per_m,
        cached_input_tokens=cached,
        price_cached_per_m=prices.cached_input_per_m,
        fixed_cost=fixed_cost,
    )
    return Estimate(
        peak_rps=rps,
        peak_input_tps=tokens_per_second(rps, workload.input_tokens),
        peak_output_tps=tokens_per_second(rps, workload.output_tokens),
        inflight=inflight_requests(rps, workload.avg_latency_s),
        daily_input_tokens=uncached,
        daily_cached_tokens=cached,
        daily_output_tokens=out,
        cost_per_day=cost,
        cost_per_request=cost / workload.daily_requests if workload.daily_requests else 0.0,
        notes=[
            "peak_factor and active_hours are assumptions: replace with measured traffic",
            "add retries, agent steps, and failed tasks before quoting cost per successful task",
        ],
    )
