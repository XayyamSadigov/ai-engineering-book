# path: book/projects/examples/ch34/capacity.py
"""Capacity-planning helpers: Little's Law, the queueing knee, headroom, replica estimates.

These are sanity checks, not a substitute for a load test. Little's Law is exact for any stable
system; the M/M/1 curve is an idealization that shows *why* latency explodes near saturation.
The replica estimate converts token rates into a first guess that the benchmark then corrects.
"""
from __future__ import annotations

import math

from pydantic import BaseModel, Field


# ----------------------------------------------------------------------------- Little's Law


def in_flight(arrival_rate_per_s: float, mean_time_in_system_s: float) -> float:
    """L = lambda * W. Average number of requests inside the system (queued or running)."""
    _non_negative(arrival_rate_per_s, "arrival_rate_per_s")
    _non_negative(mean_time_in_system_s, "mean_time_in_system_s")
    return arrival_rate_per_s * mean_time_in_system_s


def arrival_rate(in_flight_requests: float, mean_time_in_system_s: float) -> float:
    """lambda = L / W. The arrival rate a fixed concurrency slot count can sustain."""
    if mean_time_in_system_s <= 0:
        raise ValueError("mean_time_in_system_s must be positive")
    return in_flight_requests / mean_time_in_system_s


def mean_time_in_system(in_flight_requests: float, arrival_rate_per_s: float) -> float:
    """W = L / lambda. What users wait on average when L requests share the system."""
    if arrival_rate_per_s <= 0:
        raise ValueError("arrival_rate_per_s must be positive")
    return in_flight_requests / arrival_rate_per_s


# ----------------------------------------------------------------------------- Queueing knee


def utilization(arrival_rate_per_s: float, service_rate_per_s: float) -> float:
    """rho = lambda / mu. Above 1.0 the queue grows without bound."""
    if service_rate_per_s <= 0:
        raise ValueError("service_rate_per_s must be positive")
    _non_negative(arrival_rate_per_s, "arrival_rate_per_s")
    return arrival_rate_per_s / service_rate_per_s


def mm1_response_time(service_time_s: float, rho: float) -> float:
    """Mean time in system for an M/M/1 queue: S / (1 - rho).

    Idealized (Poisson arrivals, exponential service, one server). Real LLM servers batch and
    have heavy-tailed service times, so the knee is sharper and arrives earlier. The shape of
    the curve is still the right intuition: latency is flat, then vertical.
    """
    if not 0 <= rho < 1:
        raise ValueError("rho must be in [0, 1) for a stable queue")
    return service_time_s / (1.0 - rho)


def queueing_curve(service_time_s: float, utilizations: list[float]) -> list[tuple[float, float]]:
    """(rho, mean response time) pairs; plot it once and you will remember the knee."""
    return [(rho, mm1_response_time(service_time_s, rho)) for rho in utilizations]


def headroom(capacity: float, demand: float) -> float:
    """Fraction of capacity left unused at this demand. Negative means overload."""
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    return 1.0 - demand / capacity


# ----------------------------------------------------------------------------- Replica estimate


class Workload(BaseModel):
    """Traffic shape. Use measured distributions; the mean alone hides the saturation story."""

    peak_requests_per_s: float = Field(gt=0)
    mean_input_tokens: float = Field(gt=0)
    mean_output_tokens: float = Field(gt=0)
    p95_input_tokens: float | None = None
    p95_output_tokens: float | None = None
    mean_e2e_s: float = Field(gt=0, description="measured or targeted mean end-to-end latency")


class ReplicaProfile(BaseModel):
    """What one replica sustains *while meeting the SLO*, taken from a load test, not a spec sheet."""

    decode_tokens_per_s: float = Field(gt=0, description="aggregate output tokens/s at the operating point")
    prefill_tokens_per_s: float = Field(gt=0, description="aggregate prompt tokens/s at the operating point")
    kv_budget_bytes: int = Field(gt=0, description="KV memory available after weights and headroom")
    kv_bytes_per_token: int = Field(gt=0)


class ReplicaEstimate(BaseModel):
    decode_bound_replicas: float
    prefill_bound_replicas: float
    kv_bound_replicas: float
    target_utilization: float
    failover_replicas: int
    recommended_replicas: int
    in_flight_requests: float
    binding_constraint: str

    def summary(self) -> str:
        return (
            f"in flight ~{self.in_flight_requests:.1f}; decode needs {self.decode_bound_replicas:.2f}, "
            f"prefill {self.prefill_bound_replicas:.2f}, KV {self.kv_bound_replicas:.2f} replicas at "
            f"{self.target_utilization:.0%} utilization; binding: {self.binding_constraint}; "
            f"+{self.failover_replicas} failover -> {self.recommended_replicas} replicas"
        )


def replicas_needed(
    workload: Workload,
    profile: ReplicaProfile,
    target_utilization: float = 0.6,
    failover_replicas: int = 1,
) -> ReplicaEstimate:
    """First-cut replica count from three independent constraints.

    Decode: peak output tokens/s against what a replica decodes under SLO.
    Prefill: peak prompt tokens/s against prefill throughput.
    KV: average in-flight requests (Little's Law) times per-request cache, against KV budget.
    Each is divided by ``target_utilization`` because running near 100 percent puts you past the
    queueing knee. The largest wins; failover replicas are added on top.
    """
    if not 0 < target_utilization <= 1:
        raise ValueError("target_utilization must be in (0, 1]")
    if failover_replicas < 0:
        raise ValueError("failover_replicas must be non-negative")

    decode_demand = workload.peak_requests_per_s * workload.mean_output_tokens
    prefill_demand = workload.peak_requests_per_s * workload.mean_input_tokens
    decode_r = decode_demand / (profile.decode_tokens_per_s * target_utilization)
    prefill_r = prefill_demand / (profile.prefill_tokens_per_s * target_utilization)

    l_in_flight = in_flight(workload.peak_requests_per_s, workload.mean_e2e_s)
    # Size KV for the p95 sequence when known: tail requests are what exhaust cache.
    seq_tokens = (workload.p95_input_tokens or workload.mean_input_tokens) + (
        workload.p95_output_tokens or workload.mean_output_tokens
    )
    kv_per_request = seq_tokens * profile.kv_bytes_per_token
    kv_r = (l_in_flight * kv_per_request) / (profile.kv_budget_bytes * target_utilization)

    constraints = {"decode": decode_r, "prefill": prefill_r, "kv": kv_r}
    binding = max(constraints, key=constraints.__getitem__)
    recommended = math.ceil(constraints[binding]) + failover_replicas
    return ReplicaEstimate(
        decode_bound_replicas=decode_r,
        prefill_bound_replicas=prefill_r,
        kv_bound_replicas=kv_r,
        target_utilization=target_utilization,
        failover_replicas=failover_replicas,
        recommended_replicas=max(recommended, 1),
        in_flight_requests=l_in_flight,
        binding_constraint=binding,
    )


def _non_negative(value: float, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


if __name__ == "__main__":
    # Illustrative Northwind Assist RAG traffic and a replica profile from a load test.
    wl = Workload(
        peak_requests_per_s=6, mean_input_tokens=3000, mean_output_tokens=300,
        p95_input_tokens=6000, p95_output_tokens=600, mean_e2e_s=4.0,
    )
    prof = ReplicaProfile(
        decode_tokens_per_s=1500, prefill_tokens_per_s=20000,
        kv_budget_bytes=55 * 1024**3, kv_bytes_per_token=131072,
    )
    print(f"Little's Law: {in_flight(wl.peak_requests_per_s, wl.mean_e2e_s):.0f} requests in flight at peak")
    print(replicas_needed(wl, prof).summary())
    print("M/M/1 knee (service 2 s):")
    for rho, w in queueing_curve(2.0, [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]):
        print(f"  rho={rho:.2f} -> mean response {w:6.1f} s")
