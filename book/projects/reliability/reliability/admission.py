# path: book/projects/reliability/reliability/admission.py
"""Admission control: decide at the door whether to admit, degrade, defer, or reject.

The model tier is a finite, slow, expensive resource. When arrivals exceed what it can
serve, a queue forms, and queueing delay grows without bound as utilization approaches 1.
Every request then misses its deadline *after* consuming capacity. Admission control
refuses early and cheaply instead, using four signals:

1. Per-tenant quotas (token buckets on requests and estimated tokens) for fairness: one
   tenant's batch script cannot starve the other tenant's employees.
2. Priority classes: interactive work is served first; batch work only uses spare capacity
   and is *deferred* (put back in its queue), never failed, when capacity is tight.
3. Estimated wait, from queue length and an EWMA of service time (Little's law): if a
   request would wait longer than its own deadline, doing it is pure waste.
4. Utilization thresholds that trigger *degraded* admission (smaller model, less retrieval)
   before outright rejection, because a cheaper answer now beats a full answer never.

The controller is in-process and synchronous. Run one per replica and size `capacity` to
that replica's share of downstream concurrency (provider quota / replica count).
"""
from __future__ import annotations

import itertools
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .clock import Clock
from .errors import AdmissionRejected


class Priority(str, Enum):
    INTERACTIVE = "interactive"
    BATCH = "batch"


class Action(str, Enum):
    ADMIT = "admit"
    DEGRADE = "degrade"      # admitted, but run the degraded plan (see degrade.py)
    DEFER = "defer"          # not now; batch work goes back to its queue with a delay
    REJECT = "reject"        # fail fast with 429 (quota) or 503 (overload)


class TenantQuota(BaseModel):
    requests_per_minute: float = 600.0
    tokens_per_minute: float = 2_000_000.0
    burst_fraction: float = Field(0.25, description="bucket capacity as a fraction of the per-minute limit")


class AdmissionConfig(BaseModel):
    capacity: int = Field(32, description="concurrent executions this replica can serve within SLO")
    max_queue: int = Field(32, description="interactive requests allowed to wait beyond capacity")
    batch_max_utilization: float = 0.6
    degrade_utilization: float = 0.85
    degrade_wait_s: float = 1.0
    severe_wait_s: float = 3.0
    initial_service_s: float = 4.0
    ewma_alpha: float = 0.2
    default_quota: TenantQuota = TenantQuota()
    tenant_quotas: dict[str, TenantQuota] = {}


class AdmissionRequest(BaseModel):
    tenant_id: str
    priority: Priority = Priority.INTERACTIVE
    est_tokens: int = 0
    deadline_s: float | None = Field(None, description="remaining end-to-end budget of the request")


class AdmissionDecision(BaseModel):
    action: Action
    reason: str
    degrade_level: int = 0
    retry_after_s: float | None = None
    est_wait_s: float = 0.0
    ticket: int | None = None
    priority: Priority = Priority.INTERACTIVE

    @property
    def admitted(self) -> bool:
        return self.action in (Action.ADMIT, Action.DEGRADE)

    def http_status(self) -> int:
        """Quota exhaustion is the caller's problem (429); overload is ours (503)."""
        if self.admitted:
            return 200
        return 429 if self.reason == "tenant_quota" else 503


class TokenBucket:
    """Capacity `capacity`, refilled continuously at `refill_per_s`. `take` returns the wait, 0 if taken."""

    def __init__(self, capacity: float, refill_per_s: float, clock: Clock = time.monotonic) -> None:
        self.capacity = max(1.0, capacity)
        self.refill_per_s = refill_per_s
        self.tokens = self.capacity
        self._clock = clock
        self._updated = clock()

    def _refill(self) -> None:
        now = self._clock()
        self.tokens = min(self.capacity, self.tokens + (now - self._updated) * self.refill_per_s)
        self._updated = now

    def wait_for(self, amount: float) -> float:
        self._refill()
        amount = min(amount, self.capacity)
        if self.tokens >= amount:
            return 0.0
        return (amount - self.tokens) / self.refill_per_s if self.refill_per_s > 0 else float("inf")

    def take(self, amount: float) -> None:
        self.tokens -= min(amount, self.capacity)


class _TenantBuckets:
    def __init__(self, quota: TenantQuota, clock: Clock) -> None:
        self.requests = TokenBucket(quota.requests_per_minute * quota.burst_fraction, quota.requests_per_minute / 60, clock)
        self.tokens = TokenBucket(quota.tokens_per_minute * quota.burst_fraction, quota.tokens_per_minute / 60, clock)

    def try_take(self, est_tokens: int) -> float:
        """Both buckets or neither; returns the wait when either is short."""
        wait = max(self.requests.wait_for(1), self.tokens.wait_for(est_tokens) if est_tokens else 0.0)
        if wait == 0.0:
            self.requests.take(1)
            if est_tokens:
                self.tokens.take(est_tokens)
        return wait


class AdmissionController:
    def __init__(self, config: AdmissionConfig | None = None, *, clock: Clock = time.monotonic,
                 external_queue_depth: Callable[[], int] | None = None) -> None:
        self.config = config or AdmissionConfig()
        self._clock = clock
        self._external_depth = external_queue_depth
        self._lock = threading.Lock()
        self._tenants: dict[str, _TenantBuckets] = {}
        self._in_flight: dict[int, Priority] = {}
        self._tickets = itertools.count(1)
        self._service_s = self.config.initial_service_s
        self.min_degrade_level = 0  # operator floor, e.g. set to 1 during a provider incident
        self.counts: dict[str, int] = {}

    # ------------------------------------------------------------------ load signals
    def _queue_length(self) -> int:
        beyond = max(0, len(self._in_flight) - self.config.capacity)
        return beyond + (self._external_depth() if self._external_depth else 0)

    def estimated_wait_s(self) -> float:
        """Little's law: the queue drains at capacity / service_time requests per second."""
        q = self._queue_length()
        return q * self._service_s / self.config.capacity if q else 0.0

    def utilization(self) -> float:
        return len(self._in_flight) / self.config.capacity

    def _buckets(self, tenant_id: str) -> _TenantBuckets:
        if tenant_id not in self._tenants:
            quota = self.config.tenant_quotas.get(tenant_id, self.config.default_quota)
            self._tenants[tenant_id] = _TenantBuckets(quota, self._clock)
        return self._tenants[tenant_id]

    def _decide(self, action: Action, reason: str, req: AdmissionRequest, **kw: Any) -> AdmissionDecision:
        self.counts[f"{action.value}:{reason}"] = self.counts.get(f"{action.value}:{reason}", 0) + 1
        return AdmissionDecision(action=action, reason=reason, priority=req.priority, **kw)

    # ------------------------------------------------------------------ the decision
    def admit(self, req: AdmissionRequest) -> AdmissionDecision:
        cfg = self.config
        with self._lock:
            util = self.utilization()
            wait = self.estimated_wait_s()

            # Load checks first: they do not consume the tenant's quota.
            if req.priority is Priority.BATCH and util >= cfg.batch_max_utilization:
                return self._decide(Action.DEFER, "batch_yields_to_interactive", req,
                                    retry_after_s=self._service_s, est_wait_s=wait)
            if req.priority is Priority.INTERACTIVE:
                if len(self._in_flight) >= cfg.capacity and self._queue_length() >= cfg.max_queue:
                    return self._decide(Action.REJECT, "overloaded", req, retry_after_s=wait, est_wait_s=wait)
                if req.deadline_s is not None and wait + self._service_s * 0.5 > req.deadline_s:
                    # Even a degraded answer (assumed ~half the service time) would land too late.
                    return self._decide(Action.REJECT, "would_miss_deadline", req, retry_after_s=wait, est_wait_s=wait)

            quota_wait = self._buckets(req.tenant_id).try_take(req.est_tokens)
            if quota_wait > 0.0:
                action = Action.DEFER if req.priority is Priority.BATCH else Action.REJECT
                return self._decide(action, "tenant_quota", req, retry_after_s=quota_wait, est_wait_s=wait)

            level = self.min_degrade_level
            if req.priority is Priority.INTERACTIVE:
                if wait >= cfg.severe_wait_s:
                    level = max(level, 2)
                elif wait >= cfg.degrade_wait_s or util >= cfg.degrade_utilization:
                    level = max(level, 1)
                if req.deadline_s is not None and wait + self._service_s > req.deadline_s:
                    level = max(level, 2)  # full quality would not fit; the degraded plan might
            ticket = next(self._tickets)
            self._in_flight[ticket] = req.priority
            action = Action.DEGRADE if level > 0 else Action.ADMIT
            reason = "degraded_under_load" if level > 0 else "ok"
            return self._decide(action, reason, req, degrade_level=level, est_wait_s=wait, ticket=ticket)

    def release(self, decision: AdmissionDecision, service_s: float | None = None) -> None:
        if decision.ticket is None:
            return
        with self._lock:
            self._in_flight.pop(decision.ticket, None)
            if service_s is not None and decision.degrade_level == 0:
                a = self.config.ewma_alpha
                self._service_s = (1 - a) * self._service_s + a * service_s

    @contextmanager
    def guard(self, req: AdmissionRequest) -> Iterator[AdmissionDecision]:
        """Admit or raise AdmissionRejected; always release and record service time."""
        decision = self.admit(req)
        if not decision.admitted:
            raise AdmissionRejected(f"{decision.action.value}: {decision.reason}", reason=decision.reason,
                                    retry_after_s=decision.retry_after_s)
        started = self._clock()
        try:
            yield decision
        finally:
            self.release(decision, self._clock() - started)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "in_flight": len(self._in_flight),
                "in_flight_batch": sum(1 for p in self._in_flight.values() if p is Priority.BATCH),
                "utilization": round(self.utilization(), 3),
                "queue_length": self._queue_length(),
                "est_wait_s": round(self.estimated_wait_s(), 3),
                "service_s_ewma": round(self._service_s, 3),
                "min_degrade_level": self.min_degrade_level,
                "counts": dict(self.counts),
            }


__all__ = [
    "Priority",
    "Action",
    "TenantQuota",
    "AdmissionConfig",
    "AdmissionRequest",
    "AdmissionDecision",
    "AdmissionController",
    "TokenBucket",
]
