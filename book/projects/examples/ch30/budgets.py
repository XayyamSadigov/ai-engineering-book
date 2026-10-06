# path: book/projects/examples/ch30/budgets.py
"""Spend limits per tenant per day, with alerting and three enforcement modes, plus task token budgets.

- `SpendGuard` keeps a ledger of committed and reserved spend per (tenant, UTC day). A request
  reserves its worst-case cost before the call and commits the actual cost after it, so many
  concurrent requests cannot jointly overshoot a limit that each one checked alone.
- Modes: `observe` (alert only), `degrade` (past the soft limit, route to a cheaper model with a
  smaller output cap; block at the hard limit), `enforce` (block at the hard limit).
- `BudgetedClient` wraps any `LLMClient` (normally the `ModelGateway`) and applies the guard.
- `TaskTokenBudget` caps cumulative tokens across the steps of one agent or chain run.
"""
from __future__ import annotations

import datetime as dt
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Iterator, Literal

from aie_core.llm.client import LLMClient
from aie_core.llm.errors import InvalidRequestError, LLMError
from aie_core.llm.gateway import PricingTable
from aie_core.llm.tokens import count_message_tokens
from aie_core.llm.types import Completion, CompletionRequest, StreamEvent, Usage

Mode = Literal["observe", "degrade", "enforce"]
Action = Literal["allow", "degrade", "block"]


class BudgetExceededError(LLMError):
    """Spend or token budget exhausted. Not retryable: retrying spends more of what is gone."""

    default_retryable = False


@dataclass(frozen=True)
class SpendPolicy:
    daily_limit_usd: float
    mode: Mode = "enforce"
    soft_limit_fraction: float = 0.8  # where `degrade` mode starts degrading
    alert_thresholds: tuple[float, ...] = (0.5, 0.8, 1.0)


@dataclass(frozen=True)
class Alert:
    tenant: str
    day: str
    kind: Literal["threshold", "blocked"]
    threshold: float
    spent_usd: float
    limit_usd: float


@dataclass(frozen=True)
class Decision:
    action: Action
    reservation: "Reservation | None"
    spent_usd: float
    limit_usd: float
    reason: str = ""

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.limit_usd - self.spent_usd)


@dataclass
class Reservation:
    id: str
    tenant: str
    day: str
    amount_usd: float
    open: bool = True


@dataclass
class _DayLedger:
    committed: float = 0.0
    reserved: dict[str, float] = field(default_factory=dict)
    alerted: set[str] = field(default_factory=set)

    @property
    def exposure(self) -> float:
        return self.committed + sum(self.reserved.values())


def utc_day(epoch_s: float) -> str:
    return dt.datetime.fromtimestamp(epoch_s, tz=dt.timezone.utc).date().isoformat()


class SpendGuard:
    def __init__(
        self,
        policies: dict[str, SpendPolicy],
        *,
        default_policy: SpendPolicy | None = None,
        on_alert: Callable[[Alert], None] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.policies = policies
        self.default_policy = default_policy
        self.on_alert = on_alert or (lambda alert: None)
        self._clock = clock
        self._ledgers: dict[tuple[str, str], _DayLedger] = {}
        self._lock = threading.Lock()

    def policy_for(self, tenant: str) -> SpendPolicy:
        policy = self.policies.get(tenant, self.default_policy)
        if policy is None:
            raise InvalidRequestError(f"no spend policy for tenant {tenant!r}")
        return policy

    def _ledger(self, tenant: str, day: str) -> _DayLedger:
        return self._ledgers.setdefault((tenant, day), _DayLedger())

    def spent(self, tenant: str) -> float:
        with self._lock:
            return self._ledger(tenant, utc_day(self._clock())).committed

    def reserve(self, tenant: str, estimate_usd: float) -> Decision:
        """Decide and, unless blocked, hold `estimate_usd` against today's limit."""
        policy = self.policy_for(tenant)
        day = utc_day(self._clock())
        alerts: list[Alert] = []
        with self._lock:
            ledger = self._ledger(tenant, day)
            projected = ledger.exposure + estimate_usd
            limit = policy.daily_limit_usd
            if policy.mode != "observe" and projected > limit:
                if "blocked" not in ledger.alerted:
                    ledger.alerted.add("blocked")
                    alerts.append(Alert(tenant, day, "blocked", 1.0, ledger.committed, limit))
                decision = Decision("block", None, ledger.exposure, limit, f"projected {projected:.4f} > limit {limit:.4f}")
            else:
                action: Action = "allow"
                if policy.mode == "degrade" and projected > limit * policy.soft_limit_fraction:
                    action = "degrade"
                res = Reservation(uuid.uuid4().hex[:12], tenant, day, estimate_usd)
                ledger.reserved[res.id] = estimate_usd
                decision = Decision(action, res, ledger.exposure, limit)
        for a in alerts:
            self.on_alert(a)
        return decision

    def commit(self, reservation: Reservation, actual_usd: float) -> None:
        """Replace the reservation with the actual cost and fire any threshold alerts it crosses."""
        policy = self.policy_for(reservation.tenant)
        alerts: list[Alert] = []
        with self._lock:
            if not reservation.open:
                return
            reservation.open = False
            # The reservation's day, not today's: a request that straddles midnight is charged where it started.
            ledger = self._ledger(reservation.tenant, reservation.day)
            ledger.reserved.pop(reservation.id, None)
            ledger.committed += actual_usd
            for threshold in policy.alert_thresholds:
                tag = f"t{threshold}"
                if ledger.committed >= threshold * policy.daily_limit_usd and tag not in ledger.alerted:
                    ledger.alerted.add(tag)
                    alerts.append(Alert(reservation.tenant, reservation.day, "threshold", threshold, ledger.committed, policy.daily_limit_usd))
        for a in alerts:
            self.on_alert(a)  # outside the lock: a slow pager must not stall every request

    def release(self, reservation: Reservation) -> None:
        with self._lock:
            if reservation.open:
                reservation.open = False
                self._ledger(reservation.tenant, reservation.day).reserved.pop(reservation.id, None)


class BudgetedClient:
    """An `LLMClient` that reserves worst-case cost, degrades or blocks per policy, and commits actuals.

    The tenant comes from `req.metadata["tenant"]`; a request without one is rejected, because
    unattributed spend is exactly what a budget exists to prevent.
    """

    def __init__(
        self,
        inner: LLMClient,
        guard: SpendGuard,
        pricing: PricingTable,
        *,
        default_model: str,
        degrade_model: str | None = None,
        degrade_max_tokens: int = 512,
    ) -> None:
        self.inner = inner
        self.provider = getattr(inner, "provider", "budgeted")
        self.guard = guard
        self.pricing = pricing
        self.default_model = default_model
        self.degrade_model = degrade_model
        self.degrade_max_tokens = degrade_max_tokens

    def _priced(self, model: str) -> str:
        # An unknown model prices at $0 and would slip past every limit: refuse instead (fail closed).
        if self.pricing.lookup(model) is None:
            raise InvalidRequestError(f"no price for model {model!r}; refusing unmetered spend")
        return model

    def estimate_usd(self, req: CompletionRequest) -> float:
        """Worst case: every prompt token uncached, the full output budget used."""
        usage = Usage(input_tokens=count_message_tokens(req.messages, req.model), output_tokens=req.max_tokens)
        return self.pricing.cost_usd(self._priced(req.model or self.default_model), usage)

    def _admit(self, req: CompletionRequest) -> tuple[CompletionRequest, Reservation]:
        tenant = req.metadata.get("tenant")
        if not tenant:
            raise InvalidRequestError("request has no tenant in metadata; refusing unattributed spend")
        decision = self.guard.reserve(str(tenant), self.estimate_usd(req))
        if decision.action == "block" or decision.reservation is None:
            raise BudgetExceededError(f"daily budget exhausted for tenant {tenant}: {decision.reason}")
        reservation = decision.reservation
        if decision.action == "degrade" and self.degrade_model:
            degraded = req.model_copy(
                update={
                    "model": self.degrade_model,
                    "max_tokens": min(req.max_tokens, self.degrade_max_tokens),
                    "metadata": {**req.metadata, "budget_degraded": True},
                }
            )
            # Re-reserve at the cheaper estimate so the ledger does not hold the expensive one.
            self.guard.release(reservation)
            redo = self.guard.reserve(str(tenant), self.estimate_usd(degraded))
            if redo.reservation is None:
                raise BudgetExceededError(f"daily budget exhausted for tenant {tenant}: {redo.reason}")
            return degraded, redo.reservation
        return req, reservation

    def _actual_usd(self, completion: Completion, admitted_model: str) -> float:
        raw = completion.raw or {}
        if raw.get("cache_hit"):
            return 0.0
        if "cost_usd" in raw:
            return float(raw["cost_usd"])
        # A router may return a model id the table does not price; the call already happened, so
        # charge it at the admitted model's price rather than failing and leaking the reservation.
        model = completion.model if self.pricing.lookup(completion.model) else admitted_model
        return self.pricing.cost_usd(model, completion.usage)

    def complete(self, req: CompletionRequest) -> Completion:
        admitted, res = self._admit(req)
        try:
            completion = self.inner.complete(admitted)
        except BaseException:
            self.guard.release(res)  # assumes a failed call was not billed; see the chapter for when it is
            raise
        self.guard.commit(res, self._actual_usd(completion, admitted.model or self.default_model))
        return completion

    async def acomplete(self, req: CompletionRequest) -> Completion:
        admitted, res = self._admit(req)
        try:
            completion = await self.inner.acomplete(admitted)
        except BaseException:
            self.guard.release(res)
            raise
        self.guard.commit(res, self._actual_usd(completion, admitted.model or self.default_model))
        return completion

    def stream(self, req: CompletionRequest) -> Iterator[StreamEvent]:
        admitted, res = self._admit(req)
        model = admitted.model or self.default_model
        usage: Usage | None = None
        started = False
        try:
            for ev in self.inner.stream(admitted):
                started = True
                if ev.type == "usage" and ev.usage is not None:
                    usage = ev.usage
                yield ev
        finally:
            self._settle_stream(res, model, usage, started)

    async def astream(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:
        admitted, res = self._admit(req)
        model = admitted.model or self.default_model
        usage: Usage | None = None
        started = False
        try:
            async for ev in self.inner.astream(admitted):
                started = True
                if ev.type == "usage" and ev.usage is not None:
                    usage = ev.usage
                yield ev
        finally:
            self._settle_stream(res, model, usage, started)

    def _settle_stream(self, res: Reservation, model: str, usage: Usage | None, started: bool) -> None:
        if usage is not None:
            self.guard.commit(res, self.pricing.cost_usd(model, usage))
        elif started:
            self.guard.commit(res, res.amount_usd)  # stopped mid-stream: output was billed, amount unknown
        else:
            self.guard.release(res)                 # failed before any output (e.g. a 429): nothing billed


# ----------------------------------------------------------------------------- task tokens
@dataclass
class TaskTokenBudget:
    """Cumulative token cap for one task that makes several model calls (agent loop, chain).

    Per-request limits do not stop a loop of 40 modest calls. Charge each step's usage and check
    `can_afford` before the next step; stop with a partial result instead of overrunning.
    """

    max_input_tokens: int
    max_output_tokens: int
    used_input: int = 0
    used_output: int = 0
    steps: int = 0

    def charge(self, usage: Usage) -> None:
        self.used_input += usage.input_tokens
        self.used_output += usage.output_tokens
        self.steps += 1

    def can_afford(self, next_input_tokens: int, next_max_output: int) -> bool:
        return (
            self.used_input + next_input_tokens <= self.max_input_tokens
            and self.used_output + next_max_output <= self.max_output_tokens
        )

    def require(self, next_input_tokens: int, next_max_output: int) -> None:
        if not self.can_afford(next_input_tokens, next_max_output):
            raise BudgetExceededError(
                f"task token budget exhausted after {self.steps} steps "
                f"(input {self.used_input}/{self.max_input_tokens}, output {self.used_output}/{self.max_output_tokens})"
            )


__all__ = [
    "SpendPolicy",
    "SpendGuard",
    "Decision",
    "Reservation",
    "Alert",
    "BudgetExceededError",
    "BudgetedClient",
    "TaskTokenBudget",
    "utc_day",
]
