# path: book/capstone/northwind-assist/northwind_assist/llm/models.py
"""The model layer: provider client -> circuit breaker -> ModelGateway -> Chapter 7 router.

Order of wrapping follows Chapter 29: the breaker sits *inside* the gateway, around the raw
provider client, so an open circuit raises a retryable CircuitOpenError that the gateway turns
into an immediate fallback (or failure) instead of a backoff that burns the deadline. The
router (Chapter 7) decides which catalog alias serves a request; the gateway (Chapter 3) owns
retries, timeouts, concurrency and per-attempt spans. `RoutedClient` is the per-request view:
it pins the routed model id, applies the degraded plan's output cap and the request deadline,
and meters usage into the request's cost record, including on streams.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from aie_core.llm.client import LLMClient
from aie_core.llm.gateway import ModelGateway, PricingTable, RetryPolicy
from aie_core.llm.types import Completion, CompletionRequest, StreamEvent, ToolSpec, Usage
from aie_core.settings import Settings as CoreSettings
from aie_core.settings import make_provider_client
from catalog import ModelCatalog, northwind_catalog  # type: ignore[import-not-found]
from reliability import CircuitBreakerClient, CircuitBreakerRegistry, Deadline, DegradedPlan
from router import RouteDecision, Router, northwind_router  # type: ignore[import-not-found]

from ..config import Settings
from .demo import make_demo_llm

PRIMARY = "llm:primary"
BACKUP = "llm:backup"


def provider_client(core: CoreSettings | None = None) -> LLMClient:
    """The bare provider adapter (aie_core.make_provider_client): no gateway, no tracer of its own.
    The capstone composes its own gateway around it, so a request has one retry loop and one tracer."""
    core = core or CoreSettings()
    if core.llm_provider == "fake":
        return make_demo_llm()
    return make_provider_client(core)


def fallback_client(core: CoreSettings | None = None) -> LLMClient | None:
    """Same-capability backup (LLM_FALLBACK_MODEL), tried by the gateway on retryable failures."""
    core = core or CoreSettings()
    if core.llm_provider == "fake" or not core.llm_fallback_model:
        return None
    return make_provider_client(core, model=core.llm_fallback_model)


def catalog_for(settings: Settings) -> ModelCatalog:
    """Chapter 7's illustrative catalog, with model ids replaced by NA_MODEL_MAP for real providers."""
    cat = northwind_catalog()
    if settings.model_map:
        profiles = [p.model_copy(update={"model_id": settings.model_map.get(p.alias, p.model_id)})
                    for p in cat.profiles.values()]
        cat = ModelCatalog.from_profiles(profiles)
    return cat


@dataclass
class UsageRecord:
    model: str
    usage: Usage
    cost_usd: float
    purpose: str


@dataclass
class UsageMeter:
    """Everything one request spent on models. Summed into the cost ledger at the end."""

    records: list[UsageRecord] = field(default_factory=list)

    def add(self, model: str, usage: Usage, cost: float, purpose: str) -> None:
        self.records.append(UsageRecord(model, usage, cost, purpose))

    @property
    def cost_usd(self) -> float:
        return round(sum(r.cost_usd for r in self.records), 8)

    @property
    def input_tokens(self) -> int:
        return sum(r.usage.input_tokens for r in self.records)

    @property
    def output_tokens(self) -> int:
        return sum(r.usage.output_tokens for r in self.records)

    @property
    def models(self) -> list[str]:
        return sorted({r.model for r in self.records})


class RoutedClient:
    """An LLMClient bound to one routing decision, one deadline, one degraded plan, one meter."""

    def __init__(self, inner: LLMClient, *, model_id: str, alias: str, pricing: PricingTable, meter: UsageMeter,
                 deadline: Deadline | None, plan: DegradedPlan | None, purpose: str) -> None:
        self.inner = inner
        self.provider = getattr(inner, "provider", "gateway")
        self.model_id = model_id
        self.alias = alias
        self.pricing = pricing
        self.meter = meter
        self.deadline = deadline
        self.plan = plan
        self.purpose = purpose
        self.supports_response_schema = getattr(inner, "supports_response_schema", False)

    def _prepare(self, req: CompletionRequest) -> CompletionRequest:
        update: dict[str, Any] = {"model": self.model_id,
                                  "metadata": {**req.metadata, "route.alias": self.alias, "purpose":
                                               req.metadata.get("purpose", self.purpose)}}
        if self.plan is not None:
            req = self.plan.apply(req)
        req = req.model_copy(update=update)
        return self.deadline.apply(req) if self.deadline is not None else req

    def _record(self, model: str, usage: Usage, raw: dict[str, Any] | None = None) -> None:
        raw_cost = (raw or {}).get("cost_usd")
        cost = float(raw_cost) if raw_cost is not None else self.pricing.cost_usd(model, usage)
        self.meter.add(model, usage, cost, self.purpose)

    def complete(self, req: CompletionRequest) -> Completion:
        completion = self.inner.complete(self._prepare(req))
        self._record(completion.model, completion.usage, completion.raw)
        raw = dict(completion.raw or {})
        raw.setdefault("cost_usd", self.meter.records[-1].cost_usd)
        return completion.model_copy(update={"raw": raw})

    def stream(self, req: CompletionRequest) -> Iterator[StreamEvent]:
        prepared = self._prepare(req)
        usage: Usage | None = None
        for ev in self.inner.stream(prepared):
            if ev.type == "usage" and ev.usage is not None:
                usage = ev.usage
            yield ev
        self._record(self.model_id, usage or Usage())

    async def acomplete(self, req: CompletionRequest) -> Completion:  # pragma: no cover - sync service
        return self.complete(req)

    async def astream(self, req: CompletionRequest):  # pragma: no cover - sync service
        for ev in self.stream(req):
            yield ev


class ModelLayer:
    def __init__(self, settings: Settings, *, client: LLMClient | None = None, tracer: Any = None,
                 breakers: CircuitBreakerRegistry | None = None, backup: LLMClient | None = None) -> None:
        self.settings = settings
        self.catalog = catalog_for(settings)
        self.pricing = PricingTable(self.catalog.pricing())
        self.breakers = breakers or CircuitBreakerRegistry(
            failure_rate_threshold=settings.breaker_failure_rate, min_calls=settings.breaker_min_calls,
            open_s=settings.breaker_open_s)
        self.raw = client or provider_client()
        guarded = CircuitBreakerClient(self.raw, self.breakers.get(PRIMARY))
        backup = backup if backup is not None else (fallback_client() if client is None else None)
        fallbacks = [CircuitBreakerClient(backup, self.breakers.get(BACKUP))] if backup is not None else []
        # Degraded plans count a model dependency as lost only when *every* one has an open breaker.
        self.model_dependencies: tuple[str, ...] = (PRIMARY, BACKUP) if fallbacks else (PRIMARY,)
        self.gateway = ModelGateway(guarded, fallbacks=fallbacks,
                                    retry=RetryPolicy(max_attempts=2, base_delay_s=0.05, max_delay_s=0.5),
                                    pricing=self.pricing, tracer=tracer, default_timeout_s=settings.request_deadline_s)
        clients = {alias: self.gateway for alias in self.catalog.profiles}
        self.router: Router = northwind_router(self.catalog, clients, tracer=tracer)

    def decide(self, *, task: str, needs_tools: bool = False, risk: str | None = None,
               plan: DegradedPlan | None = None) -> RouteDecision:
        """Route on the request's *shape*; the deterministic intent router already chose the task."""
        tools = [ToolSpec(name="probe", description="capability probe", parameters={"type": "object"})] \
            if needs_tools else None
        probe = CompletionRequest(messages=[], tools=tools, metadata={"task": task, **({"risk": risk} if risk else {})})
        decision = self.router.route(probe)
        if plan is not None and plan.model_tier == "small" and not needs_tools:
            decision = decision.model_copy(update={"candidates": ["nw-small", *decision.candidates],
                                                   "substitutions": [*decision.substitutions, "degraded: small tier"]})
        return decision

    def client_for(self, decision: RouteDecision, *, meter: UsageMeter, deadline: Deadline | None,
                   plan: DegradedPlan | None, purpose: str) -> RoutedClient:
        alias = decision.model
        return RoutedClient(self.gateway, model_id=self.catalog.get(alias).model_id, alias=alias,
                            pricing=self.pricing, meter=meter, deadline=deadline, plan=plan, purpose=purpose)

    def open_dependencies(self) -> set[str]:
        return set(self.breakers.open_dependencies())


__all__ = ["ModelLayer", "RoutedClient", "UsageMeter", "UsageRecord", "provider_client", "fallback_client",
           "catalog_for", "PRIMARY", "BACKUP"]
