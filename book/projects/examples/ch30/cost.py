# path: book/projects/examples/ch30/cost.py
"""Cost per successful task: the unit that compares architectures honestly.

`CostModel` prices a `CostScenario` (one attempt at a task: model calls, embeddings,
reranking, tool fees, retries, human review, fixed infrastructure) with an
`aie_core.PricingTable`, divides by the success rate, and projects monthly spend.
`chargeback` aggregates real spend per tenant from trace JSONL written by `JsonlTracer`.
Every price used in this chapter is illustrative.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from pydantic import BaseModel, Field, model_validator

from aie_core.llm.gateway import PricingTable
from aie_core.llm.types import Usage


class CostScenario(BaseModel):
    """One attempt at one task. Token counts are per model call; counts of calls are per attempt."""

    name: str
    model: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    model_calls: float = Field(default=1.0, ge=0)  # agent steps, chain stages, judge calls
    retry_rate: float = Field(default=0.0, ge=0)  # extra calls per call caused by retries and repairs
    embedding_tokens: int = Field(default=0, ge=0)
    embedding_price_per_1m: float = Field(default=0.0, ge=0)
    rerank_calls: float = Field(default=0.0, ge=0)
    rerank_price_per_call: float = Field(default=0.0, ge=0)
    tool_fees_usd: float = Field(default=0.0, ge=0)
    infra_usd: float = Field(default=0.0, ge=0)  # amortized self-hosted GPU, vector DB, storage per attempt
    success_rate: float = Field(default=1.0, gt=0, le=1)
    human_review_rate: float = Field(default=0.0, ge=0, le=1)
    human_review_usd: float = Field(default=0.0, ge=0)  # loaded cost of one review

    @model_validator(mode="after")
    def _cached_le_input(self) -> "CostScenario":
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("cached_input_tokens cannot exceed input_tokens")
        return self


@dataclass(frozen=True)
class CostBreakdown:
    scenario: str
    components: dict[str, float]  # USD per attempt, by component
    per_attempt: float
    per_successful_task: float

    def share(self, component: str) -> float:
        return self.components.get(component, 0.0) / self.per_attempt if self.per_attempt else 0.0

    def dominant(self) -> str:
        return max(self.components, key=lambda k: self.components[k])


@dataclass(frozen=True)
class Comparison:
    scenario: str
    per_successful_task: float
    delta_vs_baseline_pct: float


@dataclass(frozen=True)
class MonthlyProjection:
    scenario: str
    successful_tasks: float
    attempts: float
    total_usd: float
    by_component: dict[str, float]


class CostModel:
    def __init__(self, pricing: PricingTable) -> None:
        self.pricing = pricing

    def model_call_usd(self, s: CostScenario) -> float:
        usage = Usage(input_tokens=s.input_tokens, output_tokens=s.output_tokens, cached_input_tokens=s.cached_input_tokens)
        return self.pricing.cost_usd(s.model, usage)

    def breakdown(self, s: CostScenario) -> CostBreakdown:
        calls = s.model_calls * (1 + s.retry_rate)
        components = {
            "llm": self.model_call_usd(s) * calls,
            "embedding": s.embedding_tokens * s.embedding_price_per_1m / 1_000_000,
            "rerank": s.rerank_calls * s.rerank_price_per_call,
            "tools": s.tool_fees_usd,
            "infra": s.infra_usd,
            "human_review": s.human_review_rate * s.human_review_usd,
        }
        per_attempt = sum(components.values())
        # Failed attempts are paid for too. If 1 in 5 attempts fails, each success carries 1.25 attempts.
        return CostBreakdown(s.name, components, per_attempt, per_attempt / s.success_rate)

    def compare(self, baseline: CostScenario, *alternatives: CostScenario) -> list[Comparison]:
        base = self.breakdown(baseline).per_successful_task
        out = []
        for s in (baseline, *alternatives):
            cost = self.breakdown(s).per_successful_task
            out.append(Comparison(s.name, cost, (cost - base) / base * 100 if base else 0.0))
        return sorted(out, key=lambda c: c.per_successful_task)

    def what_if(self, s: CostScenario, **changes: Any) -> CostBreakdown:
        """Re-price a scenario with some fields changed, e.g. what_if(s, input_tokens=s.input_tokens + 1000)."""
        return self.breakdown(s.model_copy(update={"name": f"{s.name}*", **changes}))

    def mix(self, name: str, parts: list[tuple[float, CostScenario]]) -> CostBreakdown:
        """Price a traffic mix, e.g. a router sending 70% of tasks to a small model.

        Cost per successful task of a mix is total spend over total successes, not the average of
        each route's cost per success: a cheap route with a low success rate drags the whole mix.
        """
        weight = sum(w for w, _ in parts)
        if weight <= 0:
            raise ValueError("weights must sum to a positive number")
        components: dict[str, float] = defaultdict(float)
        successes = 0.0
        for w, s in parts:
            b = self.breakdown(s)
            for k, v in b.components.items():
                components[k] += v * w / weight
            successes += s.success_rate * w / weight
        per_attempt = sum(components.values())
        return CostBreakdown(name, dict(components), per_attempt, per_attempt / successes)

    def monthly(self, s: CostScenario, successful_tasks_per_day: float, days: int = 30) -> MonthlyProjection:
        b = self.breakdown(s)
        successes = successful_tasks_per_day * days
        attempts = successes / s.success_rate
        return MonthlyProjection(
            scenario=s.name,
            successful_tasks=successes,
            attempts=attempts,
            total_usd=attempts * b.per_attempt,
            by_component={k: v * attempts for k, v in b.components.items()},
        )


# ----------------------------------------------------------------------------- chargeback
UNATTRIBUTED = "_unattributed"


@dataclass
class TenantUsage:
    tenant: str
    spend_usd: float = 0.0
    avoided_usd: float = 0.0  # what cache hits would have cost: the cache's value, not a charge
    model_calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    other_usd: float = 0.0  # embeddings, reranking, tool fees recorded on their own spans
    tasks: int = 0
    successes: int = 0
    by_feature: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    allocated_shared_usd: float = 0.0

    @property
    def total_usd(self) -> float:
        return self.spend_usd + self.other_usd + self.allocated_shared_usd

    @property
    def cost_per_successful_task(self) -> float | None:
        return self.total_usd / self.successes if self.successes else None

    @property
    def prompt_cache_ratio(self) -> float:
        return self.cached_input_tokens / self.input_tokens if self.input_tokens else 0.0


@dataclass
class ChargebackReport:
    tenants: dict[str, TenantUsage]

    @property
    def total_usd(self) -> float:
        return sum(t.total_usd for t in self.tenants.values())

    @property
    def unattributed_share(self) -> float:
        u = self.tenants.get(UNATTRIBUTED)
        return u.total_usd / self.total_usd if u and self.total_usd else 0.0

    def allocate_shared(self, shared_usd: float) -> None:
        """Split a shared fixed cost (platform, idle GPUs, vector DB) by each tenant's share of direct spend."""
        direct = {k: t.spend_usd + t.other_usd for k, t in self.tenants.items() if k != UNATTRIBUTED}
        total = sum(direct.values())
        for k, d in direct.items():
            self.tenants[k].allocated_shared_usd = shared_usd * d / total if total else shared_usd / len(direct)


def load_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def chargeback(records: Iterable[Mapping[str, Any]], pricing: PricingTable | None = None, *, llm_span: str = "llm.complete", task_span: str = "task") -> ChargebackReport:
    """Aggregate spend per tenant from span records (the dicts `JsonlTracer` writes).

    - `llm.complete` spans carry tokens and `cost_usd` (from the gateway). A cache hit is avoided
      spend, not spend: the gateway records `cost_usd=0` and `avoided_cost_usd=<price>`. A hit span
      without `avoided_cost_usd` contributes its `cost_usd` (or the table price) instead. Either way
      the amount lands in `avoided_usd`.
    - Any other span with `cost_usd` (embedding, rerank, tool) counts toward `other_usd`.
    - `task` spans with a boolean `success` give the denominator for cost per successful task.
    - Spans without a `tenant` land in `_unattributed`; watch that share, it should be near zero.
    """
    tenants: dict[str, TenantUsage] = {}

    def usage_for(tenant: str) -> TenantUsage:
        if tenant not in tenants:
            tenants[tenant] = TenantUsage(tenant)
        return tenants[tenant]

    for rec in records:
        attrs = rec.get("attributes", {})
        t = usage_for(str(attrs.get("tenant") or UNATTRIBUTED))
        feature = str(attrs.get("feature", "unknown"))
        name = rec.get("name")
        if name == llm_span:
            cost = float(attrs.get("cost_usd") or 0.0)
            usage = Usage(
                input_tokens=int(attrs.get("input_tokens", 0)),
                output_tokens=int(attrs.get("output_tokens", 0)),
                cached_input_tokens=int(attrs.get("cached_input_tokens", 0)),
            )
            if cost == 0.0 and pricing is not None and attrs.get("model"):
                cost = pricing.cost_usd(str(attrs["model"]), usage)  # no cost_usd on the span: price it here
            if attrs.get("cache_hit"):
                t.cache_hits += 1
                avoided = attrs.get("avoided_cost_usd")
                t.avoided_usd += float(avoided) if avoided else cost  # no avoided_cost_usd: use the span's cost
                continue
            if rec.get("status") == "error" and not usage.input_tokens:
                continue  # failed before the provider billed anything
            t.model_calls += 1
            t.spend_usd += cost
            t.input_tokens += usage.input_tokens
            t.cached_input_tokens += usage.cached_input_tokens
            t.output_tokens += usage.output_tokens
            t.by_feature[feature] += cost
        elif name == task_span:
            t.tasks += 1
            if attrs.get("success") is True:
                t.successes += 1
        elif attrs.get("cost_usd"):
            cost = float(attrs["cost_usd"])
            t.other_usd += cost
            t.by_feature[feature] += cost
    return ChargebackReport(tenants)


__all__ = [
    "CostScenario",
    "CostBreakdown",
    "Comparison",
    "MonthlyProjection",
    "CostModel",
    "TenantUsage",
    "ChargebackReport",
    "chargeback",
    "load_jsonl",
    "UNATTRIBUTED",
]
