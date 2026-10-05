# path: book/projects/reliability/reliability/degrade.py
"""Graceful degradation: named, pre-tested plans that trade quality for availability.

A degraded mode is a product decision written down before the incident, not an improvised
`if` added during one. Each level is a `DegradedPlan`: which model tier, how much retrieval,
whether to rerank, how long the answer may be, whether agents and side-effecting tools are
allowed, and what the user is told. `DegradePolicy.resolve` combines three inputs into one
plan: the admission controller's level, the circuit breakers that are open, and operator
switches (read-only mode, a forced floor). Every plan must have an evaluation run on file
(Chapter 25), or you do not know what quality you are degrading *to*.
"""
from __future__ import annotations

from collections.abc import Iterable
from enum import IntEnum

from aie_core.llm.types import CompletionRequest
from pydantic import BaseModel


class DegradeLevel(IntEnum):
    NORMAL = 0     # full pipeline
    REDUCED = 1    # cheaper: smaller retrieval, no rerank, shorter answers, same model
    MINIMAL = 2    # small model, minimal retrieval, no agents
    STATIC = 3     # no model: cached answers, canned FAQ, or "try again later" with a ticket link


class DegradedPlan(BaseModel):
    level: DegradeLevel
    model_tier: str = "primary"            # resolved to a concrete model by the router (Chapter 7)
    retrieval_k: int = 8
    rerank: bool = True
    max_output_tokens: int = 1024
    allow_agents: bool = True
    allow_side_effects: bool = True
    use_model: bool = True
    user_notice: str | None = None
    reasons: list[str] = []

    def allows_tool(self, side_effecting: bool) -> bool:
        return self.use_model and (self.allow_side_effects or not side_effecting)

    def apply(self, req: CompletionRequest, models: dict[str, str] | None = None) -> CompletionRequest:
        """Stamp the plan on a request: model for the tier, output cap, and telemetry metadata."""
        update: dict[str, object] = {
            "max_tokens": min(req.max_tokens, self.max_output_tokens),
            "metadata": {**req.metadata, "degrade_level": int(self.level), "degrade_reasons": list(self.reasons)},
        }
        if models and self.model_tier in models:
            update["model"] = models[self.model_tier]
        return req.model_copy(update=update)


DEFAULT_PLANS: dict[DegradeLevel, DegradedPlan] = {
    DegradeLevel.NORMAL: DegradedPlan(level=DegradeLevel.NORMAL),
    DegradeLevel.REDUCED: DegradedPlan(level=DegradeLevel.REDUCED, retrieval_k=4, rerank=False, max_output_tokens=512),
    DegradeLevel.MINIMAL: DegradedPlan(
        level=DegradeLevel.MINIMAL, model_tier="small", retrieval_k=3, rerank=False, max_output_tokens=384,
        allow_agents=False, user_notice="Northwind Assist is running in a reduced mode; answers may be shorter.",
    ),
    DegradeLevel.STATIC: DegradedPlan(
        level=DegradeLevel.STATIC, retrieval_k=0, rerank=False, max_output_tokens=0, allow_agents=False,
        allow_side_effects=False, use_model=False,
        user_notice="Northwind Assist is temporarily unavailable. Matching help articles are listed below.",
    ),
}


class DegradePolicy:
    """Maps signals to a plan. Dependency names follow the breaker registry: 'llm:primary', 'rerank', ...

    `model_dependencies` are same-capability providers behind the gateway. Switching to a
    *different* model (smaller, shorter context, no tools) is a routing decision with
    compatibility checks (Chapter 7); here it only happens through a plan's `model_tier`,
    which the router resolves.
    """

    def __init__(
        self,
        plans: dict[DegradeLevel, DegradedPlan] | None = None,
        *,
        model_dependencies: tuple[str, ...] = ("llm:primary", "llm:backup"),
        rerank_dependency: str = "rerank",
        retrieval_dependency: str = "retrieval",
    ) -> None:
        self.plans = plans or DEFAULT_PLANS
        self.model_dependencies = model_dependencies
        self.rerank_dependency = rerank_dependency
        self.retrieval_dependency = retrieval_dependency
        self.read_only = False       # operator switch: no side effects anywhere
        self.forced_level = DegradeLevel.NORMAL

    def resolve(self, admission_level: int = 0, open_dependencies: Iterable[str] = ()) -> DegradedPlan:
        open_deps = set(open_dependencies)
        reasons: list[str] = []
        level = DegradeLevel(max(int(self.forced_level), min(int(admission_level), int(DegradeLevel.STATIC))))
        if admission_level:
            reasons.append(f"admission:{admission_level}")
        if self.forced_level:
            reasons.append(f"operator:{int(self.forced_level)}")
        open_models = [d for d in self.model_dependencies if d in open_deps]
        if open_models and len(open_models) == len(self.model_dependencies):
            level = DegradeLevel.STATIC
            reasons.append("all_models_open")
        elif open_models and level < DegradeLevel.REDUCED:
            # The surviving provider now carries all traffic: shrink each request so it can.
            level = DegradeLevel.REDUCED
            reasons.extend(f"open:{d}" for d in open_models)

        plan = self.plans[level].model_copy(deep=True)
        if self.rerank_dependency in open_deps and plan.rerank:
            plan.rerank = False
            reasons.append(f"open:{self.rerank_dependency}")
        if self.retrieval_dependency in open_deps and plan.retrieval_k:
            plan.retrieval_k = 0  # answer without evidence only if the product allows it; Chapter 13 abstains
            reasons.append(f"open:{self.retrieval_dependency}")
        if self.read_only:
            plan.allow_side_effects = False
            plan.user_notice = plan.user_notice or "Actions are temporarily disabled; answers are still available."
            reasons.append("read_only")
        plan.reasons = reasons
        return plan


__all__ = ["DegradeLevel", "DegradedPlan", "DegradePolicy", "DEFAULT_PLANS"]
