# path: book/projects/examples/ch07/router.py
"""A capability-aware model router with rules, an optional classifier, and a confidence cascade.

Decision order for every request:
  1. Derive Requirements from the request (context size, tools, schema, images, data zone).
  2. Pick a route: the first matching rule, else a classifier prediction above its confidence
     floor, else the default route.
  3. Turn the route into an ordered candidate list (primary, then fallbacks) keeping only models
     with no hard capability gap. If none remain, substitute the cheapest compatible model in
     the catalog and say so; if the catalog has none, fail loudly.
  4. Execute: call candidates in order, moving on only for retryable errors. If the route has an
     escalation target and the answer is low-confidence or invalid, call the stronger model.

Retries against one model belong to aie_core's ModelGateway (Chapter 3); wrap each client in a
gateway. The router owns the decisions a gateway cannot make: which model, and which fallbacks
are safe given what this particular request needs.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from aie_core import (Completion, CompletionRequest, InvalidRequestError, LLMClient, LLMError,
                      PricingTable, ProviderUnavailableError)
from aie_core.embeddings import EmbeddingClient, cosine_similarity, normalize
from aie_core.observability import NoopTracer, Tracer

from catalog import ModelCatalog, Requirements, capability_gaps, is_compatible


class NoCompatibleModelError(InvalidRequestError):
    """No model in the catalog can serve this request as specified. Not retryable."""


class Route(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    model: str                                  # catalog alias
    fallbacks: tuple[str, ...] = ()             # tried on retryable errors or capability gaps
    reasoning_effort: str | None = None         # passed to models that expose the knob
    escalate_to: str | None = None              # confidence cascade target
    min_confidence: float = 0.0                 # escalate when confidence is below this


@dataclass(frozen=True)
class Rule:
    name: str
    route: str
    when: Callable[[CompletionRequest, Requirements], bool]


class RouteClassifier(Protocol):
    def predict(self, req: CompletionRequest) -> tuple[str, float]:
        """Return (route name, confidence in [0, 1])."""
        ...


class EmbeddingRouteClassifier:
    """Nearest-centroid routing over labeled example requests.

    Each route gets a centroid: the normalized mean embedding of its examples. A request goes
    to the closest centroid; confidence is the relative margin over the runner-up, so a request
    halfway between two routes reports low confidence and falls through to the default route.
    """

    def __init__(self, embeddings: EmbeddingClient, examples: Mapping[str, Sequence[str]]) -> None:
        if len(examples) < 2:
            raise ValueError("need examples for at least two routes")
        self.embeddings = embeddings
        self.centroids: dict[str, list[float]] = {}
        for route, texts in examples.items():
            vectors = np.array(embeddings.embed(list(texts)), dtype=float)
            self.centroids[route] = normalize(vectors.mean(axis=0))

    def predict(self, req: CompletionRequest) -> tuple[str, float]:
        text = "\n".join(m.text for m in req.messages if m.role.value == "user")
        q = self.embeddings.embed_query(text)
        sims = sorted(((cosine_similarity(q, c), r) for r, c in self.centroids.items()), reverse=True)
        (best, route), (second, _) = sims[0], sims[1]
        if best <= 0:
            return route, 0.0
        return route, max(0.0, min(1.0, (best - second) / best))


class RouteDecision(BaseModel):
    route: str
    stage: Literal["rule", "classifier", "default"]
    rule: str | None = None
    classifier_confidence: float | None = None
    candidates: list[str]                       # aliases, in execution order, all compatible
    escalate_to: str | None = None
    escalation_disabled: bool = False           # the route has a cascade, but its target cannot serve this
    min_confidence: float = 0.0
    reasoning_effort: str | None = None
    substitutions: list[str] = Field(default_factory=list)   # why the plan differs from the route
    warnings: list[str] = Field(default_factory=list)        # soft gaps: works, but differently

    @property
    def model(self) -> str:
        return self.candidates[0]


class Attempt(BaseModel):
    model: str
    outcome: Literal["ok", "error", "low_confidence", "invalid"]
    detail: str = ""
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    confidence: float | None = None


class RoutedCompletion(BaseModel):
    completion: Completion
    decision: RouteDecision
    attempts: list[Attempt]
    served_by: str
    escalated: bool = False
    degraded: bool = False                      # escalation was wanted but failed or was impossible
    model_mismatch: bool = False                # the completion reports a model other than the alias's pin

    @property
    def cost_usd(self) -> float:
        return sum(a.cost_usd for a in self.attempts)

    @property
    def latency_ms(self) -> float:
        return sum(a.latency_ms for a in self.attempts)   # sequential calls add up


Confidence = Callable[[Completion], float]
Validator = Callable[[Completion], bool]


class Router:
    def __init__(
        self,
        catalog: ModelCatalog,
        clients: Mapping[str, LLMClient],
        routes: Sequence[Route],
        *,
        default_route: str,
        rules: Sequence[Rule] = (),
        classifier: RouteClassifier | None = None,
        classifier_min_confidence: float = 0.5,
        confidence: Confidence | None = None,
        validator: Validator | None = None,
        pricing: PricingTable | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.catalog = catalog
        self.clients = dict(clients)
        self.routes = {r.name: r for r in routes}
        self.rules = list(rules)
        self.default_route = default_route
        self.classifier = classifier
        self.classifier_min_confidence = classifier_min_confidence
        self.confidence = confidence
        self.validator = validator
        self.pricing = pricing or PricingTable(catalog.pricing())
        self.tracer = tracer or NoopTracer()
        self._check_config()

    def _check_config(self) -> None:
        """Fail at startup, not at 3 a.m., if a route names something that does not exist."""
        if self.default_route not in self.routes:
            raise ValueError(f"default route {self.default_route!r} is not defined")
        for rule in self.rules:
            if rule.route not in self.routes:
                raise ValueError(f"rule {rule.name!r} targets unknown route {rule.route!r}")
        for r in self.routes.values():
            for alias in (r.model, *r.fallbacks, *([r.escalate_to] if r.escalate_to else [])):
                self.catalog.get(alias)
                if alias not in self.clients:
                    raise ValueError(f"route {r.name!r} uses {alias!r} but no client is registered for it")

    # ------------------------------------------------------------------ decide
    def _pick_route(self, req: CompletionRequest, need: Requirements) -> tuple[Route, str, str | None, float | None]:
        for rule in self.rules:
            if rule.when(req, need):
                return self.routes[rule.route], "rule", rule.name, None
        if self.classifier is not None:
            name, conf = self.classifier.predict(req)
            if conf >= self.classifier_min_confidence and name in self.routes:
                return self.routes[name], "classifier", None, conf
            return self.routes[self.default_route], "default", None, conf
        return self.routes[self.default_route], "default", None, None

    def route(self, req: CompletionRequest) -> RouteDecision:
        need = Requirements.from_request(req)
        route, stage, rule, conf = self._pick_route(req, need)
        effort = need.reasoning_effort or route.reasoning_effort
        need = need.model_copy(update={"reasoning_effort": effort})

        substitutions: list[str] = []
        planned = [route.model, *route.fallbacks]
        candidates: list[str] = []
        for alias in planned:
            gaps = [g for g in capability_gaps(self.catalog.get(alias), need) if g.hard]
            if gaps:
                substitutions.append(f"skip {alias}: " + "; ".join(f"{g.capability}: {g.detail}" for g in gaps))
            else:
                candidates.append(alias)
        if not candidates:
            # Every planned model is unfit. Look across the whole catalog, cheapest first, but only
            # among models we can actually call.
            for p in self.catalog.compatible(need, among=self.clients):
                candidates.append(p.alias)
                substitutions.append(f"catalog substitute {p.alias} for route {route.name}")
                break
        if not candidates:
            raise NoCompatibleModelError(
                f"no model satisfies route {route.name!r}: " + " | ".join(substitutions))

        escalate_to = route.escalate_to
        escalation_disabled = False
        if escalate_to is not None and not is_compatible(self.catalog.get(escalate_to), need):
            substitutions.append(f"escalation to {escalate_to} disabled: capability gap")
            escalate_to, escalation_disabled = None, True

        # Soft gaps on any candidate: a fallback that works but changes an assumption.
        warnings = [f"{alias}: {g.detail}" for alias in candidates
                    for g in capability_gaps(self.catalog.get(alias), need) if not g.hard]
        return RouteDecision(
            route=route.name, stage=stage, rule=rule, classifier_confidence=conf,
            candidates=candidates, escalate_to=escalate_to, escalation_disabled=escalation_disabled,
            min_confidence=route.min_confidence,
            reasoning_effort=effort, substitutions=substitutions, warnings=warnings,
        )

    # ----------------------------------------------------------------- execute
    def _request_for(self, alias: str, req: CompletionRequest, decision: RouteDecision) -> CompletionRequest:
        profile = self.catalog.get(alias)
        metadata = {**req.metadata, "route": decision.route, "model_alias": alias}
        if decision.reasoning_effort and decision.reasoning_effort in profile.reasoning_efforts:
            # aie_core has no effort field; adapters that support the knob read it from metadata.
            metadata["reasoning_effort"] = decision.reasoning_effort
        else:
            metadata.pop("reasoning_effort", None)
        return req.model_copy(update={"model": profile.model_id, "metadata": metadata})

    def _call(self, alias: str, req: CompletionRequest, decision: RouteDecision) -> tuple[Completion, Attempt]:
        completion = self.clients[alias].complete(self._request_for(alias, req, decision))
        cost = self.pricing.cost_usd(completion.model, completion.usage)
        return completion, Attempt(model=alias, outcome="ok", latency_ms=completion.latency_ms, cost_usd=cost)

    def _needs_escalation(self, completion: Completion, decision: RouteDecision, attempt: Attempt) -> bool:
        if self.validator is not None and not self.validator(completion):
            attempt.outcome, attempt.detail = "invalid", "validator rejected output"
            return True
        if self.confidence is not None:
            attempt.confidence = self.confidence(completion)
            if attempt.confidence < decision.min_confidence:
                attempt.outcome = "low_confidence"
                attempt.detail = f"confidence {attempt.confidence:.2f} < {decision.min_confidence:.2f}"
                return True
        return False

    def complete(self, req: CompletionRequest) -> RoutedCompletion:
        with self.tracer.span("router.complete") as span:
            decision = self.route(req)
            span.set_attribute("router.route", decision.route)
            span.set_attribute("router.stage", decision.stage)
            span.set_attribute("router.substitutions", len(decision.substitutions))
            attempts: list[Attempt] = []
            completion: Completion | None = None
            served_by = ""
            for alias in decision.candidates:
                try:
                    completion, attempt = self._call(alias, req, decision)
                except LLMError as exc:
                    attempts.append(Attempt(model=alias, outcome="error", detail=type(exc).__name__))
                    if not exc.retryable:
                        span.record_exception(exc)
                        raise
                    continue
                attempts.append(attempt)
                served_by = alias
                break
            if completion is None:
                raise ProviderUnavailableError(
                    f"all candidates failed for route {decision.route!r}: {[a.model for a in attempts]}")

            escalated = degraded = False
            target = decision.escalate_to
            if target and target != served_by and self._needs_escalation(completion, decision, attempts[-1]):
                try:
                    strong, strong_attempt = self._call(target, req, decision)
                    attempts.append(strong_attempt)
                    completion, served_by, escalated = strong, target, True
                except LLMError as exc:
                    # Keep the cheap answer but say it is below the route's bar.
                    attempts.append(Attempt(model=target, outcome="error", detail=type(exc).__name__))
                    degraded = True
            elif decision.escalation_disabled and self._needs_escalation(completion, decision, attempts[-1]):
                # The cascade could not run for this request, and the answer is below its bar.
                degraded = True
            # A client below the router (for example a gateway with its own fallback list) may have
            # served the request from a different model. The router planned for the alias's pin; a
            # mismatch means its capability check did not cover the model that actually answered.
            mismatch = completion.model != self.catalog.get(served_by).model_id
            result = RoutedCompletion(completion=completion, decision=decision, attempts=attempts,
                                      served_by=served_by, escalated=escalated, degraded=degraded,
                                      model_mismatch=mismatch)
            span.set_attribute("router.model", served_by)
            span.set_attribute("router.model_id", completion.model)
            span.set_attribute("router.model_mismatch", mismatch)
            span.set_attribute("router.attempts", len(attempts))
            span.set_attribute("router.warnings", len(decision.warnings))
            span.set_attribute("router.escalated", escalated)
            span.set_attribute("router.degraded", degraded)
            span.set_attribute("router.cost_usd", result.cost_usd)
            return result


# ---------------------------------------------------------------------- Northwind
NARROW_TASKS = frozenset({"classify_ticket", "extract_invoice", "route_intent"})


def northwind_routes(min_confidence: float = 0.8) -> list[Route]:
    return [
        Route(name="private", model="nw-small"),
        Route(name="high_assurance", model="nw-reasoning", reasoning_effort="high"),
        Route(name="long_context", model="nw-longctx", fallbacks=("nw-reasoning",)),
        Route(name="small_first", model="nw-small", fallbacks=("nw-general",),
              escalate_to="nw-general", min_confidence=min_confidence),
        Route(name="reasoning", model="nw-reasoning", reasoning_effort="medium", fallbacks=("nw-general",)),
        Route(name="general", model="nw-general", fallbacks=("nw-reasoning",)),
    ]


def northwind_rules(long_context_threshold: int = 100_000) -> list[Rule]:
    """Rules run in order; put the ones that encode policy before the ones that save money."""
    return [
        Rule("restricted_data", "private", lambda req, need: need.data_zone == "onprem"),
        Rule("high_risk", "high_assurance", lambda req, need: req.metadata.get("risk") == "high"),
        Rule("long_context", "long_context", lambda req, need: need.min_context_tokens > long_context_threshold),
        Rule("narrow_task", "small_first", lambda req, need: req.metadata.get("task") in NARROW_TASKS),
    ]


def northwind_router(catalog: ModelCatalog, clients: Mapping[str, LLMClient], *,
                     classifier: RouteClassifier | None = None, min_confidence: float = 0.8,
                     long_context_threshold: int = 100_000, classifier_min_confidence: float = 0.5,
                     confidence: Confidence | None = None, validator: Validator | None = None,
                     tracer: Tracer | None = None) -> Router:
    return Router(catalog, clients, northwind_routes(min_confidence), default_route="general",
                  rules=northwind_rules(long_context_threshold), classifier=classifier,
                  classifier_min_confidence=classifier_min_confidence, confidence=confidence,
                  validator=validator, tracer=tracer)


__all__ = [
    "NoCompatibleModelError", "Route", "Rule", "RouteClassifier", "EmbeddingRouteClassifier",
    "RouteDecision", "Attempt", "RoutedCompletion", "Router", "northwind_routes", "northwind_rules",
    "northwind_router", "NARROW_TASKS",
]
