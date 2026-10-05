# path: book/projects/p1-extraction-api/extraction_api/domain/routing.py
"""Routing: turn violations and a confidence score into accept, repair, or human review.

The score is not the model's self-reported confidence. Verbalized confidence is poorly
calibrated, so it is only one input: a field whose quote cannot be found in the document
scores zero regardless of what the model claimed. The threshold that separates accept from
review is chosen offline on labeled data (see eval/calibration.py), not guessed.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from .common import EvidenceSpan, Route, RuleViolation, Severity


class RoutingPolicy(BaseModel):
    accept_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    max_rule_repairs: int = Field(default=1, ge=0)


class RouteDecision(BaseModel):
    route: Route
    reasons: list[str] = Field(default_factory=list)


def field_scores(spans: list[EvidenceSpan], fields: tuple[str, ...], present: set[str]) -> dict[str, float]:
    """Per-field support: the model's confidence if its quote is in the text, else 0."""
    by_field = {s.field: s for s in spans}
    scores: dict[str, float] = {}
    for name in fields:
        if name not in present:
            continue
        span = by_field.get(name)
        scores[name] = span.model_confidence if span is not None and span.found else 0.0
    return scores


def document_score(scores: dict[str, float]) -> float:
    """A document is as trustworthy as its weakest critical field."""
    return min(scores.values()) if scores else 0.0


def decide(
    violations: list[RuleViolation],
    score: float,
    repairs_used: int,
    policy: RoutingPolicy,
) -> RouteDecision:
    errors = [v for v in violations if v.severity is Severity.ERROR]
    if errors:
        codes = sorted({v.code for v in errors})
        if repairs_used < policy.max_rule_repairs and any(v.repairable for v in errors):
            return RouteDecision(route=Route.REPAIR, reasons=codes)
        return RouteDecision(route=Route.HUMAN_REVIEW, reasons=codes)
    if score < policy.accept_threshold:
        return RouteDecision(route=Route.HUMAN_REVIEW, reasons=[f"LOW_CONFIDENCE:{score:.2f}"])
    return RouteDecision(route=Route.ACCEPT)


__all__ = ["RoutingPolicy", "RouteDecision", "field_scores", "document_score", "decide"]
