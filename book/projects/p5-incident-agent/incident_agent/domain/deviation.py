# path: book/projects/p5-incident-agent/incident_agent/domain/deviation.py
"""Deviation rules: deterministic reasons to replan. A replan is a model call that can make
the plan worse, so the planner is consulted again only when one of these names a concrete gap."""
from __future__ import annotations

from typing import Any, Callable

from .models import PlanStep, StepRecord

# (finished step, data returned by its tools, steps done or still queued) -> reason or None
DeviationRule = Callable[[StepRecord, list[dict[str, Any]], list[PlanStep]], str | None]


def step_failed(rec: StepRecord, data: list[dict[str, Any]], planned: list[PlanStep]) -> str | None:
    return None if rec.ok else f"step {rec.step.id} ({rec.step.tool}) did not complete: {rec.stop_reason}"


def no_evidence(rec: StepRecord, data: list[dict[str, Any]], planned: list[PlanStep]) -> str | None:
    if rec.ok and rec.step.tool.startswith("search_") and not rec.evidence_ids:
        return f"step {rec.step.id} ({rec.step.tool} '{rec.step.target}') returned no evidence"
    return None


def uncovered_dependency(rec: StepRecord, data: list[dict[str, Any]], planned: list[PlanStep]) -> str | None:
    """A metrics step found an anomalous dependency that no planned step will look at."""
    covered = {s.target for s in planned if s.tool == "query_service_metrics"}
    for d in data:
        for dep in d.get("anomalous_dependencies", []):
            if dep not in covered:
                return (f"dependency {dep} of {rec.step.target} is anomalous and no step examines it; "
                        f"add query_service_metrics and get_recent_deploys for {dep}")
    return None


DEFAULT_RULES: tuple[DeviationRule, ...] = (step_failed, no_evidence, uncovered_dependency)

__all__ = ["DeviationRule", "step_failed", "no_evidence", "uncovered_dependency", "DEFAULT_RULES"]
