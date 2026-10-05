# path: book/projects/p5-incident-agent/incident_agent/domain/models.py
"""Typed records of an investigation. Pure data, no I/O."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

RESEARCH_TOOLS = ("query_service_metrics", "get_recent_deploys", "search_incidents", "search_runbooks")
EvidenceKind = Literal["metric", "deploy", "incident", "runbook"]


class Alert(BaseModel):
    id: str
    rule: str
    service: str
    tenant: str
    fired_at: datetime
    summary: str


class PlanStep(BaseModel):
    id: str = Field(pattern=r"^s[0-9]+$")
    tool: Literal["query_service_metrics", "get_recent_deploys", "search_incidents", "search_runbooks"]
    target: str = Field(min_length=2, description="service name for metrics/deploys, search text for searches")
    objective: str = Field(min_length=8)


class Plan(BaseModel):
    steps: list[PlanStep] = Field(min_length=1, max_length=12)
    hypothesis: str = ""


class Evidence(BaseModel):
    """One citable source the investigation actually observed."""

    id: str                     # the citation key, e.g. inc-2026-02-tracking-latency or metric:trackline.p95_ms
    kind: EvidenceKind
    title: str = ""
    text: str                   # the excerpt a reviewer or judge can check a claim against
    step: str = ""


class StepRecord(BaseModel):
    step: PlanStep
    run_id: str
    ok: bool
    stop_reason: str
    finding: str | None
    evidence_ids: list[str] = Field(default_factory=list)


class Problem(BaseModel):
    code: Literal["missing_section", "uncited_claim", "unknown_citation", "runbook_missing",
                  "runbook_not_in_catalog", "runbook_not_retrieved"]
    message: str


class RoundRecord(BaseModel):
    round: int
    score: float
    dod_problems: list[Problem] = Field(default_factory=list)
    judge: dict[str, Any] | None = None
    accepted: bool = False


class Status(str, Enum):
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"   # deterministic DoD passed; a human must approve posting
    NEEDS_REVISION = "needs_revision"         # DoD never passed within the revision budget; cannot be posted
    PUBLISHED = "published"
    REJECTED = "rejected"
    FAILED = "failed"


class Investigation(BaseModel):
    id: str
    alert: Alert
    requested_by: str
    principal: dict[str, Any]
    status: Status = Status.RUNNING
    detail: str = ""
    plans: list[Plan] = Field(default_factory=list)
    deviations: list[str] = Field(default_factory=list)
    steps: list[StepRecord] = Field(default_factory=list)
    evidence: dict[str, Evidence] = Field(default_factory=dict)
    rounds: list[RoundRecord] = Field(default_factory=list)
    report: str | None = None
    judge_passed: bool | None = None
    publish_run_id: str | None = None
    posted_message_id: str | None = None
    decided_by: str | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

    def trajectory(self) -> list[tuple[str, str]]:
        """(tool, target) per executed step, in order: the backbone of trajectory tests."""
        return [(s.step.tool, s.step.target) for s in self.steps]


__all__ = ["RESEARCH_TOOLS", "EvidenceKind", "Alert", "PlanStep", "Plan", "Evidence", "StepRecord", "Problem",
           "RoundRecord", "Status", "Investigation"]
