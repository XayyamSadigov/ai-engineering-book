# path: book/projects/p6-research-team/research_team/contracts.py
"""Message contracts between agents: typed task and result envelopes plus the payload schemas.

Agents never exchange free-form chat. A parent sends a TaskEnvelope (who, what, with which
budget, which output schema, which tools) and receives a ResultEnvelope (status, validated
output, evidence, usage, errors). Every envelope carries task_id, parent_id, and trace_id so
the event logs of all agents can be stitched into one tree after the fact.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agentkit import Budget


class Role(str, Enum):
    PLANNER = "planner"            # the supervisor, decomposition phase
    RESEARCHER = "researcher"
    VERIFIER = "verifier"
    SYNTHESIZER = "synthesizer"    # the supervisor, synthesis phase
    SINGLE = "single"              # the single-agent baseline


class TaskStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BUDGET_EXHAUSTED = "budget_exhausted"   # the child hit its own slice
    SKIPPED = "skipped"                     # never started: spawn cap, global budget, duplicate, deadline


class BudgetSlice(BaseModel):
    """The part of the global budget granted to one child. Converted to an agentkit Budget."""

    model_config = ConfigDict(frozen=True)

    max_steps: int = Field(default=6, ge=1)
    max_tokens: int = Field(default=12_000, ge=1)
    max_cost_usd: float | None = Field(default=None, gt=0)
    deadline_s: float | None = Field(default=None, gt=0)
    max_tool_calls: int | None = Field(default=8, ge=0)

    def to_agent_budget(self) -> Budget:
        return Budget(max_steps=self.max_steps, max_tokens=self.max_tokens, max_cost_usd=self.max_cost_usd,
                      deadline_s=self.deadline_s, max_tool_calls=self.max_tool_calls)


class ErrorInfo(BaseModel):
    code: str                       # stop reason or refusal reason, machine-readable
    message: str
    retryable: bool = False


class AgentUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    steps: int = 0
    tool_calls: int = 0
    model_calls: int = 0
    latency_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: "AgentUsage") -> "AgentUsage":
        return AgentUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=round(self.cost_usd + other.cost_usd, 8),
            steps=self.steps + other.steps,
            tool_calls=self.tool_calls + other.tool_calls,
            model_calls=self.model_calls + other.model_calls,
            latency_ms=self.latency_ms + other.latency_ms,
        )


# ----------------------------------------------------------------------------- payload schemas
class SubQuestion(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9_-]{1,24}$")
    question: str = Field(min_length=5, max_length=400)
    topic: str = ""                 # short label for the answer section, e.g. "Business Travel Policy"
    rationale: str = ""


class Plan(BaseModel):
    """Planner output. The size bound is deliberately loose; the spawn cap is the real limit."""

    subquestions: list[SubQuestion] = Field(min_length=1, max_length=12)


class EvidenceRef(BaseModel):
    doc_id: str
    passage_id: str
    quote: str = Field(default="", max_length=600)


class Claim(BaseModel):
    claim_id: str = ""
    text: str = Field(min_length=3, max_length=600)
    evidence: list[EvidenceRef] = Field(min_length=1)
    source_task: str = ""           # filled by the supervisor, never by the model


class ResearchFindings(BaseModel):
    subquestion: str
    claims: list[Claim] = Field(default_factory=list, max_length=12)
    gaps: list[str] = Field(default_factory=list)


class ClaimVerdict(BaseModel):
    claim_id: str
    supported: bool
    reason: str = ""


class VerificationReport(BaseModel):
    verdicts: list[ClaimVerdict]


class RejectedClaim(BaseModel):
    claim: Claim
    reason: str
    rejected_by: Literal["verifier", "deterministic", "both", "unverified"]


class Conflict(BaseModel):
    claim_ids: tuple[str, str]
    doc_ids: tuple[str, str]
    unit: str
    values: tuple[list[str], list[str]]
    preferred_doc: str              # the more recently updated document
    reason: str


# ----------------------------------------------------------------------------- envelopes
def objective_key(objective: str) -> str:
    """Normalized hash of an objective; equal keys mean duplicated work."""
    norm = " ".join(re.findall(r"[a-z0-9]+", objective.lower()))
    return hashlib.sha256(norm.encode()).hexdigest()[:16]


class TaskEnvelope(BaseModel):
    """What a parent sends to a child. `principal` is trusted and never rendered to the model."""

    model_config = ConfigDict(frozen=True)

    task_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    parent_id: str | None
    trace_id: str
    parent_span_id: str | None = None
    sender: str
    recipient: Role
    depth: int = Field(default=1, ge=0)
    objective: str
    constraints: list[str] = Field(default_factory=list)
    inputs: dict[str, Any] = Field(default_factory=dict)
    output_schema: str                                   # name of the payload model, e.g. "ResearchFindings"
    allowed_tools: list[str] = Field(default_factory=list)
    budget: BudgetSlice
    principal: dict[str, Any] = Field(default_factory=dict, exclude=True)
    created_at: float = Field(default_factory=time.time)

    @property
    def objective_key(self) -> str:
        return objective_key(self.objective)

    def render(self) -> str:
        """The goal text the child agent sees: the task as JSON plus the output schema."""
        schema = SCHEMAS.get(self.output_schema)
        body = {
            "task_id": self.task_id,
            "objective": self.objective,
            "constraints": self.constraints,
            "inputs": self.inputs,
            "allowed_tools": self.allowed_tools,
            "budget": {"max_steps": self.budget.max_steps, "max_tokens": self.budget.max_tokens},
            "output_schema": self.output_schema,
        }
        text = "TASK\n" + json.dumps(body, ensure_ascii=False, indent=1)
        if schema is not None:
            text += "\nOUTPUT JSON SCHEMA\n" + json.dumps(schema.model_json_schema(), ensure_ascii=False)
        return text


class ResultEnvelope(BaseModel):
    """What a child returns. `output` has been validated against the requested schema."""

    task_id: str
    parent_id: str | None
    trace_id: str
    sender: Role
    run_id: str | None                     # the child's agentkit event log; None if it never ran
    status: TaskStatus
    stop_reason: str | None = None
    output: dict[str, Any] | None = None
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    usage: AgentUsage = Field(default_factory=AgentUsage)
    errors: list[ErrorInfo] = Field(default_factory=list)
    objective: str = ""

    @property
    def ok(self) -> bool:
        return self.status is TaskStatus.SUCCEEDED


class AnswerReport(BaseModel):
    """The final product of either architecture, so the benchmark can compare like with like."""

    architecture: str
    question: str
    trace_id: str
    status: Literal["complete", "partial", "failed"]
    answer: str
    accepted_claims: list[Claim] = Field(default_factory=list)
    rejected_claims: list[RejectedClaim] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    children: list[ResultEnvelope] = Field(default_factory=list)
    usage: AgentUsage = Field(default_factory=AgentUsage)
    wall_ms: float = 0.0
    duplicate_claims: int = 0          # claims found by more than one worker (duplicated work)
    notes: list[str] = Field(default_factory=list)


SCHEMAS: dict[str, type[BaseModel]] = {
    "Plan": Plan,
    "ResearchFindings": ResearchFindings,
    "VerificationReport": VerificationReport,
}

__all__ = [
    "AgentUsage", "AnswerReport", "BudgetSlice", "Claim", "ClaimVerdict", "Conflict", "ErrorInfo", "EvidenceRef",
    "Plan", "RejectedClaim", "ResearchFindings", "ResultEnvelope", "Role", "SCHEMAS", "SubQuestion", "TaskEnvelope",
    "TaskStatus", "VerificationReport", "objective_key",
]
