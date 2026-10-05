# path: book/projects/examples/ch01/lineage.py
"""Request lineage: the record you must be able to produce for any production request.

Chapter 1 introduces lineage as a set of questions. This module turns the questions into
fields, so that "can we answer this about request X?" becomes "is this field populated and
consistent?". Standard library only; later chapters replace these dataclasses with the
pydantic models and tracing spans of ``aie_core`` (see Chapter 3 and Chapter 31).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal


class EvalOutcome(str, Enum):
    """Result of the offline or online evaluation attached to this request, if any."""

    PASS = "pass"
    FAIL = "fail"
    UNSCORED = "unscored"


def content_hash(text: str) -> str:
    """Short, stable fingerprint for prompts and outputs. Lets you compare without storing text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ModelRef:
    provider: str
    name: str
    version: str | None = None  # provider snapshot or deployment tag, when the API exposes one


@dataclass(frozen=True)
class PromptRef:
    name: str  # registry key, e.g. "policy_answer"
    version: str  # registry version, e.g. "7"
    content_hash: str  # fingerprint of the rendered system prompt actually sent


@dataclass(frozen=True)
class EvidenceRef:
    source_id: str  # document identity in the knowledge base
    chunk_id: str
    score: float  # final ranking score after rerank
    acl_groups: tuple[str, ...]  # groups allowed to read this document
    tenant: str = "shared"  # document tenant tag: "shared" or one tenant such as "retail"


@dataclass(frozen=True)
class ToolEvent:
    name: str
    offered: bool  # was the tool in the request's tool list?
    called: bool  # did the model request it and did the application execute it?
    arguments_hash: str | None = None
    outcome: Literal["ok", "error", "denied", "pending_approval"] | None = None
    latency_ms: float | None = None


@dataclass(frozen=True)
class PolicyGate:
    name: str  # e.g. "evidence_gate", "tenant_filter", "output_pii_scan"
    decision: Literal["allow", "deny", "escalate"]
    reason: str | None = None


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class RequestLineage:
    """Everything you need to answer the lineage questions for one request."""

    request_id: str
    tenant: str
    principal: str  # authenticated user id
    principal_groups: tuple[str, ...]
    model: ModelRef
    prompt: PromptRef
    usage: Usage
    latency_ms: float
    output_hash: str
    output_kind: Literal["answer", "fallback", "refusal"] = "answer"  # what the user was shown
    index_version: str | None = None  # knowledge-index build that served the evidence
    evidence: list[EvidenceRef] = field(default_factory=list)
    tools: list[ToolEvent] = field(default_factory=list)
    gates: list[PolicyGate] = field(default_factory=list)
    eval_outcome: EvalOutcome = EvalOutcome.UNSCORED
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # --- the lineage questions, as code -------------------------------------------------

    def unanswered_questions(self) -> list[str]:
        """Return the lineage questions this record cannot answer. Empty list means complete."""
        missing: list[str] = []
        if not self.model.name:
            missing.append("which model was called")
        if not self.prompt.version or not self.prompt.content_hash:
            missing.append("which prompt version was used")
        if not self.output_hash:
            missing.append("what output was shown")
        if self.evidence and not self.index_version:
            missing.append("which index version served the evidence")
        if self.latency_ms <= 0:
            missing.append("how long it took")
        if self.usage.total_tokens == 0:
            missing.append("how many tokens it consumed")
        if not self.gates:
            missing.append("which policy gates ran")
        return missing

    def consistency_violations(self) -> list[str]:
        """Cross-field checks. Each violation is a production bug, not a logging gap."""
        problems: list[str] = []
        for ev in self.evidence:
            if "all" not in ev.acl_groups and not set(ev.acl_groups) & set(self.principal_groups):
                problems.append(
                    f"evidence {ev.source_id}/{ev.chunk_id} is outside the caller's groups"
                )
            if ev.tenant not in ("shared", self.tenant):
                problems.append(
                    f"evidence {ev.source_id}/{ev.chunk_id} belongs to tenant {ev.tenant}, "
                    f"not the caller's tenant {self.tenant}"
                )
        for tool in self.tools:
            if tool.called and not tool.offered:
                problems.append(f"tool {tool.name} was called but never offered to the model")
        denied = [g.name for g in self.gates if g.decision == "deny"]
        if denied and self.output_kind == "answer":
            # A deny must end in a fallback or refusal message, never in the model's answer.
            problems.append(f"gates {denied} denied but the model's answer was shown")
        return problems

    def is_complete(self) -> bool:
        return not self.unanswered_questions() and not self.consistency_violations()

    # --- serialization -----------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["eval_outcome"] = self.eval_outcome.value
        data["started_at"] = self.started_at.isoformat()
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RequestLineage":
        return cls(
            request_id=data["request_id"],
            tenant=data["tenant"],
            principal=data["principal"],
            principal_groups=tuple(data["principal_groups"]),
            model=ModelRef(**data["model"]),
            prompt=PromptRef(**data["prompt"]),
            usage=Usage(**data["usage"]),
            latency_ms=data["latency_ms"],
            output_hash=data["output_hash"],
            output_kind=data.get("output_kind", "answer"),
            index_version=data.get("index_version"),
            evidence=[
                EvidenceRef(**{**e, "acl_groups": tuple(e["acl_groups"])}) for e in data["evidence"]
            ],
            tools=[ToolEvent(**t) for t in data["tools"]],
            gates=[PolicyGate(**g) for g in data["gates"]],
            eval_outcome=EvalOutcome(data["eval_outcome"]),
            started_at=datetime.fromisoformat(data["started_at"]),
        )
