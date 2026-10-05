# path: book/projects/ragkit/ragkit/generation/schema.py
"""The grounded answer contract: what the model must return and what the UI receives.

The model returns a `GroundedAnswer`. Code validates it (validator.py), resolves evidence ids
to real sources, and wraps the result in an `AnswerEnvelope` for the client. The model never
produces URLs, titles or document ids; it produces evidence ids such as "E2", and the
application maps them to stable source metadata after generation.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .support import EID_RE, markers

AnswerStatus = Literal["answered", "partial", "insufficient_evidence", "conflict"]
Confidence = Literal["high", "medium", "low"]


class Claim(BaseModel):
    """One atomic factual statement and the evidence that supports it."""

    text: str = Field(description="One factual statement, without citation markers.")
    citations: list[str] = Field(
        default_factory=list, description="Evidence ids that support this claim, for example ['E1']."
    )
    quote: str | None = Field(
        default=None,
        description="Optional short verbatim quote from a cited evidence block that supports the claim.",
    )

    @field_validator("citations", mode="before")
    @classmethod
    def _normalize_ids(cls, v: object) -> object:
        # Models sometimes write "[E1]" or "e1"; accept the id, reject everything else later.
        if isinstance(v, list):
            out: list[str] = []
            for item in v:
                found = EID_RE.findall(str(item).upper())
                out.extend(found or [str(item)])
            return list(dict.fromkeys(out))
        return v


class GroundedAnswer(BaseModel):
    """Structured output of the generator. Field descriptions double as instructions to the model."""

    status: AnswerStatus = Field(
        description=(
            "answered: every part of the question is supported. partial: some parts are supported, "
            "others are listed in missing_info. insufficient_evidence: the evidence does not answer the "
            "question. conflict: sources disagree; answer with the newer source and describe the conflict."
        )
    )
    answer: str = Field(description="2-6 sentences for the user. End each factual sentence with markers like [E1].")
    claims: list[Claim] = Field(default_factory=list, description="Every factual statement in the answer.")
    missing_info: list[str] = Field(
        default_factory=list, description="What the user asked that the evidence does not cover."
    )
    conflicts: list[str] = Field(
        default_factory=list, description="One line per disagreement between sources, naming both ids."
    )
    confidence: Confidence = Field(default="medium", description="How directly the evidence answers the question.")

    @property
    def cited_ids(self) -> list[str]:
        """Union of claim citations and inline markers, first-seen order."""
        seen: dict[str, None] = {}
        for eid in markers(self.answer):
            seen.setdefault(eid, None)
        for claim in self.claims:
            for eid in claim.citations:
                seen.setdefault(eid, None)
        return list(seen)

    @classmethod
    def insufficient(cls, missing: str, *, answer: str | None = None) -> "GroundedAnswer":
        return cls(
            status="insufficient_evidence",
            answer=answer or "I could not find this in the documents available to you.",
            claims=[],
            missing_info=[missing],
            confidence="low",
        )


class ResolvedCitation(BaseModel):
    """An evidence id mapped back to stable source metadata, computed by code after generation."""

    eid: str
    chunk_ids: list[str]
    doc_id: str
    version: str
    title: str
    source_uri: str | None = None
    section: str = ""
    updated_at: str | None = None
    flagged: bool = False  # the block contained instruction-like text (shown with a warning badge)


Severity = Literal["error", "warning"]
IssueCode = Literal[
    "unknown_citation",  # cited id was never shown to the model
    "uncited_claim",  # claim has no valid citation
    "unsupported_claim",  # cited evidence does not contain the claim's content or numbers
    "support_only_flagged",  # the only support is instruction-like text inside a flagged block
    "quote_not_found",  # quote is not a verbatim span of any cited block
    "judge_unsupported",  # the LLM judge hook marked the claim unsupported
    "uncited_sentence",  # answer prose with factual content but no marker
    "status_inconsistent",  # e.g. answered with zero claims
    "stale_source_preferred",  # cites the older side of a detected conflict without the newer side
    "conflict_unreported",  # cites both sides of a conflict but status is not conflict
    "cites_flagged_source",  # cites a block flagged for instruction-like content
]


class ValidationIssue(BaseModel):
    code: IssueCode
    severity: Severity
    detail: str
    claim_index: int | None = None
    eid: str | None = None


class AnswerEnvelope(BaseModel):
    """What the API returns to a UI: renderable text, resolved sources, and machine-readable state.

    `text` keeps [E#] markers so the client can render them as clickable chips that open
    `citations[eid]`. `action` tells the client which layout to use (answer, caveat banner,
    abstention card with suggested next steps, or escalation notice).
    """

    request_id: str
    status: AnswerStatus
    action: Literal["answer", "answer_with_caveat", "abstain", "escalate"]
    text: str
    citations: list[ResolvedCitation] = Field(default_factory=list)
    missing_info: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    notices: list[str] = Field(default_factory=list)  # user-safe caveats (stale source, partial answer)
    confidence: Confidence = "low"
    issues: list[ValidationIssue] = Field(default_factory=list)  # for logs and internal UIs, not end users
    prompt_version: str | None = None


__all__ = [
    "AnswerEnvelope",
    "AnswerStatus",
    "Claim",
    "Confidence",
    "GroundedAnswer",
    "IssueCode",
    "ResolvedCitation",
    "Severity",
    "ValidationIssue",
]
