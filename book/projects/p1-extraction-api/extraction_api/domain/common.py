# path: book/projects/p1-extraction-api/extraction_api/domain/common.py
"""Types shared by every document type: routing outcomes, rule violations, evidence spans."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    INVOICE = "invoice"
    SUPPORT_TICKET = "support_ticket"
    OTHER = "other"


class Route(str, Enum):
    ACCEPT = "accept"              # safe to hand to downstream systems
    REPAIR = "repair"              # one more targeted model attempt is worth it
    HUMAN_REVIEW = "human_review"  # a person decides


class Severity(str, Enum):
    ERROR = "error"      # blocks automatic acceptance
    WARNING = "warning"  # recorded, does not block


class RuleViolation(BaseModel):
    """One failed check. `repairable` means a re-ask could plausibly fix it (the model may
    have misread the document); non-repairable violations go straight to a human."""

    code: str
    field: str | None = None
    detail: str
    severity: Severity = Severity.ERROR
    repairable: bool = False


class EvidenceSpan(BaseModel):
    """Where in the source text a field's value came from. Offsets are computed by code,
    never trusted from the model; `start is None` means the quote was not found."""

    field: str
    quote: str
    start: int | None = None
    end: int | None = None
    model_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def found(self) -> bool:
        return self.start is not None


__all__ = ["DocumentType", "Route", "Severity", "RuleViolation", "EvidenceSpan"]
