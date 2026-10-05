# path: book/projects/p1-extraction-api/extraction_api/application/ports.py
"""The review queue port. Adapters (in-memory, SQLite, later Postgres) implement it."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from ..domain.common import DocumentType, RuleViolation

ReviewStatus = Literal["pending", "approved", "corrected", "rejected"]
Decision = Literal["approve", "correct", "reject"]
_STATUS_FOR: dict[str, ReviewStatus] = {"approve": "approved", "correct": "corrected", "reject": "rejected"}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ReviewResolution(BaseModel):
    decision: Decision
    reviewer: str = Field(min_length=1, max_length=100)
    corrected_data: dict[str, Any] | None = None
    note: str | None = Field(default=None, max_length=2000)
    resolved_at: datetime = Field(default_factory=utcnow)

    @property
    def status(self) -> ReviewStatus:
        return _STATUS_FOR[self.decision]


class ReviewItem(BaseModel):
    review_id: str
    request_id: str
    document_id: str | None = None
    tenant: str | None = None            # review visibility is scoped by tenant
    doc_type: DocumentType
    reasons: list[str]
    violations: list[RuleViolation] = Field(default_factory=list)
    data: dict[str, Any] | None = None   # the machine's best attempt, shown to the reviewer
    document_text: str                   # needed to review; subject to retention policy
    prompt_version: str
    created_at: datetime = Field(default_factory=utcnow)
    status: ReviewStatus = "pending"
    resolution: ReviewResolution | None = None


class ReviewNotFound(KeyError):
    pass


class AlreadyResolved(RuntimeError):
    pass


class ReviewQueue(Protocol):
    def enqueue(self, item: ReviewItem) -> ReviewItem: ...
    def get(self, review_id: str) -> ReviewItem: ...
    def list(
        self, status: ReviewStatus | None = "pending", limit: int = 50, *, tenant: str | None = None
    ) -> list[ReviewItem]: ...   # tenant=None: all tenants (callers must scope it)
    def resolve(self, review_id: str, resolution: ReviewResolution) -> ReviewItem: ...
    def counts(self) -> dict[str, int]: ...


__all__ = [
    "ReviewStatus", "Decision", "ReviewResolution", "ReviewItem", "ReviewNotFound",
    "AlreadyResolved", "ReviewQueue", "utcnow",
]
