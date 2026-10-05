# path: book/projects/p1-extraction-api/extraction_api/api/schemas.py
"""HTTP request and response bodies that are not domain objects."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..application import BatchItem, DocumentIn
from ..application.ports import Decision


class BatchRequest(BaseModel):
    documents: list[DocumentIn] = Field(min_length=1)


class BatchSummary(BaseModel):
    total: int
    accepted: int
    human_review: int
    errors: int
    llm_calls: int
    input_tokens: int
    output_tokens: int


class BatchResponse(BaseModel):
    request_id: str
    summary: BatchSummary
    items: list[BatchItem]


class ResolveRequest(BaseModel):
    decision: Decision
    reviewer: str = Field(min_length=1, max_length=100)
    corrected_data: dict[str, Any] | None = None
    note: str | None = Field(default=None, max_length=2000)


__all__ = ["BatchRequest", "BatchSummary", "BatchResponse", "ResolveRequest"]
