# path: book/projects/ragkit/ragkit/retrieval/types.py
"""Shared retrieval contract for Chapters 12-15 and Project 3.

Every retriever, fuser and reranker in ragkit speaks these types, so stages can be
recombined and evaluated independently (Ch 14 measures each stage on the same objects).
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ..documents import Chunk


class Principal(BaseModel):
    """Who is asking. Retrieval filters by this before anything reaches a model."""

    user_id: str
    tenant: str
    groups: list[str] = Field(default_factory=lambda: ["all"])


class RetrievalQuery(BaseModel):
    text: str
    principal: Principal
    k: int = 10
    filters: dict[str, Any] = Field(default_factory=dict)  # e.g. {"tags_any": [...], "doc_ids": [...]}
    original_text: str | None = None  # the user's raw question when `text` was rewritten


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    stage: str  # "bm25" | "dense" | "fusion" | "rerank" | ...
    rank: int  # 1-based rank within the producing stage
    signals: dict[str, float] = Field(default_factory=dict)  # per-stage scores kept for debugging


class RetrievalResult(BaseModel):
    query: RetrievalQuery
    hits: list[ScoredChunk]
    trace: dict[str, Any] = Field(default_factory=dict)  # per-stage candidate ids, latencies, k values

    @property
    def chunk_ids(self) -> list[str]:
        return [h.chunk.id for h in self.hits]

    @property
    def doc_ids(self) -> list[str]:
        seen: list[str] = []
        for h in self.hits:
            if h.chunk.doc_id not in seen:
                seen.append(h.chunk.doc_id)
        return seen


@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query: RetrievalQuery) -> RetrievalResult: ...


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: RetrievalQuery, candidates: list[ScoredChunk], k: int) -> list[ScoredChunk]: ...


def visible(chunk: Chunk, principal: Principal) -> bool:
    """ACL rule used across the book: shared group membership and tenant match (or 'shared')."""
    tenant_ok = chunk.tenant in (None, "shared", principal.tenant)
    return tenant_ok and bool(set(chunk.acl_groups) & set(principal.groups))


__all__ = ["Principal", "RetrievalQuery", "ScoredChunk", "RetrievalResult", "Retriever", "Reranker", "visible"]
