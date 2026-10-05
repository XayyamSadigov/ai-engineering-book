# path: book/projects/p3-rag-assistant/rag_assistant/domain/models.py
"""Pure data types shared by ingestion, retrieval, answering and the API. No I/O here."""
from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field
from ragkit.generation import AnswerEnvelope, ResolvedCitation
from ragkit.retrieval import Principal

DocStatus = Literal["active", "deleting", "deleted"]
ChangeKind = Literal["new", "content", "acl_only", "metadata", "reindex", "unchanged", "skipped_tombstone", "skipped_stale", "purged"]
AskMode = Literal["answer", "sources_only", "blocked", "unavailable"]

SHARED_TENANT = "shared"


def now() -> float:
    return time.time()


class DocRecord(BaseModel):
    """The registry row for one logical document: identity, lineage, and where it is indexed.

    `fingerprint` covers everything that changes what the indexes hold (content, ACL, tenant,
    authority metadata, chunker, pipeline version). Two ingestions with equal fingerprints are
    the same ingestion, which is what makes the worker idempotent.
    """

    doc_id: str
    tenant: str
    acl_groups: list[str]
    version: str
    title: str = ""
    content_hash: str
    fingerprint: str
    connector: str
    uri: str
    chunk_ids: list[str] = Field(default_factory=list)
    context_keys: dict[str, str] = Field(default_factory=dict)  # chunk id -> ContextualEnricher key
    index_versions: list[str] = Field(default_factory=list)
    status: DocStatus = "active"
    submitted_seq: int = 0
    deleted_seq: int | None = None
    submitted_at: float = 0.0  # when the change was observed (start of the freshness clock)
    indexed_at: float | None = None  # when it became searchable
    deleted_at: float | None = None
    purged_at: float | None = None
    authority: int = 1
    effective_date: str | None = None
    supersedes: list[str] = Field(default_factory=list)

    @property
    def freshness_lag_s(self) -> float | None:
        if self.indexed_at is None:
            return None
        return max(0.0, self.indexed_at - self.submitted_at)


class IngestPayload(BaseModel):
    """What travels on the queue: a pointer to the source, never the authority to skip checks."""

    op: Literal["upsert", "purge"]
    doc_id: str
    connector: str = "folder"
    uri: str = ""
    content_hash: str = ""
    seq: int
    submitted_at: float
    target_versions: list[str] = Field(default_factory=list)  # empty: every writable version
    reason: str = "change"


class IngestOutcome(BaseModel):
    doc_id: str
    change: ChangeKind
    added: int = 0
    unchanged: int = 0
    removed: int = 0
    reembedded: int = 0
    versions: list[str] = Field(default_factory=list)
    freshness_lag_s: float | None = None


class SourceSnippet(BaseModel):
    """A retrieved source shown without a generated answer (degraded mode) or alongside one."""

    doc_id: str
    title: str
    version: str
    section: str = ""
    source_uri: str | None = None
    snippet: str


class AskResponse(BaseModel):
    """The API response for /v1/ask (non-streaming)."""

    request_id: str
    mode: AskMode
    answer: AnswerEnvelope | None = None
    sources: list[SourceSnippet] = Field(default_factory=list)
    citations: list[ResolvedCitation] = Field(default_factory=list)
    degraded: list[str] = Field(default_factory=list)
    cache: Literal["hit", "miss", "bypass"] = "miss"
    index_version: str = ""
    message: str | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)


class IndexStatus(BaseModel):
    active_version: str
    building_version: str | None = None
    previous_version: str | None = None
    tenancy_mode: str
    documents_active: int
    documents_deleting: int
    chunks_indexed: dict[str, int]
    generations: dict[str, int]
    queue_depth: int
    dead_letters: int
    freshness_lag_p95_s: float | None = None
    freshness_lag_max_s: float | None = None
    freshness_slo_s: float
    freshness_slo_met: bool
    oldest_pending_purge_s: float | None = None
    breakers: dict[str, str] = Field(default_factory=dict)


class RequestScope(BaseModel):
    """Everything about a request that changes what it may see. Cache keys are built from it."""

    principal: Principal
    index_version: str
    generations: dict[str, int]

    def key_parts(self) -> tuple[Any, ...]:
        return (self.index_version, tuple(sorted(self.generations.items())))


__all__ = [
    "AskMode", "AskResponse", "ChangeKind", "DocRecord", "DocStatus", "IndexStatus", "IngestOutcome",
    "IngestPayload", "RequestScope", "SHARED_TENANT", "SourceSnippet", "now",
]
