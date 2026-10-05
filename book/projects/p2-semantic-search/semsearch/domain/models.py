# path: book/projects/p2-semantic-search/semsearch/domain/models.py
"""The records a vector store holds and the shapes of a query and its results.

A vector is never stored alone: every row carries what authorization (tenant, acl_groups),
citation (doc_id, doc_version, ordinal), filtering (tags, metadata), and reindexing
(embedding_model) need. If a field is missing here, no later stage can recover it.
"""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

Vector = list[float]

_NAMESPACE_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")


def make_namespace(index_name: str, embedding_model: str, index_version: str) -> str:
    """The embedding model is part of the index identity: vectors from two models never mix."""
    model = re.sub(r"[^a-z0-9_.-]+", "-", embedding_model.lower()).strip("-")
    return f"{index_name}:{model}:{index_version}"


def chunk_id(doc_id: str, doc_version: str, ordinal: int) -> str:
    """Deterministic, so re-ingesting the same content upserts instead of duplicating."""
    return f"{doc_id}@{doc_version}#{ordinal:04d}"


class VectorRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    namespace: str
    tenant: str
    doc_id: str
    doc_version: str
    ordinal: int = Field(ge=0)
    text: str
    vector: Vector
    acl_groups: tuple[str, ...]
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding_model: str

    @field_validator("namespace")
    @classmethod
    def _valid_namespace(cls, v: str) -> str:
        if not _NAMESPACE_RE.match(v):
            raise ValueError(f"invalid namespace {v!r}")
        return v

    @field_validator("acl_groups")
    @classmethod
    def _acl_not_empty(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        # A record nobody may read is a bug; a record with no ACL must not mean "everyone".
        if not v:
            raise ValueError("acl_groups must not be empty; use ('all',) for public documents")
        return v


class SearchFilter(BaseModel):
    """Conjunction of constraints. None means 'no constraint on this field'.

    tenants: record.tenant must be one of these.
    acl_groups: record.acl_groups must intersect these (the caller's groups).
    tags_any: record.tags must intersect these.
    doc_ids: record.doc_id must be one of these.
    """

    model_config = ConfigDict(frozen=True)

    tenants: tuple[str, ...] | None = None
    acl_groups: tuple[str, ...] | None = None
    tags_any: tuple[str, ...] | None = None
    doc_ids: tuple[str, ...] | None = None

    def is_empty(self) -> bool:
        return all(v is None for v in (self.tenants, self.acl_groups, self.tags_any, self.doc_ids))


class SearchHit(BaseModel):
    id: str
    doc_id: str
    doc_version: str
    ordinal: int
    tenant: str
    score: float  # cosine similarity in [-1, 1]; higher is closer
    text: str
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocVersion(BaseModel):
    """What the store believes is indexed for one document; drives incremental ingestion."""

    doc_id: str
    doc_version: str
    chunks: int


__all__ = [
    "Vector",
    "VectorRecord",
    "SearchFilter",
    "SearchHit",
    "DocVersion",
    "make_namespace",
    "chunk_id",
]
