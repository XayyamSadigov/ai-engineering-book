# path: book/projects/ragkit/ragkit/retrieval/common.py
"""Helpers every retrieval stage shares: metadata filters, the text that gets indexed,
re-ranking ScoredChunk lists, and a stopwatch for stage latencies.

Filters are part of authorization's neighborhood, so they fail loudly: an unknown filter key
raises instead of being ignored, because a typo that silently drops a filter widens results.
"""
from __future__ import annotations

import re
import time
from typing import Any, Iterable, Mapping

from ..documents import Chunk
from .types import ScoredChunk

# Supported keys of RetrievalQuery.filters. All are document-level properties copied onto chunks.
FILTER_KEYS = frozenset({"tags_any", "doc_ids", "exclude_doc_ids", "updated_after", "source_types"})


class UnknownFilterError(ValueError):
    """A filter key that no retriever understands. Raised, never ignored."""


def validate_filters(filters: Mapping[str, Any]) -> None:
    unknown = set(filters) - FILTER_KEYS
    if unknown:
        raise UnknownFilterError(f"unknown filter keys {sorted(unknown)}; supported: {sorted(FILTER_KEYS)}")


def matches_filters(chunk: Chunk, filters: Mapping[str, Any]) -> bool:
    """Conjunction of constraints; an absent key means no constraint on that field."""
    if not filters:
        return True
    validate_filters(filters)
    meta = chunk.metadata
    if "tags_any" in filters:
        tags = set(meta.get("tags") or [])
        if not tags & set(filters["tags_any"]):
            return False
    if "doc_ids" in filters and chunk.doc_id not in set(filters["doc_ids"]):
        return False
    if "exclude_doc_ids" in filters and chunk.doc_id in set(filters["exclude_doc_ids"]):
        return False
    if "updated_after" in filters:
        updated = str(meta.get("updated_at") or "")
        # ISO-8601 dates compare correctly as strings; a chunk without a date fails closed
        if not updated or updated <= str(filters["updated_after"]):
            return False
    if "source_types" in filters and meta.get("source_type") not in set(filters["source_types"]):
        return False
    return True


def indexed_text(chunk: Chunk) -> str:
    """What lexical and dense indexes see: optional contextual prefix, breadcrumb, body.

    `Chunk.text` stays the exact source for citations; the prefix written by contextual
    retrieval (ragkit.retrieval.contextual) lives in metadata and only affects search.
    """
    prefix = str(chunk.metadata.get("context_prefix") or "").strip()
    body = chunk.embedding_text()
    return f"{prefix}\n\n{body}" if prefix else body


_IDENT_RE = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+")


def extract_identifiers(text: str) -> list[str]:
    """Compound tokens that mix letters and digits ("INC-2025-1142", "RET-002"), deduplicated, in order."""
    seen: dict[str, None] = {}
    for m in _IDENT_RE.finditer(text):
        tok = m.group(0)
        if any(ch.isdigit() for ch in tok) and any(ch.isalpha() for ch in tok):
            seen.setdefault(tok, None)
    return list(seen)


def rerank_list(items: Iterable[ScoredChunk], stage: str) -> list[ScoredChunk]:
    """Sort by score (desc), break ties deterministically by chunk id, assign 1-based ranks."""
    ordered = sorted(items, key=lambda s: (-s.score, s.chunk.id))
    return [s.model_copy(update={"rank": i, "stage": stage}) for i, s in enumerate(ordered, start=1)]


class Stopwatch:
    """`with Stopwatch() as sw: ...; sw.ms` - wall-clock milliseconds for a stage."""

    def __enter__(self) -> "Stopwatch":
        self._start = time.perf_counter()
        self.ms = 0.0
        return self

    def __exit__(self, *exc: object) -> None:
        self.ms = round((time.perf_counter() - self._start) * 1000.0, 3)


__all__ = [
    "FILTER_KEYS",
    "UnknownFilterError",
    "validate_filters",
    "matches_filters",
    "indexed_text",
    "extract_identifiers",
    "rerank_list",
    "Stopwatch",
]
