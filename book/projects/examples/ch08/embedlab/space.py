# path: book/projects/examples/ch08/embedlab/space.py
"""Embedding spaces, a versioned in-memory index, and re-embedding plans.

A vector is meaningless without the space it was produced in. The space is everything that
changes the geometry: model, dimensions, text preparation, instruction prefixes, normalization,
and any post-processing such as mean-centering. Two vectors may be compared only if their
spaces are equal, so the index refuses mixed writes and mismatched queries.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
from pydantic import BaseModel, ConfigDict

from aie_core.llm.tokens import count_tokens

from .vector_math import as_matrix, l2_normalize_rows


class EmbeddingSpace(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    dimensions: int
    text_prep: str = "v1"
    query_prefix: str = ""
    passage_prefix: str = ""
    normalized: bool = True
    post_process: str = "none"  # e.g. "mean-center:<hash>" or "truncate:256"

    @property
    def fingerprint(self) -> str:
        blob = json.dumps(self.model_dump(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]


class SpaceMismatchError(ValueError):
    """Raised when vectors from one space meet vectors from another."""


@dataclass(frozen=True)
class Hit:
    id: str
    score: float
    metadata: Mapping[str, Any]


class VectorIndex:
    """Exact cosine search over one embedding space. Chapter 9 replaces the matrix with ANN."""

    def __init__(self, space: EmbeddingSpace) -> None:
        self.space = space
        self.ids: list[str] = []
        self.metadata: list[dict[str, Any]] = []
        self._matrix = np.zeros((0, space.dimensions), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    def _check(self, space: EmbeddingSpace) -> None:
        if space != self.space:
            raise SpaceMismatchError(
                f"index space {self.space.fingerprint} ({self.space.model}) != "
                f"incoming space {space.fingerprint} ({space.model}); re-embed instead of mixing"
            )

    def add(self, ids: list[str], vectors: Any, space: EmbeddingSpace, metadata: list[dict[str, Any]] | None = None) -> None:
        self._check(space)
        m = as_matrix(vectors)
        if m.shape[0] != len(ids):
            raise ValueError("ids and vectors differ in length")
        if m.shape[1] != self.space.dimensions:
            raise SpaceMismatchError(f"expected {self.space.dimensions} dims, got {m.shape[1]}")
        if self.space.normalized:
            m = l2_normalize_rows(m)
        # float32 halves memory versus float64 with no measurable ranking change for retrieval
        self._matrix = np.vstack([self._matrix, m.astype(np.float32)])
        self.ids.extend(ids)
        self.metadata.extend(metadata or [{} for _ in ids])

    def delete(self, ids: list[str]) -> int:
        """Remove items (document deletion, right-to-erasure). Returns how many were removed.
        Deleting the source text without deleting its vectors and cache entries is a leak."""
        drop = set(ids)
        keep = [n for n, i in enumerate(self.ids) if i not in drop]
        removed = len(self.ids) - len(keep)
        self._matrix = self._matrix[keep]
        self.ids = [self.ids[n] for n in keep]
        self.metadata = [self.metadata[n] for n in keep]
        return removed

    def search(
        self,
        query_vector: Any,
        space: EmbeddingSpace,
        k: int = 5,
        where: Callable[[Mapping[str, Any]], bool] | None = None,
    ) -> list[Hit]:
        self._check(space)
        if not self.ids:
            return []
        q = l2_normalize_rows(query_vector)[0].astype(np.float32)
        scores = self._matrix @ q
        if not self.space.normalized:
            norms = np.linalg.norm(self._matrix, axis=1)
            scores = np.where(norms > 0, scores / np.where(norms > 0, norms, 1.0), 0.0)
        order = np.argsort(-scores, kind="stable")
        hits: list[Hit] = []
        for i in order:
            if where is not None and not where(self.metadata[i]):
                continue
            hits.append(Hit(self.ids[i], float(scores[i]), self.metadata[i]))
            if len(hits) == k:
                break
        return hits


class ReembedPlan(BaseModel):
    old_space: str
    new_space: str
    items: int
    estimated_tokens: int
    estimated_cost_usd: float
    estimated_minutes: float | None = None  # wall-clock at the rate limit; usually the real constraint
    reason: str


def plan_reembed(
    index: VectorIndex,
    new_space: EmbeddingSpace,
    texts_by_id: Mapping[str, str],
    price_per_million_tokens: float,
    tokens_per_minute: float | None = None,
) -> ReembedPlan | None:
    """None when the spaces match. Otherwise every stored item must be re-embedded: there is no
    safe way to translate vectors between two models, so the corpus text is the source of truth.
    Pass the provider's tokens-per-minute limit to get the migration window, not just the bill."""
    if new_space == index.space:
        return None
    changed = [k for k, v in new_space.model_dump().items() if index.space.model_dump()[k] != v]
    tokens = sum(count_tokens(texts_by_id[i]) for i in index.ids)
    return ReembedPlan(
        old_space=index.space.fingerprint,
        new_space=new_space.fingerprint,
        items=len(index),
        estimated_tokens=tokens,
        estimated_cost_usd=round(tokens / 1_000_000 * price_per_million_tokens, 6),
        estimated_minutes=round(tokens / tokens_per_minute, 2) if tokens_per_minute else None,
        reason="changed: " + ", ".join(changed),
    )


__all__ = ["EmbeddingSpace", "Hit", "ReembedPlan", "SpaceMismatchError", "VectorIndex", "plan_reembed"]
