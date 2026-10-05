# path: book/projects/ragkit/ragkit/retrieval/mmr.py
"""Embedding-space MMR over ScoredChunk lists: the Chapter 8 formula, applied to retrieval hits.

`mmr_select` is the function form for callers that already hold vectors: relevance is cosine to the
query vector, redundancy is cosine between candidates, exactly as in Chapter 8's
`embedlab.usecases.recommend.mmr`. Vectors come from the caller (for example the ones the dense
retriever embedded) or are computed in one batch with an `EmbeddingClient`.

Inside `RetrievalPipeline`, prefer `diversity.MMRDiversifier` as the `diversifier` stage: after a
reranker, the reranker's score is a better relevance signal than raw cosine, so that class rescales
reranker scores instead. Both share `mmr_order`.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from aie_core.embeddings import EmbeddingClient

from .common import indexed_text, rerank_list
from .diversity import cosine_matrix, mmr_order
from .types import ScoredChunk


def mmr_select(
    query_vec: Sequence[float],
    candidates: Sequence[ScoredChunk],
    k: int,
    lambda_: float = 0.7,
    min_relevance: float | None = None,
    *,
    vectors: Sequence[Sequence[float]] | None = None,
    embeddings: EmbeddingClient | None = None,
) -> list[ScoredChunk]:
    """Pick k hits maximizing lambda_*cos(query, c) - (1-lambda_)*max cos(c, selected).

    `min_relevance` is a cosine floor: candidates below it are never selected, because an
    unrelated chunk is maximally "diverse". Returned hits are re-ranked 1..k with stage "diversify"
    and carry `prior_rank`, `mmr_relevance`, `mmr_redundancy`, and `mmr_score` signals.
    """
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda_ must be in [0, 1]")
    if k <= 0 or not candidates:
        return []
    if vectors is None:
        if embeddings is None:
            raise ValueError("mmr_select needs candidate vectors or an EmbeddingClient")
        vectors = embeddings.embed([indexed_text(c.chunk) for c in candidates])
    if len(vectors) != len(candidates):
        raise ValueError("one vector per candidate is required")
    sim = cosine_matrix(list(vectors))
    q = np.asarray(query_vec, dtype=np.float64)
    v = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(v, axis=1) * (np.linalg.norm(q) or 1.0)
    rel = [float(x) for x in (v @ q) / np.where(norms == 0, 1.0, norms)]
    order = mmr_order(rel, sim, k, lambda_, min_relevance if min_relevance is not None else float("-inf"))
    n = len(order)
    out = [
        candidates[i].model_copy(update={
            "score": float(n - pos),
            "signals": {**candidates[i].signals, "prior_rank": float(candidates[i].rank),
                        "mmr_relevance": round(rel[i], 6), "mmr_redundancy": round(red, 6),
                        "mmr_score": round(score, 6)},
        })
        for pos, (i, score, red) in enumerate(order)
    ]
    return rerank_list(out, "diversify")


__all__ = ["mmr_select"]
