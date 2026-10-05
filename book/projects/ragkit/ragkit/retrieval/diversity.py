# path: book/projects/ragkit/ragkit/retrieval/diversity.py
"""Diversity-aware selection: maximal marginal relevance (MMR) over the reranked shortlist.

Chapter 5 explains why diversity matters for the evidence budget and Chapter 8 implements MMR
over raw embeddings. This module is the retrieval-stage version. It runs after the reranker on a
pool somewhat larger than final_k and picks final_k hits greedily:

    mmr(c) = lambda_ * relevance(c) - (1 - lambda_) * max(similarity(c, s) for s in selected)

Two details differ from the textbook formula, and both matter in a pipeline:

- relevance is the incoming score rescaled to [0, 1] over the pool, because reranker scores are
  not cosines (an RRF score of 0.03 and a cross-encoder logit of 7.2 cannot be mixed with a
  similarity in [0, 1]). Non-negative scores (RRF, lexical overlap, LLM grades) are divided by the
  pool maximum, which keeps their ratios; min-max scaling would pin the weakest pool member at 0
  and make it unselectable however different it is. Scores that can be negative (cross-encoder
  logits) are min-max scaled. A pool with equal scores falls back to rank order;
- similarity between candidates is cosine over embeddings when an EmbeddingClient is supplied, and
  token-set Jaccard otherwise. Jaccard needs no model call and catches near-copies and overlapping
  chunks, which is most of the redundancy in a RAG shortlist; embeddings also catch paraphrases.

`min_relevance` (on the normalized scale) keeps weak candidates out of the selection. Without it,
MMR fills the last slots with unrelated chunks, because an unrelated chunk is maximally "diverse".
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from aie_core.embeddings import EmbeddingClient

from .bm25 import BM25Tokenizer
from .common import indexed_text, rerank_list
from .types import RetrievalQuery, ScoredChunk


def normalized_relevance(candidates: Sequence[ScoredChunk]) -> list[float]:
    """Rescale incoming scores to [0, 1]: score / max when none is negative, else min-max."""
    if not candidates:
        return []
    scores = [c.score for c in candidates]
    lo, hi = min(scores), max(scores)
    if hi - lo > 1e-12:
        if lo >= 0.0:
            return [s / hi for s in scores]
        return [(s - lo) / (hi - lo) for s in scores]
    n = len(candidates)
    return [1.0 - i / n for i in range(n)]  # incoming order is the only signal left


def jaccard_matrix(texts: Sequence[str], tokenizer: BM25Tokenizer | None = None) -> np.ndarray:
    tok = tokenizer or BM25Tokenizer()
    sets = [set(tok.tokenize(t)) for t in texts]
    n = len(sets)
    m = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            union = len(sets[i] | sets[j])
            m[i, j] = m[j, i] = len(sets[i] & sets[j]) / union if union else 0.0
    return m


def cosine_matrix(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    v = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / np.where(norms == 0, 1.0, norms)
    return v @ v.T


def mmr_order(relevance: Sequence[float], similarity: np.ndarray, k: int, lambda_: float,
              min_relevance: float = 0.0) -> list[tuple[int, float, float]]:
    """Greedy MMR. Returns (index, mmr score, redundancy) in selection order."""
    pool = [i for i, r in enumerate(relevance) if r >= min_relevance]
    chosen: list[tuple[int, float, float]] = []
    while pool and len(chosen) < k:
        best, best_score, best_red = pool[0], float("-inf"), 0.0
        for i in pool:  # pool order = incoming rank, so ties keep the reranker's order
            red = max((float(similarity[i, j]) for j, _, _ in chosen), default=0.0)
            score = lambda_ * relevance[i] - (1.0 - lambda_) * red
            if score > best_score + 1e-12:
                best, best_score, best_red = i, score, red
        chosen.append((best, best_score, best_red))
        pool.remove(best)
    return chosen


class MMRDiversifier:
    """A `Reranker`-shaped stage: rerank(query, candidates, k) returns k diverse, relevant hits.

    Use it as `RetrievalPipeline(diversifier=...)`, which hands it a pool of `diversify_pool_k`
    reranked hits, or standalone on any ranked list.
    """

    name = "mmr"

    def __init__(self, *, lambda_: float = 0.7, min_relevance: float = 0.0,
                 embeddings: EmbeddingClient | None = None, tokenizer: BM25Tokenizer | None = None) -> None:
        if not 0.0 <= lambda_ <= 1.0:
            raise ValueError("lambda_ must be in [0, 1]")
        self.lambda_ = lambda_
        self.min_relevance = min_relevance
        self.embeddings = embeddings
        self.tokenizer = tokenizer or BM25Tokenizer()

    @property
    def similarity(self) -> str:
        return "cosine" if self.embeddings is not None else "jaccard"

    def similarity_matrix(self, candidates: Sequence[ScoredChunk]) -> np.ndarray:
        texts = [indexed_text(c.chunk) for c in candidates]
        if self.embeddings is not None:
            return cosine_matrix(self.embeddings.embed(texts))  # one batched call per request
        return jaccard_matrix(texts, self.tokenizer)

    def rerank(self, query: RetrievalQuery, candidates: list[ScoredChunk], k: int) -> list[ScoredChunk]:
        if k <= 0 or not candidates:
            return []
        rel = normalized_relevance(candidates)
        if self.lambda_ >= 1.0:  # pure relevance: no similarity work at all
            order = [(i, rel[i], 0.0) for i in range(len(candidates)) if rel[i] >= self.min_relevance][:k]
        else:
            order = mmr_order(rel, self.similarity_matrix(candidates), k, self.lambda_, self.min_relevance)
        n = len(order)
        out = [
            candidates[i].model_copy(update={
                # descending by selection order, so rerank_list keeps the MMR order
                "score": float(n - pos),
                "signals": {**candidates[i].signals, "prior_rank": float(candidates[i].rank),
                            "mmr_relevance": round(rel[i], 6), "mmr_redundancy": round(red, 6),
                            "mmr_score": round(score, 6)},
            })
            for pos, (i, score, red) in enumerate(order)
        ]
        return rerank_list(out, "diversify")


__all__ = ["MMRDiversifier", "cosine_matrix", "jaccard_matrix", "mmr_order", "normalized_relevance"]
