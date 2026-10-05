# path: book/projects/examples/ch08/embedlab/quality.py
"""Embedding quality on your own data: recall@k and MRR over labeled queries, a Matryoshka
truncation check, and a chunk-versus-document comparison. Chapter 14 owns the full retrieval
metric suite; this module is the minimum needed to choose and version an embedding model."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .pipeline import EmbeddingPipeline
from .vector_math import l2_normalize_rows, truncate


@dataclass(frozen=True)
class LabeledQuery:
    query: str
    relevant: frozenset[str]


def recall_at_k(ranked: Sequence[str], relevant: frozenset[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: frozenset[str]) -> float:
    for pos, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / pos
    return 0.0


@dataclass
class RetrievalReport:
    k: int
    recall: float
    mrr: float
    n_queries: int
    misses: list[str] = field(default_factory=list)

    def row(self, label: str) -> str:
        return f"{label:<28} recall@{self.k}={self.recall:.3f}  MRR={self.mrr:.3f}  misses={len(self.misses)}/{self.n_queries}"


def ranked_ids(
    query_matrix: np.ndarray,
    corpus_matrix: np.ndarray,
    corpus_ids: Sequence[str],
    group_of: Callable[[str], str] = lambda x: x,
    depth: int = 50,
) -> list[list[str]]:
    """Rank the corpus for every query, then collapse items to their group (for example
    section -> document) keeping first occurrence, so metrics are measured at the level
    the labels were written at."""
    sims = l2_normalize_rows(query_matrix) @ l2_normalize_rows(corpus_matrix).T
    out: list[list[str]] = []
    for row in sims:
        order = np.argsort(-row, kind="stable")[:depth]
        seen: dict[str, None] = {}
        for i in order:
            seen.setdefault(group_of(corpus_ids[i]), None)
        out.append(list(seen))
    return out


def score_rankings(rankings: list[list[str]], queries: Sequence[LabeledQuery], k: int) -> RetrievalReport:
    recalls, rrs, misses = [], [], []
    for ranked, q in zip(rankings, queries):
        r = recall_at_k(ranked, q.relevant, k)
        recalls.append(r)
        rrs.append(reciprocal_rank(ranked, q.relevant))
        if r == 0.0:
            misses.append(q.query)
    return RetrievalReport(k, float(np.mean(recalls)), float(np.mean(rrs)), len(queries), misses)


def evaluate_retrieval(
    pipeline: EmbeddingPipeline,
    queries: Sequence[LabeledQuery],
    corpus_ids: Sequence[str],
    corpus_texts: Sequence[str],
    k: int = 3,
    group_of: Callable[[str], str] = lambda x: x,
) -> RetrievalReport:
    corpus = pipeline.embed_passages(list(corpus_texts))
    qm = pipeline.embed_queries([q.query for q in queries])
    return score_rankings(ranked_ids(qm, corpus, corpus_ids, group_of), queries, k)


@dataclass(frozen=True)
class TruncationRow:
    dims: int
    recall: float
    mrr: float
    neighbor_overlap: float  # share of full-dimension top-k neighbours still in the truncated top-k


def truncation_report(
    query_matrix: np.ndarray,
    corpus_matrix: np.ndarray,
    corpus_ids: Sequence[str],
    queries: Sequence[LabeledQuery],
    dims_list: Sequence[int],
    k: int = 3,
    group_of: Callable[[str], str] = lambda x: x,
) -> list[TruncationRow]:
    """Does a prefix of the vector keep the ranking? A model trained Matryoshka-style is built so
    that it does; any other model must be measured, never assumed."""
    full = ranked_ids(query_matrix, corpus_matrix, corpus_ids, group_of)
    rows = []
    for d in dims_list:
        ranked = ranked_ids(truncate(query_matrix, d), truncate(corpus_matrix, d), corpus_ids, group_of)
        rep = score_rankings(ranked, queries, k)
        overlap = float(np.mean([len(set(a[:k]) & set(b[:k])) / k for a, b in zip(full, ranked)]))
        rows.append(TruncationRow(d, rep.recall, rep.mrr, overlap))
    return rows


def nearest_neighbors(
    matrix: np.ndarray, ids: Sequence[str], probe_ids: Sequence[str], k: int = 3
) -> dict[str, list[tuple[str, float]]]:
    """Eyeball check: for a few known items, list their nearest neighbours. Run it on every
    candidate model before computing any metric; nonsense neighbours end the evaluation early."""
    m = l2_normalize_rows(matrix)
    pos = {i: n for n, i in enumerate(ids)}
    out = {}
    for pid in probe_ids:
        sims = m @ m[pos[pid]]
        order = [j for j in np.argsort(-sims, kind="stable") if j != pos[pid]][:k]
        out[pid] = [(ids[j], round(float(sims[j]), 3)) for j in order]
    return out


__all__ = [
    "LabeledQuery",
    "RetrievalReport",
    "TruncationRow",
    "evaluate_retrieval",
    "nearest_neighbors",
    "ranked_ids",
    "recall_at_k",
    "reciprocal_rank",
    "score_rankings",
    "truncation_report",
]
