# path: book/projects/p2-semantic-search/semsearch/domain/metrics.py
"""Retrieval metrics at the document level. Chapter 14 owns the full metric catalogue."""
from __future__ import annotations

from typing import Iterable, Sequence


def unique_in_order(items: Iterable[str]) -> list[str]:
    """Collapse chunk-level results to documents, keeping first-seen rank."""
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        raise ValueError("recall is undefined without relevant items")
    return len(rel & set(ranked[:k])) / len(rel)


def precision_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    rel = set(relevant)
    return sum(1 for r in ranked[:k] if r in rel) / k


def reciprocal_rank(ranked: Sequence[str], relevant: Iterable[str]) -> float:
    rel = set(relevant)
    for i, r in enumerate(ranked, start=1):
        if r in rel:
            return 1.0 / i
    return 0.0


def overlap_at_k(approx: Sequence[str], exact: Sequence[str], k: int) -> float:
    """ANN recall: the fraction of the exact top-k that the approximate search also returned."""
    truth = list(exact[:k])
    if not truth:
        return 1.0
    return len(set(approx[:k]) & set(truth)) / len(truth)


__all__ = ["unique_in_order", "recall_at_k", "precision_at_k", "reciprocal_rank", "overlap_at_k"]
