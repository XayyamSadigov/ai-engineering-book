# path: book/projects/examples/ch08/embedlab/usecases/recommend.py
"""Related items ("tickets like this one", "see also" for a document) with maximal marginal
relevance, so the list is not five copies of the nearest neighbour."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..vector_math import centroid, l2_normalize_rows


def mmr(
    query: np.ndarray,
    candidates: np.ndarray,
    k: int,
    lambda_: float = 0.7,
    exclude: Sequence[int] = (),
    min_relevance: float = 0.0,
) -> list[int]:
    """Greedy MMR: pick the candidate maximizing lambda*sim(query) - (1-lambda)*max sim(selected).

    `min_relevance` drops weak candidates before diversifying. Without it, MMR happily fills the
    list with unrelated items, because an unrelated item is maximally "diverse"."""
    c = l2_normalize_rows(candidates)
    q = l2_normalize_rows(query)[0]
    rel = c @ q
    skip = set(exclude)
    pool = [i for i in range(len(c)) if i not in skip and rel[i] >= min_relevance]
    chosen: list[int] = []
    while pool and len(chosen) < k:
        if chosen:
            redundancy = (c[pool] @ c[chosen].T).max(axis=1)
        else:
            redundancy = np.zeros(len(pool))
        scores = lambda_ * rel[pool] - (1 - lambda_) * redundancy
        best = pool[int(np.argmax(scores))]
        chosen.append(best)
        pool.remove(best)
    return chosen


def related_items(matrix: np.ndarray, item: int, k: int = 5, lambda_: float = 0.7, min_relevance: float = 0.0) -> list[int]:
    return mmr(matrix[item], matrix, k, lambda_, exclude=[item], min_relevance=min_relevance)


def recommend_for_history(
    matrix: np.ndarray, history: Sequence[int], k: int = 5, lambda_: float = 0.7, min_relevance: float = 0.0
) -> list[int]:
    """Profile = centroid of what the user engaged with; exclude what they already saw."""
    profile = centroid(matrix[list(history)])
    return mmr(profile, matrix, k, lambda_, exclude=history, min_relevance=min_relevance)


__all__ = ["mmr", "recommend_for_history", "related_items"]
