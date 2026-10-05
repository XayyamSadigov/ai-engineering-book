# path: book/projects/examples/ch08/embedlab/vector_math.py
"""VectorMath: the handful of operations every embedding feature in Chapter 8 is built from.

Everything works on 2-D NumPy matrices of shape (n, d). Single-vector helpers from
aie_core (cosine_similarity, normalize, top_k) are re-exported so callers have one import site.
"""
from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

from aie_core.embeddings import cosine_similarity, normalize, top_k

Metric = Literal["cosine", "dot", "euclidean"]
ArrayLike = Sequence[Sequence[float]] | np.ndarray


def as_matrix(vectors: ArrayLike) -> np.ndarray:
    m = np.asarray(vectors, dtype=np.float64)
    if m.ndim == 1:
        m = m.reshape(1, -1)
    if m.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {m.shape}")
    return m


def l2_normalize_rows(vectors: ArrayLike) -> np.ndarray:
    """Divide each row by its L2 norm. Zero rows stay zero instead of becoming NaN."""
    m = as_matrix(vectors)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)
    return m / safe


def is_normalized(vectors: ArrayLike, tol: float = 1e-3) -> bool:
    norms = np.linalg.norm(as_matrix(vectors), axis=1)
    nonzero = norms[norms > 0]
    return bool(nonzero.size == 0 or np.all(np.abs(nonzero - 1.0) <= tol))


def pairwise(a: ArrayLike, b: ArrayLike, metric: Metric = "cosine") -> np.ndarray:
    """Score every row of `a` against every row of `b`. Higher is always more similar,
    so Euclidean is returned as negative distance."""
    x, y = as_matrix(a), as_matrix(b)
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"dimension mismatch: {x.shape[1]} vs {y.shape[1]}")
    if metric == "cosine":
        return l2_normalize_rows(x) @ l2_normalize_rows(y).T
    if metric == "dot":
        return x @ y.T
    if metric == "euclidean":
        sq = (x**2).sum(1)[:, None] + (y**2).sum(1)[None, :] - 2.0 * (x @ y.T)
        return -np.sqrt(np.maximum(sq, 0.0))
    raise ValueError(f"unknown metric {metric!r}")


def similarity(a: Sequence[float], b: Sequence[float], metric: Metric = "cosine") -> float:
    return float(pairwise([a], [b], metric)[0, 0])


def rank(query: Sequence[float], matrix: ArrayLike, metric: Metric = "cosine", k: int = 10) -> list[tuple[int, float]]:
    """Exact (brute-force) nearest neighbours under any metric, best first."""
    scores = pairwise([query], matrix, metric)[0]
    k = max(0, min(k, scores.shape[0]))
    order = np.argsort(-scores, kind="stable")[:k]
    return [(int(i), float(scores[i])) for i in order]


def euclidean_from_cosine(cos: float) -> float:
    """For unit vectors ||a - b||^2 = 2 - 2 cos(a, b), so the two metrics rank identically."""
    return float(np.sqrt(max(0.0, 2.0 - 2.0 * cos)))


def truncate(vectors: ArrayLike, dims: int, renormalize: bool = True) -> np.ndarray:
    """Keep the first `dims` coordinates (Matryoshka-style). Re-normalize, because a prefix
    of a unit vector is shorter than 1 and dot-product scores would otherwise shrink."""
    m = as_matrix(vectors)
    if not 0 < dims <= m.shape[1]:
        raise ValueError(f"dims must be in 1..{m.shape[1]}, got {dims}")
    cut = m[:, :dims]
    return l2_normalize_rows(cut) if renormalize else cut


def centroid(vectors: ArrayLike, renormalize: bool = True) -> np.ndarray:
    """Mean direction of a set of vectors. Averages unit vectors first so long texts do not dominate."""
    c = l2_normalize_rows(vectors).mean(axis=0)
    if renormalize:
        n = np.linalg.norm(c)
        return c / n if n > 0 else c
    return c


def similarity_profile(vectors: ArrayLike) -> dict[str, float]:
    """Distribution of off-diagonal cosine similarities in a sample.

    A model whose unrelated texts already score 0.7 against each other (anisotropy) needs very
    different thresholds from one where they score 0.1. Measure before choosing any threshold.
    """
    m = l2_normalize_rows(vectors)
    n = m.shape[0]
    if n < 2:
        return {"mean": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0}
    sims = m @ m.T
    off = sims[np.triu_indices(n, k=1)]
    return {
        "mean": float(off.mean()),
        "p05": float(np.percentile(off, 5)),
        "p50": float(np.percentile(off, 50)),
        "p95": float(np.percentile(off, 95)),
    }


def mean_center(vectors: ArrayLike, mean: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Subtract the corpus mean and re-normalize. Spreads out an anisotropic space; the same
    `mean` must then be applied to every query, so it becomes part of the embedding version."""
    m = as_matrix(vectors)
    mu = m.mean(axis=0) if mean is None else mean
    return l2_normalize_rows(m - mu), mu


__all__ = [
    "Metric",
    "as_matrix",
    "centroid",
    "cosine_similarity",
    "euclidean_from_cosine",
    "is_normalized",
    "l2_normalize_rows",
    "mean_center",
    "normalize",
    "pairwise",
    "rank",
    "similarity",
    "similarity_profile",
    "top_k",
    "truncate",
]
