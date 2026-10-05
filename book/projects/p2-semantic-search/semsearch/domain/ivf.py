# path: book/projects/p2-semantic-search/semsearch/domain/ivf.py
"""A teaching-size IVF (inverted file) index on NumPy, to measure ANN recall against exact.

IVF partitions vectors into `n_lists` clusters with k-means. A query is compared with the
centroids, the `nprobe` closest clusters are scanned exhaustively, and the rest are skipped.
Recall and latency both grow with nprobe; nprobe == n_lists is exact search with overhead.
Production engines add compression (product quantization) and SIMD; the trade-off curve is
the same.
"""
from __future__ import annotations

import numpy as np


def _unit_rows(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


def exact_top_k(matrix: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    scores = matrix @ query
    k = min(k, scores.shape[0])
    top = np.argpartition(-scores, k - 1)[:k]
    return top[np.argsort(-scores[top], kind="stable")]


class IVFIndex:
    def __init__(self, n_lists: int, *, iterations: int = 10, seed: int = 0) -> None:
        if n_lists <= 0:
            raise ValueError("n_lists must be positive")
        self.n_lists = n_lists
        self.iterations = iterations
        self.seed = seed
        self.centroids: np.ndarray | None = None
        self.lists: list[np.ndarray] = []
        self.vectors: np.ndarray | None = None

    def fit(self, vectors: np.ndarray) -> "IVFIndex":
        x = _unit_rows(np.asarray(vectors, dtype=np.float32))
        n = x.shape[0]
        if n < self.n_lists:
            raise ValueError("need at least n_lists vectors to train")
        rng = np.random.default_rng(self.seed)
        centroids = x[rng.choice(n, size=self.n_lists, replace=False)].copy()
        assign = np.zeros(n, dtype=np.int64)
        for _ in range(self.iterations):  # spherical k-means: assign by max cosine
            assign = np.argmax(x @ centroids.T, axis=1)
            for c in range(self.n_lists):
                members = x[assign == c]
                if len(members):
                    centroids[c] = members.mean(axis=0)
                else:  # re-seed an empty cluster so no list is permanently dead
                    centroids[c] = x[rng.integers(n)]
            centroids = _unit_rows(centroids)
        assign = np.argmax(x @ centroids.T, axis=1)
        self.centroids = centroids
        self.vectors = x
        self.lists = [np.flatnonzero(assign == c) for c in range(self.n_lists)]
        return self

    def search(self, query: np.ndarray, k: int, nprobe: int) -> tuple[np.ndarray, int]:
        """Returns (row ids of the approximate top-k, number of vectors scored)."""
        if self.centroids is None or self.vectors is None:
            raise RuntimeError("index is not trained")
        q = np.asarray(query, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        nprobe = max(1, min(nprobe, self.n_lists))
        probe = np.argsort(-(self.centroids @ q))[:nprobe]
        candidates = np.concatenate([self.lists[c] for c in probe])
        if candidates.size == 0:
            return np.array([], dtype=np.int64), 0
        local = exact_top_k(self.vectors[candidates], q, k)
        return candidates[local], int(candidates.size)

    def list_sizes(self) -> list[int]:
        return [int(len(ix)) for ix in self.lists]


__all__ = ["IVFIndex", "exact_top_k"]
