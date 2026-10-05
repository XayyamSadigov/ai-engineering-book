# path: book/projects/examples/ch08/embedlab/usecases/clustering.py
"""Topic discovery: k-means on normalized embeddings, k chosen by silhouette, clusters named
by their most distinctive words so a human can review them."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from ..corpus import tokenize
from ..vector_math import l2_normalize_rows


@dataclass
class ClusterResult:
    k: int
    labels: np.ndarray
    centroids: np.ndarray
    silhouette: float

    def sizes(self) -> list[int]:
        return np.bincount(self.labels, minlength=self.k).tolist()


def cluster(vectors: np.ndarray, k: int, seed: int = 0) -> ClusterResult:
    """k-means minimizes Euclidean distance; on unit vectors that is equivalent to maximizing
    cosine, so normalize first and the clusters follow angular similarity."""
    x = l2_normalize_rows(vectors)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(x)
    sil = float(silhouette_score(x, km.labels_, metric="cosine")) if 1 < k < len(x) else 0.0
    return ClusterResult(k, km.labels_, l2_normalize_rows(km.cluster_centers_), sil)


def choose_k(vectors: np.ndarray, ks: Sequence[int], seed: int = 0) -> tuple[ClusterResult, list[tuple[int, float]]]:
    results = [cluster(vectors, k, seed) for k in ks if 1 < k < len(vectors)]
    best = max(results, key=lambda r: r.silhouette)
    return best, [(r.k, round(r.silhouette, 4)) for r in results]


def describe_clusters(texts: Sequence[str], labels: np.ndarray, top_n: int = 5) -> dict[int, list[str]]:
    """Class-based term weighting: words frequent inside the cluster and rare across clusters."""
    per_cluster: dict[int, Counter[str]] = {}
    for text, lab in zip(texts, labels):
        per_cluster.setdefault(int(lab), Counter()).update(tokenize(text))
    n_clusters = len(per_cluster)
    cluster_df = Counter(w for c in per_cluster.values() for w in c)
    out = {}
    for lab, counts in per_cluster.items():
        total = sum(counts.values()) or 1
        scored = {w: (c / total) * np.log(1 + n_clusters / cluster_df[w]) for w, c in counts.items()}
        out[lab] = [w for w, _ in sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]]
    return dict(sorted(out.items()))


def purity(labels: Sequence[int], gold: Sequence[str]) -> float:
    """Share of items whose cluster's majority gold label matches their own. Only meaningful
    when gold labels exist; real topic discovery is reviewed by people, not scored."""
    by_cluster: dict[int, Counter[str]] = {}
    for lab, g in zip(labels, gold):
        by_cluster.setdefault(int(lab), Counter())[g] += 1
    return sum(c.most_common(1)[0][1] for c in by_cluster.values()) / max(1, len(gold))


__all__ = ["ClusterResult", "choose_k", "cluster", "describe_clusters", "purity"]
