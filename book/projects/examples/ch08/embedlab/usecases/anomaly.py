# path: book/projects/examples/ch08/embedlab/usecases/anomaly.py
"""Anomaly detection by distance to the nearest class centroid, and a batch drift score.

The threshold is a quantile of distances seen on known-normal data, so "anomalous" means
"farther from every known topic than 95% (say) of normal traffic was"."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..vector_math import centroid, l2_normalize_rows


@dataclass(frozen=True)
class AnomalyScore:
    distance: float  # 1 - cosine to the nearest centroid
    nearest: str
    is_anomaly: bool


class CentroidAnomalyDetector:
    def __init__(self, quantile: float = 0.95) -> None:
        self.quantile = quantile
        self.names: list[str] = []
        self.centroids = np.zeros((0, 0))
        self.threshold = 1.0

    def fit(self, vectors: np.ndarray, labels: Sequence[str] | None = None) -> "CentroidAnomalyDetector":
        """With labels, one centroid per class (normal traffic is multi-modal: a single global
        centroid sits between topics and flags nothing useful). Without labels, one centroid."""
        x = l2_normalize_rows(vectors)
        if labels is None:
            self.names, self.centroids = ["all"], centroid(x)[None, :]
        else:
            lab = np.asarray(labels)
            self.names = sorted(set(labels))
            self.centroids = np.vstack([centroid(x[lab == c]) for c in self.names])
        self.threshold = float(np.quantile(self._distances(x), self.quantile))
        return self

    def calibrate(self, held_out_normal: np.ndarray) -> "CentroidAnomalyDetector":
        """Re-set the threshold on known-normal items the centroids were NOT fit on. Distances
        on the fitting data are optimistic (each item pulled its own centroid closer), so an
        in-sample threshold flags more than 1 - quantile of new normal traffic."""
        self.threshold = float(np.quantile(self._distances(l2_normalize_rows(held_out_normal)), self.quantile))
        return self

    def flag_rate(self, vectors: np.ndarray) -> float:
        """Share of a batch flagged. On normal traffic it should sit near 1 - quantile; a sustained
        rise is a drift signal that, unlike drift_score, also catches new topics hidden in a mix."""
        return float(np.mean([s.is_anomaly for s in self.score(vectors)])) if len(vectors) else 0.0

    def _distances(self, x: np.ndarray) -> np.ndarray:
        return 1.0 - (x @ self.centroids.T).max(axis=1)

    def score(self, vectors: np.ndarray) -> list[AnomalyScore]:
        x = l2_normalize_rows(vectors)
        sims = x @ self.centroids.T
        nearest = sims.argmax(axis=1)
        dist = 1.0 - sims.max(axis=1)
        return [AnomalyScore(float(d), self.names[n], bool(d > self.threshold)) for d, n in zip(dist, nearest)]


def drift_score(reference: np.ndarray, current: np.ndarray) -> float:
    """1 - cosine between batch centroids. Zero means the batch points the same way as the
    reference; track it per day or per tenant and alert on a sustained rise. It only sees a
    shift of the mean: a new topic that is 5% of traffic barely moves it, so pair it with
    CentroidAnomalyDetector.flag_rate."""
    return float(1.0 - centroid(reference) @ centroid(current))


__all__ = ["AnomalyScore", "CentroidAnomalyDetector", "drift_score"]
