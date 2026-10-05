# path: book/projects/examples/ch08/embedlab/usecases/dedup.py
"""Semantic deduplication with a threshold chosen from labeled pairs, not guessed."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..pipeline import EmbeddingPipeline
from ..vector_math import l2_normalize_rows


@dataclass(frozen=True)
class LabeledPair:
    a: str
    b: str
    is_duplicate: bool


@dataclass(frozen=True)
class ThresholdPoint:
    threshold: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def pair_scores(pipeline: EmbeddingPipeline, pairs: Sequence[LabeledPair]) -> np.ndarray:
    a = l2_normalize_rows(pipeline.embed_passages([p.a for p in pairs]))
    b = l2_normalize_rows(pipeline.embed_passages([p.b for p in pairs]))
    return (a * b).sum(axis=1)


def sweep_thresholds(scores: np.ndarray, labels: Sequence[bool], thresholds: Sequence[float] | None = None) -> list[ThresholdPoint]:
    y = np.asarray(labels, dtype=bool)
    if thresholds is None:
        thresholds = sorted(set(np.round(scores, 4).tolist()))  # every distinct score is a candidate
    points = []
    for t in thresholds:
        pred = scores >= t
        tp = int((pred & y).sum())
        fp = int((pred & ~y).sum())
        fn = int((~pred & y).sum())
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        points.append(ThresholdPoint(float(t), precision, recall, f1, tp, fp, fn))
    return points


def select_threshold(points: Sequence[ThresholdPoint], min_precision: float = 0.95) -> ThresholdPoint | None:
    """Highest recall among thresholds that meet the precision floor; ties go to the higher
    (safer) threshold. None means no threshold is safe and dedup must not be automatic."""
    ok = [p for p in points if p.precision >= min_precision and p.tp > 0]
    if not ok:
        return None
    return max(ok, key=lambda p: (p.recall, p.threshold))


def find_duplicates(matrix: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    """All pairs at or above threshold. O(n^2): fine for thousands, use ANN (Chapter 9) beyond."""
    m = l2_normalize_rows(matrix)
    sims = m @ m.T
    i, j = np.where(np.triu(sims, k=1) >= threshold)
    return sorted(((int(a), int(b), float(sims[a, b])) for a, b in zip(i, j)), key=lambda x: -x[2])


def duplicate_groups(n: int, pairs: Sequence[tuple[int, int, float]]) -> list[list[int]]:
    """Union-find over duplicate pairs. Each group keeps its lowest index as the canonical item."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _ in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups: dict[int, list[int]] = {}
    for x in range(n):
        groups.setdefault(find(x), []).append(x)
    return [g for g in groups.values() if len(g) > 1]


__all__ = ["LabeledPair", "ThresholdPoint", "duplicate_groups", "find_duplicates", "pair_scores", "select_threshold", "sweep_thresholds"]
