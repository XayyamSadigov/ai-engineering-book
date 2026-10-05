# path: book/projects/examples/ch08/embedlab/usecases/classify.py
"""Ticket classification without training a model: k-nearest-neighbour vote and nearest
centroid, both with an abstain rule so uncertain tickets go to a person or an LLM."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

import numpy as np

from ..vector_math import centroid, l2_normalize_rows


@dataclass(frozen=True)
class Prediction:
    label: str | None  # None = abstain
    score: float
    margin: float
    evidence: list[tuple[int, float]] = field(default_factory=list)


class Classifier(Protocol):
    def fit(self, vectors: np.ndarray, labels: Sequence[str]) -> "Classifier": ...
    def predict(self, vector: np.ndarray) -> Prediction: ...


class KNNClassifier:
    """Similarity-weighted vote among the k most similar labeled examples. Adding a labeled
    example changes behavior immediately; there is no training step."""

    def __init__(self, k: int = 5, min_similarity: float = 0.0, min_margin: float = 0.0) -> None:
        self.k, self.min_similarity, self.min_margin = k, min_similarity, min_margin
        self.matrix = np.zeros((0, 0))
        self.labels: list[str] = []

    def fit(self, vectors: np.ndarray, labels: Sequence[str]) -> "KNNClassifier":
        self.matrix = l2_normalize_rows(vectors)
        self.labels = list(labels)
        return self

    def predict(self, vector: np.ndarray) -> Prediction:
        sims = self.matrix @ l2_normalize_rows(vector)[0]
        order = np.argsort(-sims, kind="stable")[: self.k]
        votes: dict[str, float] = defaultdict(float)
        for i in order:
            votes[self.labels[i]] += max(0.0, float(sims[i]))
        ranked = sorted(votes.items(), key=lambda kv: -kv[1])
        total = sum(votes.values()) or 1.0
        top_label, top_vote = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = (top_vote - second) / total
        evidence = [(int(i), float(sims[i])) for i in order]
        if float(sims[order[0]]) < self.min_similarity or margin < self.min_margin:
            return Prediction(None, float(sims[order[0]]), margin, evidence)
        return Prediction(top_label, top_vote / total, margin, evidence)


class CentroidClassifier:
    """One normalized mean vector per class; predict the closest. Cheap (one row per class),
    robust to label noise, weak when a class has several distinct sub-topics."""

    def __init__(self, min_similarity: float = 0.0, min_margin: float = 0.0) -> None:
        self.min_similarity, self.min_margin = min_similarity, min_margin
        self.classes: list[str] = []
        self.centroids = np.zeros((0, 0))

    def fit(self, vectors: np.ndarray, labels: Sequence[str]) -> "CentroidClassifier":
        x = l2_normalize_rows(vectors)
        self.classes = sorted(set(labels))
        lab = np.asarray(labels)
        self.centroids = np.vstack([centroid(x[lab == c]) for c in self.classes])
        return self

    def predict(self, vector: np.ndarray) -> Prediction:
        sims = self.centroids @ l2_normalize_rows(vector)[0]
        order = np.argsort(-sims, kind="stable")
        top = float(sims[order[0]])
        margin = top - float(sims[order[1]]) if len(order) > 1 else top
        evidence = [(int(i), float(sims[i])) for i in order[:3]]
        if top < self.min_similarity or margin < self.min_margin:
            return Prediction(None, top, margin, evidence)
        return Prediction(self.classes[order[0]], top, margin, evidence)


@dataclass(frozen=True)
class ClassificationReport:
    accuracy_on_answered: float
    coverage: float  # share of items the classifier did not abstain on
    overall_accuracy: float  # abstentions count as wrong
    confusions: dict[tuple[str, str], int]


def leave_one_out(vectors: np.ndarray, labels: Sequence[str], make: Callable[[], Classifier]) -> ClassificationReport:
    """Hold out each item in turn, fit on the rest, predict it. Honest for small labeled sets."""
    n = len(labels)
    labels = list(labels)
    correct = answered = 0
    confusions: dict[tuple[str, str], int] = defaultdict(int)
    for i in range(n):
        mask = np.arange(n) != i
        clf = make().fit(vectors[mask], [labels[j] for j in range(n) if j != i])
        pred = clf.predict(vectors[i])
        if pred.label is None:
            continue
        answered += 1
        if pred.label == labels[i]:
            correct += 1
        else:
            confusions[(labels[i], pred.label)] += 1
    return ClassificationReport(
        accuracy_on_answered=correct / answered if answered else 0.0,
        coverage=answered / n if n else 0.0,
        overall_accuracy=correct / n if n else 0.0,
        confusions=dict(confusions),
    )


__all__ = ["CentroidClassifier", "ClassificationReport", "Classifier", "KNNClassifier", "Prediction", "leave_one_out"]
