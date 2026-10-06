# path: book/projects/evalkit/evalkit/metrics/classification.py
"""Classification metrics, thresholds with costs, and calibration.

Routers, guardrails, abstention decisions, relevance filters, and LLM judges with a pass
threshold are all classifiers. Everything here applies to them unchanged.
"""
from __future__ import annotations

from collections.abc import Hashable, Sequence

from pydantic import BaseModel

from .deterministic import PRF, prf_from_counts


# ============================================================================ multi-class
class ConfusionMatrix:
    """Counts of (true label, predicted label). Rows are truth, columns are predictions."""

    def __init__(self, y_true: Sequence[Hashable], y_pred: Sequence[Hashable], labels: Sequence[Hashable] | None = None):
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        seen = list(dict.fromkeys([*y_true, *y_pred]))
        self.labels: list[Hashable] = list(labels) if labels is not None else sorted(seen, key=str)
        extra = set(seen) - set(self.labels)
        if extra:
            raise ValueError(f"labels missing from the label set: {sorted(map(str, extra))}")
        self._idx = {lab: i for i, lab in enumerate(self.labels)}
        n = len(self.labels)
        self.matrix: list[list[int]] = [[0] * n for _ in range(n)]
        for t, p in zip(y_true, y_pred):
            self.matrix[self._idx[t]][self._idx[p]] += 1
        self.n = len(y_true)

    def count(self, true_label: Hashable, pred_label: Hashable) -> int:
        return self.matrix[self._idx[true_label]][self._idx[pred_label]]

    def tp(self, label: Hashable) -> int:
        return self.count(label, label)

    def fp(self, label: Hashable) -> int:
        j = self._idx[label]
        return sum(row[j] for row in self.matrix) - self.tp(label)

    def fn(self, label: Hashable) -> int:
        return sum(self.matrix[self._idx[label]]) - self.tp(label)

    def support(self, label: Hashable) -> int:
        return sum(self.matrix[self._idx[label]])

    def per_label(self, label: Hashable) -> PRF:
        return prf_from_counts(self.tp(label), self.fp(label), self.fn(label), zero_division=0.0)

    @property
    def accuracy(self) -> float:
        return sum(self.matrix[i][i] for i in range(len(self.labels))) / self.n if self.n else 0.0

    def macro(self, labels: Sequence[Hashable] | None = None) -> PRF:
        """Unweighted mean over labels that occur in truth: every class counts equally."""
        labs = [lab for lab in (labels or self.labels) if self.support(lab) > 0]
        if not labs:
            return prf_from_counts(0, 0, 0)
        per = [self.per_label(lab) for lab in labs]
        p = sum(x.precision for x in per) / len(per)
        r = sum(x.recall for x in per) / len(per)
        f = sum(x.f1 for x in per) / len(per)
        return PRF(precision=p, recall=r, f1=f, tp=sum(x.tp for x in per), fp=sum(x.fp for x in per), fn=sum(x.fn for x in per))

    def micro(self) -> PRF:
        """Pooled counts: dominated by frequent classes. Equals accuracy for single-label tasks."""
        tp = sum(self.tp(lab) for lab in self.labels)
        fp = sum(self.fp(lab) for lab in self.labels)
        fn = sum(self.fn(lab) for lab in self.labels)
        return prf_from_counts(tp, fp, fn, zero_division=0.0)

    def most_confused(self, k: int = 5) -> list[tuple[Hashable, Hashable, int]]:
        pairs = [
            (t, p, self.count(t, p)) for t in self.labels for p in self.labels if t != p and self.count(t, p) > 0
        ]
        return sorted(pairs, key=lambda x: -x[2])[:k]

    def to_markdown(self) -> str:
        head = "| true \\ pred | " + " | ".join(map(str, self.labels)) + " |"
        sep = "|---" * (len(self.labels) + 1) + "|"
        rows = [f"| {lab} | " + " | ".join(str(v) for v in self.matrix[i]) + " |" for i, lab in enumerate(self.labels)]
        return "\n".join([head, sep, *rows])


def classification_report(
    y_true: Sequence[Hashable], y_pred: Sequence[Hashable], labels: Sequence[Hashable] | None = None
) -> dict[str, object]:
    cm = ConfusionMatrix(y_true, y_pred, labels)
    return {
        "n": cm.n,
        "accuracy": cm.accuracy,
        "macro": cm.macro().model_dump(),
        "micro": cm.micro().model_dump(),
        "per_label": {str(lab): {**cm.per_label(lab).model_dump(), "support": cm.support(lab)} for lab in cm.labels},
    }


# ============================================================================ binary thresholds
class ThresholdPoint(BaseModel):
    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    flagged: int
    cost: float


def binary_counts(y_true: Sequence[bool], scores: Sequence[float], threshold: float) -> tuple[int, int, int, int]:
    """(tp, fp, fn, tn) when predicting positive for score >= threshold."""
    tp = fp = fn = tn = 0
    for y, s in zip(y_true, scores):
        pred = s >= threshold
        if pred and y:
            tp += 1
        elif pred:
            fp += 1
        elif y:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def threshold_sweep(
    y_true: Sequence[bool],
    scores: Sequence[float],
    thresholds: Sequence[float] | None = None,
    *,
    cost_fp: float = 1.0,
    cost_fn: float = 1.0,
    cost_per_flag: float = 0.0,
) -> list[ThresholdPoint]:
    """Metrics and expected cost at each threshold.

    cost = fn * cost_fn + fp * cost_fp + (tp + fp) * cost_per_flag. `cost_per_flag` models a
    review or handling cost paid for every positive decision, right or wrong.
    """
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length")
    ths = sorted(set(thresholds)) if thresholds is not None else sorted(set(scores))
    out: list[ThresholdPoint] = []
    for th in ths:
        tp, fp, fn, tn = binary_counts(y_true, scores, th)
        prf = prf_from_counts(tp, fp, fn, zero_division=0.0)
        out.append(
            ThresholdPoint(
                threshold=th,
                tp=tp,
                fp=fp,
                fn=fn,
                tn=tn,
                precision=prf.precision,
                recall=prf.recall,
                f1=prf.f1,
                flagged=tp + fp,
                cost=fn * cost_fn + fp * cost_fp + (tp + fp) * cost_per_flag,
            )
        )
    return out


def best_threshold(points: Sequence[ThresholdPoint], *, by: str = "cost", min_recall: float | None = None,
                   min_precision: float | None = None) -> ThresholdPoint:
    """Pick the operating point: lowest cost (default) or highest F1, subject to floors."""
    eligible = [
        p for p in points
        if (min_recall is None or p.recall >= min_recall) and (min_precision is None or p.precision >= min_precision)
    ]
    if not eligible:
        raise ValueError("no threshold satisfies the constraints")
    if by == "cost":
        return min(eligible, key=lambda p: (p.cost, -p.threshold))
    if by == "f1":
        return max(eligible, key=lambda p: (p.f1, p.threshold))
    raise ValueError("by must be 'cost' or 'f1'")


# ============================================================================ calibration
class CalibrationBin(BaseModel):
    lower: float
    upper: float
    count: int
    mean_confidence: float
    observed_rate: float

    @property
    def gap(self) -> float:
        return abs(self.mean_confidence - self.observed_rate)


def calibration_bins(y_true: Sequence[bool], probs: Sequence[float], n_bins: int = 10) -> list[CalibrationBin]:
    """Equal-width reliability bins over [0, 1]. Empty bins are omitted."""
    if len(y_true) != len(probs):
        raise ValueError("y_true and probs must have the same length")
    buckets: list[list[tuple[bool, float]]] = [[] for _ in range(n_bins)]
    for y, p in zip(y_true, probs):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"probability out of range: {p}")
        buckets[min(int(p * n_bins), n_bins - 1)].append((bool(y), p))
    out = []
    for i, b in enumerate(buckets):
        if b:
            out.append(
                CalibrationBin(
                    lower=i / n_bins,
                    upper=(i + 1) / n_bins,
                    count=len(b),
                    mean_confidence=sum(p for _, p in b) / len(b),
                    observed_rate=sum(1 for y, _ in b if y) / len(b),
                )
            )
    return out


def expected_calibration_error(y_true: Sequence[bool], probs: Sequence[float], n_bins: int = 10) -> float:
    """ECE: count-weighted mean |confidence - observed rate| across bins."""
    bins = calibration_bins(y_true, probs, n_bins)
    n = sum(b.count for b in bins)
    return sum(b.count * b.gap for b in bins) / n if n else 0.0


def brier_score(y_true: Sequence[bool], probs: Sequence[float]) -> float:
    """Mean squared error between probability and outcome; rewards calibration and sharpness."""
    if len(y_true) != len(probs):
        raise ValueError("y_true and probs must have the same length")
    if not y_true:
        return 0.0
    return sum((p - float(y)) ** 2 for y, p in zip(y_true, probs)) / len(y_true)


__all__ = [
    "ConfusionMatrix",
    "classification_report",
    "ThresholdPoint",
    "binary_counts",
    "threshold_sweep",
    "best_threshold",
    "CalibrationBin",
    "calibration_bins",
    "expected_calibration_error",
    "brier_score",
]
