# path: book/projects/p1-extraction-api/extraction_api/eval/calibration.py
"""Choose the accept threshold from data, and check whether scores mean what they say.

The question a threshold answers is operational: "among documents we auto-accept, what
fraction are fully correct?" Pick the lowest threshold whose accepted set meets the target
precision; everything below it goes to review. Coverage (the auto-accept rate) is the price.
Fit on a validation split and report on a held-out split, never on the same data.
"""
from __future__ import annotations

from pydantic import BaseModel


class ThresholdChoice(BaseModel):
    threshold: float
    precision: float
    coverage: float
    accepted: int
    total: int


class ReliabilityBin(BaseModel):
    lower: float
    upper: float
    count: int
    mean_score: float
    accuracy: float


def choose_threshold(scores: list[float], correct: list[bool], target_precision: float) -> ThresholdChoice | None:
    if len(scores) != len(correct) or not scores:
        raise ValueError("scores and correct must be non-empty and of equal length")
    n = len(scores)
    best: ThresholdChoice | None = None
    for t in sorted(set(scores), reverse=True):  # descending: coverage only grows
        accepted = [c for s, c in zip(scores, correct) if s >= t]
        precision = sum(accepted) / len(accepted)
        if precision >= target_precision:
            best = ThresholdChoice(threshold=t, precision=precision, coverage=len(accepted) / n,
                                   accepted=len(accepted), total=n)
    return best


def reliability(scores: list[float], correct: list[bool], bins: int = 5) -> list[ReliabilityBin]:
    out: list[ReliabilityBin] = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, s in enumerate(scores) if lo <= s < hi or (b == bins - 1 and s == 1.0)]
        if not idx:
            continue
        out.append(ReliabilityBin(
            lower=lo, upper=hi, count=len(idx),
            mean_score=sum(scores[i] for i in idx) / len(idx),
            accuracy=sum(correct[i] for i in idx) / len(idx),
        ))
    return out


def expected_calibration_error(scores: list[float], correct: list[bool], bins: int = 5) -> float:
    n = len(scores)
    return sum(b.count / n * abs(b.mean_score - b.accuracy) for b in reliability(scores, correct, bins)) if n else 0.0


__all__ = ["ThresholdChoice", "ReliabilityBin", "choose_threshold", "reliability", "expected_calibration_error"]
