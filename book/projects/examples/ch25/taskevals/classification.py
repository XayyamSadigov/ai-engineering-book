# path: book/projects/examples/ch25/taskevals/classification.py
"""Classification evaluators: per-case correctness plus run-level macro-F1, per-class recall,
and calibration, computed from the Run record alone so a CI gate can recompute them later.

Macro-F1 and calibration are properties of the whole run, not of a case, so they do not fit
evalkit's per-case metric rules. The per-case evaluator stores (gold, predicted, confidence)
in the score detail; `classification_aggregates` rebuilds the confusion matrix from that.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from evalkit import EvalCase, Run, Score
from evalkit.metrics import ConfusionMatrix, brier_score, calibration_bins, expected_calibration_error


class LabelEvaluator:
    """`output` is {"category": str, "confidence": float}; `case.expected` is {"category": str}."""

    name = "label_correct"
    version = "1"
    metric_names = ["label_correct", "label_valid"]

    def __init__(self, labels: Sequence[str]) -> None:
        self.labels = list(labels)

    def __call__(self, case: EvalCase, output: Any) -> list[Score]:
        pred = output.get("category")
        gold = case.expected["category"]
        conf = float(output.get("confidence", 1.0))
        ok = pred == gold
        return [
            Score(name="label_correct", value=float(ok), passed=ok,
                  detail={"gold": gold, "pred": pred, "confidence": conf}),
            Score(name="label_valid", value=float(pred in self.labels), passed=pred in self.labels),
        ]


class ClassificationAggregates(BaseModel):
    n: int
    accuracy: float
    macro_f1: float
    per_class_recall: dict[str, float]
    per_class_support: dict[str, int]
    ece: float
    brier: float
    most_confused: list[tuple[str, str, int]]
    reliability: list[dict[str, float]]

    def metric(self, name: str) -> float:
        """Lookup used by the release gate: macro_f1, accuracy, ece, brier, recall:<class>."""
        if name.startswith("recall:"):
            return self.per_class_recall.get(name.split(":", 1)[1], float("nan"))
        return float(getattr(self, name))


def classification_aggregates(run: Run, labels: Sequence[str], *, n_bins: int = 5) -> ClassificationAggregates:
    rows = [r.details["label_correct"] for r in run.results if isinstance(r.details.get("label_correct"), dict)]
    if not rows:
        raise ValueError("run has no label_correct details; was LabelEvaluator used?")
    gold = [d["gold"] for d in rows]
    # A prediction outside the label set is wrong; map it to a sentinel so the matrix accepts it.
    pred = [d["pred"] if d["pred"] in labels else "__invalid__" for d in rows]
    cm = ConfusionMatrix(gold, pred, labels=[*labels, "__invalid__"])
    correct = [g == p for g, p in zip(gold, pred)]
    conf = [min(1.0, max(0.0, float(d["confidence"]))) for d in rows]
    return ClassificationAggregates(
        n=len(rows),
        accuracy=cm.accuracy,
        macro_f1=cm.macro(labels).f1,
        per_class_recall={lab: cm.per_label(lab).recall for lab in labels if cm.support(lab)},
        per_class_support={lab: cm.support(lab) for lab in labels if cm.support(lab)},
        ece=expected_calibration_error(correct, conf, n_bins=n_bins),
        brier=brier_score(correct, conf),
        most_confused=[(str(t), str(p), c) for t, p, c in cm.most_confused(5)],
        reliability=[{"lower": b.lower, "upper": b.upper, "count": b.count, "confidence": b.mean_confidence,
                      "accuracy": b.observed_rate} for b in calibration_bins(correct, conf, n_bins)],
    )


class CascadePoint(BaseModel):
    threshold: float
    accuracy: float
    escalation_rate: float
    cost_per_case: float


def evaluate_cascade(
    gold: Sequence[str],
    small: Sequence[tuple[str, float]],
    large: Sequence[str],
    thresholds: Sequence[float],
    *,
    cost_small: float = 1.0,
    cost_large: float = 20.0,
) -> list[CascadePoint]:
    """Evaluate small-model-first with escalation below a confidence threshold, as one system.

    Every case pays for the small model; escalated cases also pay for the large one. Costs are
    illustrative units. Choose the threshold on dev data and report the chosen point on the holdout.
    """
    out = []
    n = len(gold)
    for th in thresholds:
        correct = esc = 0
        for g, (s_label, s_conf), l_label in zip(gold, small, large):
            if s_conf < th:
                esc += 1
                correct += l_label == g
            else:
                correct += s_label == g
        out.append(CascadePoint(threshold=th, accuracy=correct / n, escalation_rate=esc / n,
                                cost_per_case=cost_small + cost_large * esc / n))
    return out


__all__ = ["LabelEvaluator", "ClassificationAggregates", "classification_aggregates", "CascadePoint",
           "evaluate_cascade"]
