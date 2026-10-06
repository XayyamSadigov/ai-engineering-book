# path: book/projects/examples/ch33/eval_protocol.py
"""Evaluation protocol for a fine-tuning candidate (Chapter 33).

The protocol answers one question: does the candidate earn its place against the baselines on
a frozen holdout, including the slices and the operating costs? It is deliberately metric-only
and model-agnostic: you feed it predictions, it never calls a model. That keeps the comparison
honest (every system sees identical inputs) and offline (tests run without a provider).

Metrics are implemented by hand so you can read them. Chapter 24 (evalkit) owns the general
statistics machinery; this module stays specific to classification candidates.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Callable, Iterable

import numpy as np
from pydantic import BaseModel, Field


class HoldoutExample(BaseModel):
    id: str
    text: str
    label: str
    slices: dict[str, str] = Field(default_factory=dict)  # e.g. {"language": "de", "length": "long"}


class Prediction(BaseModel):
    example_id: str
    label: str
    confidence: float = 1.0  # probability of the predicted label; 1.0 when the system has none
    latency_ms: float = 0.0
    cost_usd: float = 0.0  # illustrative accounting; use PricingTable from aie_core in projects


class SystemRun(BaseModel):
    """All predictions of one system (a baseline or the candidate) on the frozen holdout."""

    name: str
    predictions: list[Prediction]

    def by_id(self) -> dict[str, Prediction]:
        return {p.example_id: p for p in self.predictions}


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def per_class_prf(y_true: list[str], y_pred: list[str], labels: Iterable[str]) -> dict[str, dict[str, float]]:
    tp, fp, fn = Counter(), Counter(), Counter()
    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    out: dict[str, dict[str, float]] = {}
    for label in labels:
        precision = tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] else 0.0
        recall = tp[label] / (tp[label] + fn[label]) if tp[label] + fn[label] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp[label] + fn[label]}
    return out


def macro_f1(y_true: list[str], y_pred: list[str], labels: Iterable[str]) -> float:
    """Unweighted mean of per-class F1. Every class counts equally, so rare classes can sink it.
    That is the point: a 40-way classifier that nails the 5 big classes and guesses the rest
    has a great accuracy and a poor macro-F1."""
    labels = list(labels)
    prf = per_class_prf(y_true, y_pred, labels)
    return float(np.mean([prf[l]["f1"] for l in labels])) if labels else 0.0


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0


class CalibrationBin(BaseModel):
    lo: float
    hi: float
    count: int
    mean_confidence: float
    accuracy: float


def calibration_bins(confidences: list[float], correct: list[bool], n_bins: int = 10) -> tuple[list[CalibrationBin], float]:
    """Reliability bins plus expected calibration error (ECE).

    A system is calibrated when predictions made with confidence 0.8 are right about 80% of the
    time. Calibration matters here because the escalation policy (send low-confidence tickets
    to a stronger model or a human) only works if confidence means something.
    """
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[CalibrationBin] = []
    ece = 0.0
    total = len(conf)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf >= lo) & ((conf < hi) if i < n_bins - 1 else (conf <= hi))
        count = int(mask.sum())
        if count == 0:
            bins.append(CalibrationBin(lo=lo, hi=hi, count=0, mean_confidence=0.0, accuracy=0.0))
            continue
        mean_conf = float(conf[mask].mean())
        acc = float(corr[mask].mean())
        ece += (count / total) * abs(acc - mean_conf)
        bins.append(CalibrationBin(lo=lo, hi=hi, count=count, mean_confidence=mean_conf, accuracy=acc))
    return bins, float(ece)


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------


class EvalReport(BaseModel):
    system: str
    n: int
    accuracy: float
    macro_f1: float
    per_class: dict[str, dict[str, float]]
    ece: float
    calibration: list[CalibrationBin]
    latency_p50_ms: float
    latency_p95_ms: float
    cost_per_1k_usd: float
    slice_macro_f1: dict[str, dict[str, float]]  # slice_name -> slice_value -> macro-F1
    coverage: float  # fraction of holdout examples that received a prediction


def evaluate(holdout: list[HoldoutExample], run: SystemRun, labels: list[str] | None = None) -> EvalReport:
    labels = labels or sorted({h.label for h in holdout})
    preds = run.by_id()
    paired = [(h, preds[h.id]) for h in holdout if h.id in preds]
    if not paired:
        raise ValueError(f"{run.name}: no predictions matched the holdout ids")
    y_true = [h.label for h, _ in paired]
    y_pred = [p.label for _, p in paired]
    correct = [t == p for t, p in zip(y_true, y_pred)]
    bins, ece = calibration_bins([p.confidence for _, p in paired], correct)
    latencies = np.array([p.latency_ms for _, p in paired])
    costs = np.array([p.cost_usd for _, p in paired])

    slice_scores: dict[str, dict[str, float]] = defaultdict(dict)
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, (h, _) in enumerate(paired):
        for name, value in h.slices.items():
            groups[(name, value)].append(i)
    for (name, value), idx in groups.items():
        slice_scores[name][value] = macro_f1([y_true[i] for i in idx], [y_pred[i] for i in idx],
                                             sorted({y_true[i] for i in idx}))

    return EvalReport(
        system=run.name,
        n=len(paired),
        accuracy=accuracy(y_true, y_pred),
        macro_f1=macro_f1(y_true, y_pred, labels),
        per_class=per_class_prf(y_true, y_pred, labels),
        ece=ece,
        calibration=bins,
        latency_p50_ms=float(np.percentile(latencies, 50)),
        latency_p95_ms=float(np.percentile(latencies, 95)),
        cost_per_1k_usd=float(costs.mean() * 1000),
        slice_macro_f1=dict(slice_scores),
        coverage=len(paired) / len(holdout),
    )


def comparison_table(reports: list[EvalReport]) -> str:
    """Markdown table for the decision review. Numbers from fakes are illustrative by construction."""
    head = "| system | n | macro-F1 | accuracy | ECE | p95 ms | cost/1k |\n|---|---|---|---|---|---|---|"
    rows = [
        f"| {r.system} | {r.n} | {r.macro_f1:.3f} | {r.accuracy:.3f} | {r.ece:.3f} | {r.latency_p95_ms:.0f} | {r.cost_per_1k_usd:.2f} |"
        for r in reports
    ]
    return "\n".join([head, *rows])


# --------------------------------------------------------------------------------------
# Decision rule
# --------------------------------------------------------------------------------------


class ShipRule(BaseModel):
    """Thresholds agreed *before* training. Changing them after seeing results is how teams ship regressions."""

    max_macro_f1_drop_vs_reference: float = 0.01  # candidate may trail the strong baseline by this much
    critical_labels: list[str] = Field(default_factory=list)
    min_critical_recall: float = 0.90
    min_recall_floor: float = 0.50  # no class may collapse below this
    max_slice_drop: float = 0.03  # per-slice macro-F1 may not trail the reference by more
    min_cost_reduction: float = 0.50  # candidate must cost at most 50% of the reference
    max_latency_p95_ms: float | None = None
    max_ece: float = 0.10
    regression_suite_min_pass: float = 0.98


class ShipDecision(BaseModel):
    ship: bool
    reasons: list[str]


def decide_ship(
    candidate: EvalReport,
    reference: EvalReport,
    rule: ShipRule,
    regression_pass_rate: float = 1.0,
) -> ShipDecision:
    """Mechanical gate. Every failing check becomes a reason; an empty list means ship.

    The reference is normally the prompted strong model: the thing you are trying to replace.
    """
    reasons: list[str] = []
    # Same frozen holdout, every row answered: otherwise the two reports measure different things.
    if candidate.coverage < 1.0 or candidate.n != reference.n:
        reasons.append(f"candidate answered {candidate.coverage:.0%} of the holdout (n={candidate.n}) "
                       f"versus reference n={reference.n}; compare on the same complete holdout")
    if candidate.macro_f1 < reference.macro_f1 - rule.max_macro_f1_drop_vs_reference:
        reasons.append(f"macro-F1 {candidate.macro_f1:.3f} trails reference {reference.macro_f1:.3f} "
                       f"by more than {rule.max_macro_f1_drop_vs_reference:.3f}")
    for label in rule.critical_labels:
        rec = candidate.per_class.get(label, {}).get("recall", 0.0)
        if rec < rule.min_critical_recall:
            reasons.append(f"critical label {label!r} recall {rec:.3f} < {rule.min_critical_recall:.2f}")
    for label, m in candidate.per_class.items():
        if m["support"] > 0 and m["recall"] < rule.min_recall_floor:
            reasons.append(f"label {label!r} recall {m['recall']:.3f} below floor {rule.min_recall_floor:.2f}")
    for name, values in candidate.slice_macro_f1.items():
        for value, score in values.items():
            ref_score = reference.slice_macro_f1.get(name, {}).get(value)
            if ref_score is not None and score < ref_score - rule.max_slice_drop:
                reasons.append(f"slice {name}={value} macro-F1 {score:.3f} trails reference {ref_score:.3f}")
    if reference.cost_per_1k_usd > 0:
        ratio = candidate.cost_per_1k_usd / reference.cost_per_1k_usd
        if ratio > 1 - rule.min_cost_reduction:
            reasons.append(f"cost ratio {ratio:.2f} does not reach the required reduction of {rule.min_cost_reduction:.0%}")
    if rule.max_latency_p95_ms is not None and candidate.latency_p95_ms > rule.max_latency_p95_ms:
        reasons.append(f"p95 latency {candidate.latency_p95_ms:.0f} ms exceeds {rule.max_latency_p95_ms:.0f} ms")
    if candidate.ece > rule.max_ece:
        reasons.append(f"ECE {candidate.ece:.3f} exceeds {rule.max_ece:.2f}; escalation policy cannot trust confidence")
    if regression_pass_rate < rule.regression_suite_min_pass:
        reasons.append(f"regression suite pass rate {regression_pass_rate:.3f} < {rule.regression_suite_min_pass:.2f}")
    return ShipDecision(ship=not reasons, reasons=reasons)


# --------------------------------------------------------------------------------------
# Escalation (cascade) evaluation
# --------------------------------------------------------------------------------------


def cascade(
    small: SystemRun,
    strong: SystemRun,
    threshold: float,
    name: str = "cascade",
) -> SystemRun:
    """Compose a system: take the small model's answer when it is confident, otherwise the strong model's.

    Cost and latency add up on escalated items (the small model was already called). Evaluate
    the *composed* system; the small model's standalone score is not what users experience.
    """
    strong_by_id = strong.by_id()
    merged: list[Prediction] = []
    for p in small.predictions:
        if p.confidence >= threshold or p.example_id not in strong_by_id:
            merged.append(p)
        else:
            s = strong_by_id[p.example_id]
            merged.append(Prediction(example_id=p.example_id, label=s.label, confidence=s.confidence,
                                     latency_ms=p.latency_ms + s.latency_ms, cost_usd=p.cost_usd + s.cost_usd))
    return SystemRun(name=name, predictions=merged)


def escalation_rate(small: SystemRun, threshold: float) -> float:
    return sum(p.confidence < threshold for p in small.predictions) / len(small.predictions) if small.predictions else 0.0


def choose_threshold(
    validation: list[HoldoutExample],
    small: SystemRun,
    strong: SystemRun,
    labels: list[str],
    target_macro_f1: float,
    candidates: Iterable[float] = tuple(np.round(np.arange(0.5, 1.0, 0.05), 2)),
    score: Callable[[EvalReport], float] = lambda r: r.macro_f1,
) -> float | None:
    """Pick the lowest threshold whose cascade meets the target on *validation* data.

    Lowest, because a lower threshold escalates less and therefore costs less. The test set is
    never consulted here; if it were, its score would stop being an estimate of production.
    """
    for t in sorted(candidates):
        report = evaluate(validation, cascade(small, strong, t), labels)
        if score(report) >= target_macro_f1:
            return float(t)
    return None
