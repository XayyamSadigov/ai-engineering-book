# path: book/projects/examples/ch25/tests/test_classification.py
from __future__ import annotations

import pytest

from evalkit import Dataset, EvalCase, run_target

from taskevals.classification import LabelEvaluator, classification_aggregates, evaluate_cascade
from taskevals.suites import LABELS, run_suite


def _run(pairs: list[tuple[str, str, float]]):
    ds = Dataset([EvalCase(id=f"c{i}", input={}, expected={"category": g}) for i, (g, _, _) in enumerate(pairs)],
                 name="toy")
    preds = {f"c{i}": {"category": p, "confidence": c} for i, (_, p, c) in enumerate(pairs)}
    return run_target(lambda case: preds[case.id], ds, evaluators=[LabelEvaluator(["a", "b", "rare"])], concurrency=1)


def test_accuracy_hides_a_rare_class_that_macro_f1_and_recall_expose() -> None:
    pairs = [("a", "a", 0.9)] * 45 + [("b", "b", 0.9)] * 45 + [("rare", "a", 0.9)] * 10
    agg = classification_aggregates(_run(pairs), ["a", "b", "rare"])
    assert agg.accuracy == pytest.approx(0.9)
    assert agg.per_class_recall["rare"] == 0.0
    assert agg.macro_f1 < 0.65
    assert agg.metric("recall:rare") == 0.0


def test_invalid_label_counts_as_wrong_not_as_crash() -> None:
    agg = classification_aggregates(_run([("a", "a", 0.9), ("b", "banana", 0.4)]), ["a", "b", "rare"])
    assert agg.accuracy == 0.5 and agg.per_class_recall["b"] == 0.0


def test_overconfident_scorer_has_high_ece() -> None:
    pairs = [("a", "a", 0.95)] * 6 + [("a", "b", 0.95)] * 4
    agg = classification_aggregates(_run(pairs), ["a", "b", "rare"], n_bins=5)
    assert agg.ece == pytest.approx(0.35, abs=1e-9)


def test_candidate_fixes_security_recall_and_keeps_injection_cases() -> None:
    base, cand = run_suite("classification", "baseline"), run_suite("classification", "candidate")
    a_base, a_cand = classification_aggregates(base, LABELS), classification_aggregates(cand, LABELS)
    assert a_cand.per_class_recall["security_report"] > a_base.per_class_recall["security_report"]
    assert not [c for c in cand.failing_cases("label_correct") if c.startswith("CL-ADV")]


def test_cascade_is_evaluated_as_one_system() -> None:
    gold = ["a", "b", "a", "b"]
    small = [("a", 0.9), ("a", 0.4), ("b", 0.45), ("b", 0.95)]
    large = ["a", "b", "a", "a"]
    no_esc, esc = evaluate_cascade(gold, small, large, [0.0, 0.5], cost_small=1, cost_large=20)
    assert (no_esc.accuracy, no_esc.escalation_rate, no_esc.cost_per_case) == (0.5, 0.0, 1.0)
    assert (esc.accuracy, esc.escalation_rate, esc.cost_per_case) == (1.0, 0.5, 11.0)
