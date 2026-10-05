# path: book/projects/evalkit/tests/test_metrics.py
import math

import pytest
from pydantic import BaseModel

from evalkit.metrics import (
    ConfusionMatrix,
    best_threshold,
    brier_score,
    calibration_bins,
    contains,
    exact_match,
    expected_calibration_error,
    field_prf,
    forbids,
    json_schema_valid,
    numeric_close,
    parse_number,
    set_precision_recall,
    threshold_sweep,
)


# ------------------------------------------------------------------ deterministic
def test_exact_match_normalizes_strings_only_when_asked():
    assert exact_match("  The VPN!", "the vpn") == 1.0
    assert exact_match("The VPN", "the vpn", normalize=False) == 0.0
    assert exact_match({"a": 1}, {"a": 1}) == 1.0


def test_contains_and_forbids():
    ok, missing = contains("Submit within 30 days, receipts required.", ["30 days", "receipts", "manager"])
    assert not ok and missing == ["manager"]
    ok, found = contains("refund issued", ["refund", "credit"], mode="any")
    assert ok and found == ["refund"]
    ok, hits = forbids("Your password is hunter2", ["password is"])
    assert not ok and hits == ["password is"]


def test_numbers():
    assert parse_number("Total: 1,234.50 EUR") == 1234.5
    assert parse_number("none") is None
    assert numeric_close("1,000.4", 1000, abs_tol=0.5)
    assert not numeric_close(103, 100, rel_tol=0.02)
    assert numeric_close(101, 100, rel_tol=0.02)


def test_set_precision_recall_conventions():
    r = set_precision_recall(["doc-1", "doc-3"], ["doc-1", "doc-2"])
    assert (r.tp, r.fp, r.fn) == (1, 1, 1) and r.precision == 0.5 and r.recall == 0.5
    empty = set_precision_recall([], [])
    assert empty.precision == empty.recall == empty.f1 == 1.0
    nothing_cited = set_precision_recall([], ["doc-1"])
    assert nothing_cited.precision == 1.0 and nothing_cited.recall == 0.0


def test_field_prf_counts_wrong_values_as_fp_and_fn():
    pred = {"vendor": "ACME GmbH", "total": 100.0, "currency": "EUR", "po": "PO-1"}
    gold = {"vendor": "acme gmbh", "total": 120.0, "currency": "EUR", "po": None, "due": "2026-03-01"}
    r = field_prf(pred, gold)
    assert r.per_field == {"currency": "tp", "due": "fn", "po": "fp", "total": "fp+fn", "vendor": "tp"}
    assert (r.tp, r.fp, r.fn) == (2, 2, 2)
    assert r.precision == pytest.approx(0.5) and r.recall == pytest.approx(0.5)


def test_json_schema_valid_dict_schema():
    schema = {
        "type": "object",
        "required": ["category", "priority"],
        "additionalProperties": False,
        "properties": {
            "category": {"type": "string", "enum": ["vpn_network", "hardware"]},
            "priority": {"type": "integer", "minimum": 1, "maximum": 4},
            "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
        },
    }
    assert json_schema_valid('{"category": "hardware", "priority": 2}', schema) == (True, [])
    ok, errs = json_schema_valid({"category": "printer", "priority": 9, "tags": ["a", 1, "c"], "x": 1}, schema)
    assert not ok
    joined = " ".join(errs)
    for needle in ["not in enum", "maximum", "$.tags[1]", "more than 2", "unexpected field 'x'"]:
        assert needle in joined
    assert json_schema_valid("{oops", schema)[0] is False
    assert json_schema_valid({"category": "hardware", "priority": True}, schema)[0] is False  # bool is not int


def test_json_schema_valid_pydantic_model():
    class T(BaseModel):
        category: str
        priority: int

    assert json_schema_valid({"category": "x", "priority": 1}, T)[0]
    ok, errs = json_schema_valid({"category": "x"}, T)
    assert not ok and "priority" in errs[0]


# ------------------------------------------------------------------ classification
def test_confusion_matrix_matches_hand_computation_and_sklearn():
    y_true = ["vpn", "vpn", "hw", "hw", "hw", "sec"]
    y_pred = ["vpn", "hw", "hw", "hw", "vpn", "vpn"]
    cm = ConfusionMatrix(y_true, y_pred)
    assert cm.count("hw", "vpn") == 1 and cm.tp("hw") == 2
    assert cm.per_label("sec").precision == 0.0  # never predicted: zero, not a vacuous 1
    assert cm.accuracy == pytest.approx(0.5)
    assert cm.micro().f1 == pytest.approx(cm.accuracy)
    sk = pytest.importorskip("sklearn.metrics")
    assert cm.macro().f1 == pytest.approx(sk.f1_score(y_true, y_pred, average="macro", zero_division=0))
    assert cm.macro().recall == pytest.approx(sk.recall_score(y_true, y_pred, average="macro", zero_division=0))
    assert ("hw", "vpn", 1) in cm.most_confused()
    assert cm.to_markdown().startswith("| true \\ pred |")


def test_confusion_matrix_rejects_unknown_labels():
    with pytest.raises(ValueError):
        ConfusionMatrix(["a"], ["b"], labels=["a"])


def test_threshold_sweep_reproduces_fraud_cost_example():
    """1,000 transactions, 50 fraud. Costs: missed fraud 1000, review 5 per flag (chapter text)."""
    y = [True] * 50 + [False] * 950
    # Scores built so that at 0.5 we catch 45 and flag 95 legit; at 0.8 we catch 35 and flag 20 legit.
    s = [0.9] * 35 + [0.6] * 10 + [0.1] * 5 + [0.9] * 20 + [0.6] * 75 + [0.1] * 855
    pts = {p.threshold: p for p in threshold_sweep(y, s, [0.5, 0.8], cost_fn=1000, cost_fp=0, cost_per_flag=5)}
    lo, hi = pts[0.5], pts[0.8]
    assert (lo.tp, lo.fp, lo.flagged) == (45, 95, 140)
    assert lo.precision == pytest.approx(45 / 140) and lo.recall == pytest.approx(0.9)
    assert (hi.tp, hi.fp) == (35, 20) and hi.precision == pytest.approx(35 / 55)
    assert lo.cost == 5700 and hi.cost == 15275
    assert best_threshold(list(pts.values())).threshold == 0.5
    assert best_threshold(list(pts.values()), by="f1").threshold == 0.8
    with pytest.raises(ValueError):
        best_threshold(list(pts.values()), min_recall=0.95)


def test_calibration_bins_ece_and_brier():
    y = [True] * 8 + [False] * 2 + [True] * 1 + [False] * 9
    p = [0.9] * 10 + [0.2] * 10
    bins = calibration_bins(y, p, n_bins=10)
    assert [(b.count, b.observed_rate) for b in bins] == [(10, 0.1), (10, 0.8)]
    assert expected_calibration_error(y, p) == pytest.approx(0.1)
    perfect = [True, False]
    assert brier_score(perfect, [1.0, 0.0]) == 0.0
    assert math.isclose(brier_score(perfect, [0.5, 0.5]), 0.25)
    with pytest.raises(ValueError):
        calibration_bins([True], [1.2])
