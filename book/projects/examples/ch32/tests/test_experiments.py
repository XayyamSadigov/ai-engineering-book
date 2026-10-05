# path: book/projects/examples/ch32/tests/test_experiments.py
from __future__ import annotations

import json

import pytest

from northwind_triage.experiments import (
    ArmStats,
    CanaryPolicy,
    canary_decision,
    main,
    run_shadow,
    sample_size_per_arm,
    two_proportion_test,
)


def test_sample_size_worked_numbers() -> None:
    # Illustrative: 80% baseline routing accuracy, detect +3 points, alpha 0.05, power 0.8.
    assert sample_size_per_arm(0.80, 0.03) == 2629
    # A third of the effect needs roughly nine times the traffic.
    assert sample_size_per_arm(0.80, 0.01) == 24641
    with pytest.raises(ValueError):
        sample_size_per_arm(0.99, 0.02)


def test_two_proportion_test_detects_real_difference() -> None:
    t = two_proportion_test(800, 1000, 860, 1000)
    assert t.diff == pytest.approx(0.06)
    assert t.p_value < 0.01 and t.ci_low > 0


def arm(**kw) -> ArmStats:
    base = dict(requests=5000, task_successes=4000, errors=50, safety_violations=0,
                p95_latency_ms=1800, cost_per_request=0.0040)
    return ArmStats(**{**base, **kw})


def test_canary_promotes_when_guardrails_hold() -> None:
    d = canary_decision(arm(), arm(requests=3000, task_successes=2430, errors=27))
    assert d.action == "promote", d.reasons


def test_canary_rolls_back_on_any_safety_violation_even_with_little_data() -> None:
    d = canary_decision(arm(), arm(requests=40, task_successes=35, errors=0, safety_violations=1))
    assert d.action == "rollback"


def test_canary_rolls_back_on_latency_or_cost() -> None:
    assert canary_decision(arm(), arm(requests=1000, task_successes=800, p95_latency_ms=2600)).action == "rollback"
    assert canary_decision(arm(), arm(requests=1000, task_successes=800, cost_per_request=0.0050)).action == "rollback"


def test_canary_holds_until_enough_data() -> None:
    assert canary_decision(arm(), arm(requests=100, task_successes=80, errors=1)).action == "hold"


def test_canary_rolls_back_on_significant_quality_drop() -> None:
    d = canary_decision(arm(), arm(requests=2000, task_successes=1480, errors=20))   # 74% vs 80%
    assert d.action == "rollback"


def test_canary_holds_when_a_large_drop_cannot_be_excluded() -> None:
    d = canary_decision(arm(), arm(requests=450, task_successes=355, errors=4), CanaryPolicy())
    assert d.action == "hold"


def test_shadow_never_affects_served_output() -> None:
    def primary(x: int) -> str:
        return "even" if x % 2 == 0 else "odd"

    def shadow(x: int) -> str:
        if x == 3:
            raise RuntimeError("shadow crashed")
        return "even" if x % 4 == 0 else "odd"

    served, report = run_shadow(list(range(8)), primary, shadow)
    assert served == [primary(x) for x in range(8)]
    assert report.shadow_errors == 1
    assert report.total == 8
    assert 0 < report.agreement_rate < 1
    assert all(i in (2, 6) for i, _, _ in report.disagreements)


def test_cli_exit_codes_drive_the_pipeline(tmp_path, capsys) -> None:
    b, c = tmp_path / "b.json", tmp_path / "c.json"
    b.write_text(arm().model_dump_json())
    c.write_text(arm(requests=1000, task_successes=800, safety_violations=2).model_dump_json())
    assert main(["canary", "--baseline", str(b), "--canary", str(c)]) == 1
    assert json.loads(capsys.readouterr().out)["action"] == "rollback"
    assert main(["sample-size", "--baseline", "0.8", "--mde", "0.03"]) == 0
