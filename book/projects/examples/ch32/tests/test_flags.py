# path: book/projects/examples/ch32/tests/test_flags.py
from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from northwind_triage.flags import BUCKETS, FlagConfig, FlagEvaluator, bucket

USERS = [f"user-{i}" for i in range(20_000)]


def evaluator(percent: float, **extra) -> FlagEvaluator:
    return FlagEvaluator.from_dict({
        "triage.model": {"allocations": [{"variant": "candidate", "percent": percent},
                                         {"variant": "control", "percent": 100 - percent}], **extra},
    }, environment="prod")


@given(st.text(min_size=1, max_size=40), st.text(max_size=20))
def test_bucket_is_deterministic_and_in_range(unit: str, salt: str) -> None:
    b = bucket(unit, salt)
    assert 0 <= b < BUCKETS
    assert b == bucket(unit, salt)


def test_allocation_matches_requested_percentage() -> None:
    ev = evaluator(10)
    share = sum(ev.evaluate("triage.model", u).variant == "candidate" for u in USERS) / len(USERS)
    assert 0.09 < share < 0.11


def test_ramp_up_is_monotonic_nobody_leaves_treatment() -> None:
    ev5 = evaluator(5)
    ev20 = ev5.ramp("triage.model", "candidate", 20)
    in5 = {u for u in USERS if ev5.evaluate("triage.model", u).variant == "candidate"}
    in20 = {u for u in USERS if ev20.evaluate("triage.model", u).variant == "candidate"}
    assert in5 <= in20
    assert len(in20) > 3 * len(in5)


def test_kill_switch_serves_safe_variant_to_everyone() -> None:
    ev = evaluator(50).kill("triage.model")
    assignments = {ev.evaluate("triage.model", u).variant for u in USERS[:2000]}
    assert assignments == {"control"}
    assert ev.evaluate("triage.model", "user-1").reason == "kill_switch"


def test_unknown_flag_is_safe_not_an_exception() -> None:
    a = evaluator(50).evaluate("triage.modle", "user-1")  # typo
    assert (a.variant, a.reason) == ("control", "unknown_flag")


def test_overrides_and_environment_scoping() -> None:
    ev = evaluator(0, overrides={"qa-alice": "candidate"})
    assert ev.evaluate("triage.model", "qa-alice").variant == "candidate"
    staged = evaluator(100, environments=["staging"])
    assert staged.evaluate("triage.model", "user-1").reason == "disabled_environment"


def test_different_salts_give_independent_assignments() -> None:
    """Two experiments with the same salt put the same users in treatment of both, so their
    effects cannot be separated. Default salt = flag name avoids that."""
    ev = FlagEvaluator.from_dict({
        "a": {"allocations": [{"variant": "t", "percent": 50}, {"variant": "control", "percent": 50}]},
        "b": {"allocations": [{"variant": "t", "percent": 50}, {"variant": "control", "percent": 50}]},
    })
    both = sum(ev.evaluate("a", u).variant == "t" and ev.evaluate("b", u).variant == "t" for u in USERS)
    assert 0.23 < both / len(USERS) < 0.27  # independent: ~25%, correlated would be ~50%


@pytest.mark.parametrize("allocs", [
    [{"variant": "candidate", "percent": 60}, {"variant": "control", "percent": 30}],   # sums to 90
    [{"variant": "control", "percent": 50}, {"variant": "control", "percent": 50}],     # duplicate
    [{"variant": "candidate", "percent": 100}],                                         # no safe variant
])
def test_invalid_configs_fail_at_load(allocs) -> None:
    with pytest.raises(ValidationError):
        FlagConfig(name="x", allocations=allocs)


def test_ramp_cannot_exceed_control() -> None:
    with pytest.raises(ValidationError):
        evaluator(50).ramp("triage.model", "candidate", 120)
