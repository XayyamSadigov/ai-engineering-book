# path: book/projects/reliability/tests/test_slo.py
import pytest

from reliability import RequestOutcome, SLOTargets, burn_rate, evaluate, should_page


def test_burn_rate_arithmetic():
    assert burn_rate(0.98, 0.995) == pytest.approx(4.0)
    assert burn_rate(0.995, 0.995) == pytest.approx(1.0)


def test_paging_requires_both_windows():
    assert should_page(20.0, 15.0)
    assert not should_page(20.0, 2.0)            # a blip
    assert not should_page(1.0, 15.0)            # already recovered


def test_evaluate_counts_degraded_as_available_but_reports_share():
    outcomes = (
        [RequestOutcome(ok=True, ttft_s=1.0, total_s=5.0, task_success=True)] * 90
        + [RequestOutcome(ok=True, ttft_s=1.2, total_s=6.0, degraded=True)] * 8
        + [RequestOutcome(ok=False, shed=True)] * 2
    )
    r = evaluate(outcomes, SLOTargets(availability=0.99, max_degraded_share=0.05))
    assert r.availability == pytest.approx(0.98)
    assert r.degraded_share == pytest.approx(8 / 98, rel=1e-3)
    assert set(r.violations) == {"availability", "degraded_share"}
    assert r.burn_rates["availability"] == pytest.approx(2.0)
