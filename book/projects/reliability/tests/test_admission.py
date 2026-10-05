# path: book/projects/reliability/tests/test_admission.py
import pytest

from reliability import (
    Action,
    AdmissionConfig,
    AdmissionController,
    AdmissionRejected,
    AdmissionRequest,
    Priority,
    TenantQuota,
)


def ctrl(clock, **kw) -> AdmissionController:
    cfg = AdmissionConfig(capacity=4, max_queue=4, initial_service_s=4.0, **kw)
    return AdmissionController(cfg, clock=clock)


def interactive(tenant: str = "retail", deadline_s: float | None = 8.0) -> AdmissionRequest:
    return AdmissionRequest(tenant_id=tenant, priority=Priority.INTERACTIVE, est_tokens=3000, deadline_s=deadline_s)


def test_admits_under_light_load(clock):
    d = ctrl(clock).admit(interactive())
    assert d.action is Action.ADMIT and d.ticket is not None


def test_sheds_under_overload(clock):
    c = ctrl(clock)
    decisions = [c.admit(interactive(deadline_s=None)) for _ in range(12)]
    actions = [d.action for d in decisions]
    assert actions[:3] == [Action.ADMIT] * 3
    assert Action.DEGRADE in actions                 # degrade before rejecting
    assert actions[-1] is Action.REJECT
    assert decisions[-1].reason == "overloaded" and decisions[-1].http_status() == 503
    assert c.snapshot()["in_flight"] == 8            # capacity 4 + queue 4, nothing beyond


def test_rejects_when_estimated_wait_exceeds_deadline(clock):
    c = ctrl(clock)
    for _ in range(8):
        c.admit(interactive(deadline_s=None))
    # queue of 4 at 4 s service over 4 slots -> 4 s wait; a 3 s budget cannot be met
    d = c.admit(interactive(deadline_s=3.0))
    assert d.action is Action.REJECT and d.reason in {"would_miss_deadline", "overloaded"}


def test_batch_yields_to_interactive(clock):
    c = ctrl(clock, batch_max_utilization=0.5)
    batch = AdmissionRequest(tenant_id="logistics", priority=Priority.BATCH, est_tokens=1000)
    assert c.admit(batch).admitted
    c.admit(interactive())
    d = c.admit(batch)
    assert d.action is Action.DEFER and d.reason == "batch_yields_to_interactive"


def test_per_tenant_quota_is_isolated(clock):
    quota = TenantQuota(requests_per_minute=8, tokens_per_minute=10**9, burst_fraction=0.25)  # burst of 2
    c = AdmissionController(AdmissionConfig(capacity=100, default_quota=quota), clock=clock)
    results = []
    for _ in range(3):
        d = c.admit(interactive("retail"))
        results.append(d.action)
        c.release(d)
    assert results == [Action.ADMIT, Action.ADMIT, Action.REJECT]
    rejected = c.admit(interactive("retail"))
    assert rejected.reason == "tenant_quota" and rejected.http_status() == 429
    assert rejected.retry_after_s and rejected.retry_after_s > 0
    assert c.admit(interactive("logistics")).admitted   # the other tenant is unaffected
    clock.advance(rejected.retry_after_s)
    assert c.admit(interactive("retail")).admitted


def test_shed_requests_do_not_spend_quota(clock):
    quota = TenantQuota(requests_per_minute=4, tokens_per_minute=10**9, burst_fraction=1.0)
    c = AdmissionController(AdmissionConfig(capacity=1, max_queue=0, default_quota=quota), clock=clock)
    first = c.admit(interactive(deadline_s=None))
    for _ in range(5):
        assert c.admit(interactive(deadline_s=None)).reason == "overloaded"
    c.release(first)
    assert c.admit(interactive(deadline_s=None)).admitted


def test_guard_releases_and_learns_service_time(clock):
    c = ctrl(clock)
    with c.guard(interactive()) as d:
        assert d.admitted
        clock.advance(2.0)
    snap = c.snapshot()
    assert snap["in_flight"] == 0
    assert snap["service_s_ewma"] == pytest.approx(0.8 * 4.0 + 0.2 * 2.0)


def test_guard_raises_typed_rejection(clock):
    c = AdmissionController(AdmissionConfig(capacity=1, max_queue=0), clock=clock)
    c.admit(interactive(deadline_s=None))
    with pytest.raises(AdmissionRejected) as info:
        with c.guard(interactive(deadline_s=None)):
            pass
    assert info.value.reason == "overloaded" and info.value.retryable


def test_operator_floor_forces_degraded_admission(clock):
    c = ctrl(clock)
    c.min_degrade_level = 1
    d = c.admit(interactive())
    assert d.action is Action.DEGRADE and d.degrade_level == 1
