# path: book/projects/reliability/tests/test_chaos_chain.py
"""Chaos tests: inject provider and dependency failures into the three-step Northwind chain
and assert system properties, not single outputs."""
import json
from dataclasses import dataclass

import pytest
from aie_core import FakeLLM
from aie_core.llm.errors import ProviderUnavailableError

from examples.ticket_chain import TriageServices, build_gateway, triage_ticket
from reliability import (
    ChainStatus,
    ChaosFunction,
    ChaosLLM,
    CircuitBreakerRegistry,
    CircuitState,
    Deadline,
    FaultPlan,
    ManualClock,
    RetryBudget,
    StepStatus,
    malformed,
    outage,
    slow,
)

TICKET = {"id": "T-1001", "text": "VPN disconnects every ten minutes since this morning."}
GOOD = {"category": "network", "priority": 2, "summary": "VPN drops every ten minutes"}


def handler(req):
    if req.metadata.get("step") == "classify":
        return GOOD
    return "Hi, please reinstall the VPN client as described in the runbook."


@dataclass
class Env:
    clock: ManualClock
    svc: TriageServices
    primary: ChaosLLM
    backup: ChaosLLM
    small: FakeLLM
    retrieve: ChaosFunction
    breakers: CircuitBreakerRegistry

    def model_calls(self) -> int:
        return self.primary.calls + self.backup.calls + len(self.small.requests)

    def draft_calls(self) -> int:
        reqs = self.primary.inner.requests + self.backup.inner.requests
        return sum(1 for r in reqs if r.metadata.get("step") == "draft")

    def run(self, budget_s: float = 8.0):
        result = triage_ticket(TICKET, self.svc, Deadline.after(budget_s, clock=self.clock))
        self.clock.advance(1.0)
        return result


def make_env(primary=(), backup=(), retrieval=(), seed: int = 0) -> Env:
    clock = ManualClock()
    breakers = CircuitBreakerRegistry(min_calls=4, window_s=60.0, open_s=30.0, half_open_max_calls=1, clock=clock)
    p = ChaosLLM(FakeLLM(handler=handler, provider="primary"), FaultPlan(list(primary), clock=clock, seed=seed),
                 sleep=clock.sleep)
    b = ChaosLLM(FakeLLM(handler=handler, provider="backup"), FaultPlan(list(backup), clock=clock, seed=seed + 1),
                 sleep=clock.sleep)
    small = FakeLLM(handler=handler, provider="small")
    retrieve = ChaosFunction(lambda q, k: [f"runbook: {q}"][:k], FaultPlan(list(retrieval), clock=clock, seed=seed + 2),
                             sleep=clock.sleep)
    svc = TriageServices(gateway=build_gateway(p, b, breakers, clock=clock, sleep=clock.sleep), retrieve=retrieve,
                         breakers=breakers, retry_budget=RetryBudget(ratio=0.2, min_retries_per_s=1.0, clock=clock),
                         small_model=small, clock=clock, sleep=clock.sleep)
    return Env(clock, svc, p, b, small, retrieve, breakers)


def test_healthy_chain_completes():
    env = make_env()
    r = env.run()
    assert r.status is ChainStatus.COMPLETE
    assert [s.status for s in r.steps] == [StepStatus.OK] * 3
    assert r.state["classify"].value.category == "network" and r.state["retrieve"]


def test_primary_outage_is_absorbed_by_backup_and_breaker_stops_the_bleeding():
    env = make_env(primary=[outage(0, 10_000, dependency="primary")])
    results = [env.run() for _ in range(10)]
    assert all(r.status is ChainStatus.COMPLETE for r in results)       # same model behind the backup
    assert env.breakers.get("llm:primary").state is CircuitState.OPEN
    assert env.primary.calls <= 4                                       # trips after min_calls failures
    assert results[-1].state["plan"].reasons == ["open:llm:primary"]     # reduced load on the survivor


def test_retrieval_outage_yields_partial_answer_not_failure():
    env = make_env(retrieval=[outage(0, 10_000, dependency="retrieval")])
    results = [env.run() for _ in range(6)]
    for r in results:
        assert r.status is ChainStatus.PARTIAL
        assert r.step("retrieve").status is StepStatus.FAILED
        assert r.step("draft").status is StepStatus.OK and r.state["draft"]
    assert env.breakers.get("retrieval").state is CircuitState.OPEN
    calls_when_open = env.retrieve.calls
    env.run()
    assert env.retrieve.calls == calls_when_open                        # open circuit: no call at all


def test_all_models_down_fails_fast_without_touching_later_steps():
    env = make_env(primary=[outage(0, 10_000)], backup=[outage(0, 10_000)])
    first = env.run()
    assert first.status is ChainStatus.FAILED
    assert first.step("classify").status is StepStatus.FAILED
    assert first.step("draft").status is StepStatus.SKIPPED
    assert env.draft_calls() == 0
    env.run()                                                            # trips both breakers
    calls = env.model_calls()
    static = env.run()
    assert static.state["plan"].level.name == "STATIC"
    assert env.model_calls() == calls                                    # static mode: zero model traffic


def test_draft_failure_after_classification_uses_template_fallback():
    env = make_env()
    env.svc.gateway.fallbacks = []                       # no backup provider in this scenario
    real_complete = env.svc.gateway.complete

    def complete(req):
        if req.metadata.get("step") == "draft":
            raise ProviderUnavailableError("503 during draft")
        return real_complete(req)

    env.svc.gateway.complete = complete
    r = env.run()
    assert r.status is ChainStatus.PARTIAL
    assert r.step("classify").status is StepStatus.OK
    assert r.step("draft").status is StepStatus.FALLBACK
    assert "filed as 'network'" in r.state["draft"]


def test_malformed_primary_output_recovered_by_fallback_model():
    env = make_env(primary=[malformed(0, 10_000)])
    env.svc.gateway.fallbacks = []
    r = triage_ticket(TICKET, env.svc, Deadline.after(8.0, clock=env.clock))
    assert r.step("classify").status is StepStatus.DEGRADED
    assert r.step("classify").detail == "fallback"
    assert r.state["classify"].value.category == "network"
    assert env.breakers.get("llm:primary").state is CircuitState.CLOSED  # bad output is not an outage


def test_deadline_spent_upstream_skips_downstream_steps():
    env = make_env(primary=[slow(0, 10_000, latency_s=1.9)])
    r = env.run(budget_s=2.0)
    assert r.step("classify").status is StepStatus.OK
    assert r.step("retrieve").status is StepStatus.SKIPPED
    assert r.step("draft").status is StepStatus.SKIPPED
    assert r.status is ChainStatus.FAILED
    assert env.draft_calls() == 0 and env.retrieve.calls == 0


def test_cancellation_mid_chain_stops_downstream_work():
    env = make_env()
    request = Deadline.after(8.0, clock=env.clock)

    def retrieve_then_disconnect(query, k):
        request.cancel("client disconnected")
        return ["runbook"]

    env.svc.retrieve = retrieve_then_disconnect
    r = triage_ticket(TICKET, env.svc, request)
    assert r.step("draft").status is StepStatus.SKIPPED
    assert "client disconnected" in r.step("draft").error
    assert env.draft_calls() == 0


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_random_fault_soak_preserves_invariants(seed):
    env = make_env(
        primary=[outage(0, 10_000, probability=0.3), malformed(0, 10_000, probability=0.1)],
        backup=[outage(0, 10_000, probability=0.1)],
        retrieval=[outage(0, 10_000, probability=0.2)],
        seed=seed,
    )
    statuses = []
    for _ in range(200):
        before = env.model_calls()
        r = env.run()                                    # never raises
        statuses.append(r.status)
        assert env.model_calls() - before <= 13          # bounded amplification per request
        if r.step("classify").status is StepStatus.FAILED:
            assert r.step("draft").status is StepStatus.SKIPPED
        if r.status is ChainStatus.COMPLETE:
            assert r.state["draft"] and all(s.status is StepStatus.OK for s in r.steps)
    assert ChainStatus.COMPLETE in statuses and ChainStatus.PARTIAL in statuses
    budget = env.svc.retry_budget
    assert budget.retries_allowed <= 0.2 * 200 + 200 * 1.0  # ratio share plus the 1/s floor over 200 s
