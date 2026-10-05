# path: book/capstone/northwind-assist/tests/test_reliability_cost.py
"""Degraded modes, admission control, spend guard, deadlines, and cost accounting per tenant."""
from __future__ import annotations

from aie_core.llm.errors import ProviderUnavailableError
from aie_core.llm.providers import FakeLLM
from conftest import auth, chat, make
from fastapi.testclient import TestClient

from northwind_assist.api.app import create_app
from northwind_assist.config import Settings
from northwind_assist.evaluation.suites import persona_ctx
from northwind_assist.orchestrator import ChatRequest

Q = "How do I reset my password?"


def test_open_breaker_serves_static_documents_without_a_model_call():
    c = make(Settings(environment="test", breaker_min_calls=2))
    breaker = c.models.breakers.get("llm:primary")
    for _ in range(3):
        breaker.record_failure(ProviderUnavailableError("provider down"))
    before = len(c.models.raw.requests)
    r = chat(c, "ana", Q)
    meta = next(e.data for e in r.events if e.event == "meta")
    assert meta["degrade_level"] == 3 and "all_models_open" in meta["degrade_reasons"]
    assert r.status == "degraded" and r.citations and r.cost_usd == 0.0
    assert len(c.models.raw.requests) == before                     # nothing was sent to the provider
    assert any(e.event == "notice" and e.data["kind"] == "degraded" for e in r.events)


def test_provider_failure_mid_request_falls_back_to_documents():
    def down(req):
        raise ProviderUnavailableError("503 from provider")

    c = make(llm=FakeLLM(handler=down))
    r = chat(c, "ana", Q)
    assert r.status == "degraded" and r.citations
    assert r.events[-1].event == "done"


def test_agents_disabled_in_minimal_mode_but_questions_work():
    c = make()
    c.resilience.degrade.forced_level = 2      # operator switch: MINIMAL
    r = chat(c, "ana", "create ticket: vpn down at store 0412")
    assert r.status == "degraded" and not c.tools.backends.tickets.created()
    assert chat(c, "ana", Q).citations


def test_admission_rejects_when_replica_is_full():
    c = make(Settings(environment="test", admission_capacity=1, admission_max_queue=0))
    held = c.resilience.admit("retail", 100, 8.0)                      # one request in flight
    assert held.admitted
    client = TestClient(create_app(c))
    r = client.post("/v1/chat", json={"message": Q}, headers=auth(c, "ana"))
    assert r.status_code == 503 and r.json()["error"] == "overloaded" and "Retry-After" in r.headers


def test_tenant_quota_is_429_and_does_not_affect_other_tenant():
    c = make(Settings(environment="test", tenant_requests_per_minute=4))   # bucket of one request
    client = TestClient(create_app(c))
    ok = client.post("/v1/chat?stream=false", json={"message": Q}, headers=auth(c, "ana"))
    limited = client.post("/v1/chat?stream=false", json={"message": Q}, headers=auth(c, "ana"))
    other = client.post("/v1/chat?stream=false", json={"message": Q}, headers=auth(c, "lee"))
    assert ok.status_code == 200 and limited.status_code == 429 and other.status_code == 200
    assert limited.json()["error"] == "tenant_quota"


def test_spend_guard_blocks_a_tenant_over_budget():
    c = make(Settings(environment="test", tenant_daily_budget_usd={"retail": 0.0000001, "logistics": 25.0}))
    client = TestClient(create_app(c))
    r = client.post("/v1/chat", json={"message": Q}, headers=auth(c, "ana"))
    assert r.status_code == 429 and r.json()["error"] == "tenant_budget_exhausted"
    assert client.post("/v1/chat?stream=false", json={"message": Q}, headers=auth(c, "lee")).status_code == 200
    assert any(a.kind == "blocked" and a.tenant == "retail" for a in c.ledger.alerts)


def test_request_that_cannot_meet_its_deadline_is_shed():
    c = make(Settings(environment="test", request_deadline_s=0.2))
    client = TestClient(create_app(c))
    r = client.post("/v1/chat", json={"message": Q}, headers=auth(c, "ana"))
    assert r.status_code == 503 and r.json()["error"] == "would_miss_deadline"


def test_cancelled_deadline_ends_stream_with_stage_error(container):
    p = container.orchestrator.prepare(persona_ctx("ana"), ChatRequest(message=Q))
    p.ctx.deadline.cancel("client_disconnected")
    r = container.orchestrator.run(p)
    err = next(e.data for e in r.events if e.event == "error")
    assert err["stage"] == "model" and r.status == "timeout"
    assert r.events[-1].event == "done"


def test_cost_is_accounted_per_tenant_and_reported_daily(client, container):
    chat(container, "ana", Q, session="a")
    chat(container, "ana", "What is the status of the vpn service?", session="a")
    chat(container, "lee", Q, session="b")
    assert client.get("/v1/cost/daily", headers=auth(container, "ana")).status_code == 403   # agents: no
    rep = client.get("/v1/cost/daily", headers=auth(container, "ops")).json()
    tenants = {t["tenant"]: t for t in rep["tenants"]}
    assert tenants["retail"]["requests"] == 2 and tenants["logistics"]["requests"] == 1
    assert tenants["retail"]["cost_usd"] > 0 and tenants["retail"]["cost_per_successful_answer_usd"] > 0
    assert set(tenants["retail"]["by_intent"]) == {"rag.answer", "agent.action"}
    assert abs(rep["total_cost_usd"] - sum(t["cost_usd"] for t in rep["tenants"])) < 1e-9


def test_threshold_alert_fires_once_when_spend_crosses_it():
    c = make(Settings(environment="test", tenant_daily_budget_usd={"retail": 0.02, "logistics": 25.0},
                      cost_alert_thresholds=[0.05]))
    chat(c, "ana", Q, session="x")
    chat(c, "ana", "How many unused PTO days can I carry over into next year?", session="x")
    alerts = [a for a in c.ledger.alerts if a.kind == "threshold" and a.tenant == "retail"]
    assert len(alerts) == 1


def test_backup_model_takes_over_and_plan_shrinks_while_primary_is_open():
    from northwind_assist.llm.demo import make_demo_llm

    def down(req):
        raise ProviderUnavailableError("primary down")

    backup = make_demo_llm()
    c = make(Settings(environment="test", breaker_min_calls=2), llm=FakeLLM(handler=down), backup_llm=backup)
    r = chat(c, "ana", "How many unused PTO days can I carry over into next year?", session="b1")
    assert r.status == "answered" and backup.requests                # the gateway fell back
    for _ in range(3):
        c.models.breakers.get("llm:primary").record_failure(ProviderUnavailableError("down"))
    r2 = chat(c, "ana", Q, session="b2")
    meta = next(e.data for e in r2.events if e.event == "meta")
    assert meta["degrade_level"] == 1 and "open:llm:primary" in meta["degrade_reasons"]   # reduced, not static
    assert r2.citations


def test_soft_budget_limit_reduces_the_plan_and_says_why():
    # worst-case reservation is about 0.0156 USD: above 80% of 0.018 (degrade), below 0.018 (no block)
    c = make(Settings(environment="test", tenant_daily_budget_usd={"retail": 0.018, "logistics": 25.0}))
    r = chat(c, "ana", Q)
    meta = next(e.data for e in r.events if e.event == "meta")
    assert meta["degrade_level"] == 1 and "spend:soft_limit" in meta["degrade_reasons"]
    assert not any(x.startswith("admission:") for x in meta["degrade_reasons"])
