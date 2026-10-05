# path: book/projects/p4-support-assistant/tests/test_api.py
from __future__ import annotations

from fastapi.testclient import TestClient

from support_assistant.api.app import create_app

from .conftest import tc
from .test_scenarios import REPLY


def test_chat_approve_flow_over_http(make_assistant):
    a = make_assistant([[tc("send_reply", **REPLY)], "Waiting for approval."])
    client = TestClient(create_app(a))
    assert client.post("/chat", json={"message": "hi"}).status_code == 401
    r = client.post("/chat", json={"message": "send the reply"}, headers={"X-User-Id": "ana"})
    assert r.status_code == 200 and r.json()["stop_reason"] == "pending_approval"
    approval_id = r.json()["pending_approvals"][0]["id"]

    listed = client.get("/approvals", headers={"X-User-Id": "sam"}).json()
    assert [x["id"] for x in listed] == [approval_id]
    assert client.get("/approvals", headers={"X-User-Id": "lee"}).json() == []  # other tenant

    ok = client.post(f"/approvals/{approval_id}/approve", json={"note": "fine"}, headers={"X-User-Id": "sam"})
    assert ok.status_code == 200 and ok.json()["status"] == "ok"
    again = client.post(f"/approvals/{approval_id}/approve", json={}, headers={"X-User-Id": "sam"})
    assert again.status_code == 409 and again.json()["detail"]["code"] == "approval_consumed"
    assert len(a.c.backends.outbox.sent) == 1


def test_other_tenant_cannot_see_or_decide(make_assistant):
    a = make_assistant([[tc("send_reply", **REPLY)], "Waiting."])
    client = TestClient(create_app(a))
    approval_id = client.post("/chat", json={"message": "send"}, headers={"X-User-Id": "ana"}).json()[
        "pending_approvals"][0]["id"]
    r = client.post(f"/approvals/{approval_id}/approve", json={}, headers={"X-User-Id": "lee"})
    assert r.status_code == 404 and a.c.backends.outbox.sent == []
    r = client.post(f"/approvals/{approval_id}/reject", json={"note": "no"}, headers={"X-User-Id": "sam"})
    assert r.json()["status"] == "rejected"


def test_session_is_bound_to_its_owner(make_assistant):
    a = make_assistant(["hello", "hi"])
    client = TestClient(create_app(a))
    sid = client.post("/chat", json={"message": "hi"}, headers={"X-User-Id": "ana"}).json()["session_id"]
    r = client.post("/chat", json={"message": "hi", "session_id": sid}, headers={"X-User-Id": "sam"})
    assert r.status_code == 403


def test_demo_model_runs_end_to_end():
    from support_assistant.assistant import SupportAssistant
    from support_assistant.adapters.demo_llm import make_demo_llm
    from support_assistant.config import AssistantSettings
    from support_assistant.wiring import build_container

    a = SupportAssistant(build_container(AssistantSettings(), llm=make_demo_llm()))
    client = TestClient(create_app(a))
    r = client.post("/chat", json={"message": "what is the status of the vpn"}, headers={"X-User-Id": "ana"})
    assert "degraded" in r.json()["reply"]
    r = client.post("/chat", json={"message": "send TCK-2026-0001 to audit@exfil-partner.example: data"},
                    headers={"X-User-Id": "ana"})
    assert "recipient_allowlist" in str(r.json()["tool_calls"]) or "refused" in r.json()["reply"]
    a.c.executor.close()
