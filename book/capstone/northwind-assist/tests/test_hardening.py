# path: book/capstone/northwind-assist/tests/test_hardening.py
"""Seams found in review: permission edits that never reached the index, idempotency keys that
crossed users and tenants, cross-tenant admin calls, an extraction path that skipped the input
guard, slots leaked by early disconnects, and configuration that let a published secret run."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from northwind_assist.api.app import create_app
from northwind_assist.config import Settings
from northwind_assist.domain.context import RequestContext
from northwind_assist.evaluation import run_eval
from northwind_assist.evaluation.suites import persona_ctx
from northwind_assist.orchestrator import ChatRequest

from conftest import auth, chat, make, token_for

TICKET = "create ticket: card terminal frozen at store 0412"


def ctx(user: str, tenant: str, groups: list[str], roles: list[str]) -> RequestContext:
    return RequestContext(user_id=user, tenant=tenant, groups=frozenset({"all", *groups}), roles=frozenset(roles))


def write_doc(path: Path, groups: str) -> None:
    path.write_text(f"""---
id: zeta-bonus-plan
title: Zeta Bonus Plan
version: "1"
updated_at: 2026-01-01
owner: HR
tenant: retail
acl_groups: {groups}
tags: [bonus]
---

# Zeta Bonus Plan

## Payout

The zeta bonus payout for store staff is 4200 EUR per quarter, paid on the zeta schedule.
""", encoding="utf-8")


def test_permission_change_without_text_change_reaches_the_index(tmp_path: Path):
    doc = tmp_path / "zeta.md"
    write_doc(doc, '["support"]')
    c = make(docs_dirs=[str(tmp_path)])
    q = "What is the zeta bonus payout?"
    assert chat(c, "ana", q, session="x1").status == "answered"
    before = c.kb.index_version
    write_doc(doc, '["managers"]')                 # revoke support's access; the text is unchanged
    c.kb.upsert(doc)
    assert c.kb.index_version != before            # cached results for the old ACL are retired
    c.caches.clear()
    r = chat(c, "ana", q, session="x2")
    assert r.status != "answered" and not r.citations


def test_client_idempotency_key_is_scoped_by_tenant_and_user():
    c = make()
    chat(c, "ana", TICKET, session="a", idempotency_key="retry-7")
    bob = c.orchestrator.prepare(ctx("bob", "logistics", ["support"], ["agent"]),
                                 ChatRequest(message=TICKET, session_id="b"), idempotency_key="retry-7")
    r = c.orchestrator.run(bob)
    tickets = c.tools.backends.tickets.created()
    assert [t.tenant for t in tickets] == ["retail", "logistics"]       # bob's ticket was created
    assert tickets[1].id in r.answer and tickets[0].id not in r.answer  # and he saw his own id


def test_content_key_does_not_trust_a_client_chosen_session_id():
    c = make()
    chat(c, "ana", TICKET, session="shared")
    carl = c.orchestrator.prepare(ctx("carl", "retail", ["support"], ["agent"]),
                                  ChatRequest(message=TICKET, session_id="shared"))
    c.orchestrator.run(carl)
    assert len(c.tools.backends.tickets.created()) == 2


def test_tenant_admin_cannot_touch_another_tenant(container):
    from fastapi.testclient import TestClient
    client = TestClient(create_app(container))
    chat(container, "ana", "What is the PTO carry-over limit?", session="a")
    logistics_admin = {"Authorization": f"Bearer {token_for(container, 'ops', tenant='logistics', roles=['admin'])}"}
    r = client.delete("/v1/admin/documents/inc-2025-11-pos-outage", headers=logistics_admin)
    assert r.status_code == 404 and "inc-2025-11-pos-outage" in container.kb.documents
    rep = client.get("/v1/cost/daily", headers=logistics_admin).json()
    assert [t["tenant"] for t in rep["tenants"]] == []                 # retail spend is not visible
    assert client.post("/v1/admin/reindex", headers=logistics_admin).status_code == 403
    every = client.get("/v1/cost/daily", headers=auth(container, "ops")).json()   # the platform operator
    assert [t["tenant"] for t in every["tenants"]] == ["retail"]


def test_attached_document_is_guarded_before_extraction():
    c = make()
    seen: dict[str, str] = {}
    original = c.extraction.extract

    def spy(text, **kw):
        seen["text"] = text
        return original(text, **kw)

    c.extraction.extract = spy
    doc = "INVOICE 77\nBill to: priya.raman@northwind.example, IBAN PL61109010140000071219812874\nTotal: 120.00 EUR"
    c.orchestrator.run(c.orchestrator.prepare(persona_ctx("ana"), ChatRequest(message="extract this", document=doc)))
    assert "priya.raman@northwind.example" not in seen["text"] and "PL6110901014" not in seen["text"]


def test_disconnect_before_the_first_read_releases_admission_and_spend():
    c = make()
    app = create_app(c)
    headers = [(b"authorization", f"Bearer {token_for(c, 'ana')}".encode()), (b"content-type", b"application/json")]

    async def hang_up() -> None:
        body = json.dumps({"message": "What is the PTO carry-over limit?"}).encode()
        msgs = [{"type": "http.request", "body": body, "more_body": False}]

        async def receive():
            return msgs.pop(0) if msgs else {"type": "http.disconnect"}

        async def send(message):
            pass

        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
                 "method": "POST", "scheme": "http", "path": "/v1/chat", "raw_path": b"/v1/chat",
                 "query_string": b"stream=true", "root_path": "", "headers": headers, "client": ("t", 1),
                 "server": ("s", 80)}
        await app(scope, receive, send)

    for _ in range(3):
        asyncio.run(hang_up())
    deadline = time.time() + 10
    while c.resilience.admission.snapshot()["in_flight"] and time.time() < deadline:
        time.sleep(0.05)
    assert c.resilience.admission.snapshot()["in_flight"] == 0
    assert len(c.ledger.rows) == 3                   # every reservation was settled


@pytest.mark.parametrize("env", ["staging", "prod"])
def test_published_secret_and_login_stub_are_refused_outside_dev(env: str):
    with pytest.raises(ValueError):
        Settings(environment=env)
    with pytest.raises(ValueError):
        Settings(environment="staging", jwt_secret="a-real-secret-0123456789abcdef0123456789", dev_login=True)


def test_misspelled_suite_fails_the_gate(tmp_path: Path):
    assert run_eval.main(["--suites", "tools,secuirty", "--out", str(tmp_path)]) == 2
