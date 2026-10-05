# path: book/projects/p5-incident-agent/tests/test_api_cli.py
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from incident_agent.api import create_app, get_service
from incident_agent.cli import main

from .conftest import MAIN


@pytest.fixture
def client(service):
    app = create_app()
    app.dependency_overrides[get_service] = lambda: service
    return TestClient(app)


def test_http_flow_investigate_then_approve(client, service):
    r = client.post("/investigations", json={"alert_id": MAIN}, headers={"X-User": "oncall-logistics"})
    assert r.status_code == 201 and r.json()["status"] == "awaiting_approval"
    inv_id = r.json()["id"]
    assert client.get(f"/investigations/{inv_id}", headers={"X-User": "ic-logistics"}).status_code == 200
    d = client.post(f"/investigations/{inv_id}/decision", json={"approve": True, "reason": "ok"},
                    headers={"X-User": "ic-logistics"})
    assert d.status_code == 200 and d.json()["status"] == "published"
    again = client.post(f"/investigations/{inv_id}/decision", json={"approve": True},
                        headers={"X-User": "ic-logistics"})
    assert again.status_code == 409


def test_http_errors_do_not_leak(client):
    assert client.post("/investigations", json={"alert_id": MAIN}, headers={"X-User": "mallory"}).status_code == 403
    assert client.post("/investigations", json={"alert_id": "ALR-NOPE"},
                       headers={"X-User": "oncall-logistics"}).status_code == 404
    r = client.post("/investigations", json={"alert_id": MAIN}, headers={"X-User": "oncall-logistics"})
    other_tenant = client.get(f"/investigations/{r.json()['id']}", headers={"X-User": "oncall-retail"})
    assert other_tenant.status_code == 404
    assert client.post("/investigations", json={"alert_id": MAIN}).status_code == 422     # no identity header


def test_cli_investigate_show_approve_replay(service):
    out = io.StringIO()
    assert main(["investigate", MAIN], service=service, out=out) == 0
    inv_id = out.getvalue().split()[0]
    assert "replans: 1" in out.getvalue() and "## Recommended runbook" in out.getvalue()
    out = io.StringIO()
    assert main(["approve", inv_id, "--reason", "ok"], service=service, out=out) == 0
    assert "status=published" in out.getvalue()
    out = io.StringIO()
    assert main(["replay", inv_id], service=service, out=out) == 0
    assert out.getvalue().count("identical") == 6
    out = io.StringIO()
    assert main(["approve", inv_id], service=service, out=out) == 3 and "refused" in out.getvalue()
    assert main(["show", "inv-missing"], service=service, out=io.StringIO()) == 2
