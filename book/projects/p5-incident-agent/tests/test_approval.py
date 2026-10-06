# path: book/projects/p5-incident-agent/tests/test_approval.py
from __future__ import annotations

import pytest

from agentkit import JsonlEventStore, Note, TerminationReason, ToolResult
from incident_agent.adapters.channel import JsonlChannel
from incident_agent.adapters.scripted import scripted_llm
from incident_agent.adapters.store import FileInvestigationStore
from incident_agent.domain.models import Status
from incident_agent.service import IncidentService, InvalidState, NotAllowed

from .conftest import MAIN


def test_nothing_is_posted_until_a_human_approves(service):
    inv = service.investigate(MAIN, "oncall-logistics")
    assert service.channel.messages() == []
    events = service.event_store.load(inv.publish_run_id)
    assert events[-1].reason is TerminationReason.APPROVAL_REQUIRED
    assert any(isinstance(e, Note) and e.kind == "approval_required" for e in events)
    assert not any(isinstance(e, ToolResult) for e in events)


def test_approval_posts_exactly_the_reviewed_report_once(service):
    inv = service.investigate(MAIN, "oncall-logistics")
    done = service.decide(inv.id, "ic-logistics", approve=True, reason="matches dashboards")
    assert done.status is Status.PUBLISHED and done.posted_message_id == "MSG-00001"
    [msg] = service.channel.messages()
    assert msg["body"] == inv.report and msg["channel"] == "#incidents"
    with pytest.raises(InvalidState):
        service.decide(inv.id, "ic-logistics", approve=True)
    assert len(service.channel.messages()) == 1


def test_rejection_posts_nothing(service):
    inv = service.investigate(MAIN, "oncall-logistics")
    done = service.decide(inv.id, "ic-logistics", approve=False, reason="cause not confirmed yet")
    assert done.status is Status.REJECTED and service.channel.messages() == []


def test_only_oncall_staff_of_the_same_tenant_can_approve(service):
    inv = service.investigate(MAIN, "oncall-logistics")
    for user in ("analyst-logistics", "mallory"):
        with pytest.raises(NotAllowed):
            service.decide(inv.id, user, approve=True)
    with pytest.raises(KeyError):   # another tenant cannot even learn that the investigation exists
        service.decide(inv.id, "oncall-retail", approve=True)
    assert service.get(inv.id).status is Status.AWAITING_APPROVAL


def test_approval_from_a_different_process_resumes_from_durable_state(settings, kb, telemetry, tmp_path):
    def fresh() -> IncidentService:   # nothing shared but the directory
        return IncidentService(settings.model_copy(update={"state_dir": tmp_path}), llm=scripted_llm(), kb=kb,
                               telemetry=telemetry, channel=JsonlChannel(tmp_path / "channel.jsonl"),
                               store=FileInvestigationStore(tmp_path / "inv"),
                               event_store=JsonlEventStore(tmp_path / "events"))

    inv = fresh().investigate(MAIN, "oncall-logistics")
    done = fresh().decide(inv.id, "ic-logistics", approve=True, reason="ok")
    assert done.status is Status.PUBLISHED
    assert len(JsonlChannel(tmp_path / "channel.jsonl").messages()) == 1
    assert (tmp_path / "events" / f"{inv.id}.publish.jsonl").exists()
