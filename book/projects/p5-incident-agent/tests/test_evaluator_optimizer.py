# path: book/projects/p5-incident-agent/tests/test_evaluator_optimizer.py
from __future__ import annotations

import pytest

from aie_core.llm.types import CompletionRequest
from incident_agent.adapters.scripted import scripted_llm, writer as scripted_writer
from incident_agent.domain.models import Status
from incident_agent.service import InvalidState

from .conftest import MAIN


def test_first_draft_fails_the_dod_and_the_revision_passes(make_service):
    llm = scripted_llm()
    inv = make_service(llm).investigate(MAIN, "oncall-logistics")
    first, second = inv.rounds
    assert {p.code for p in first.dod_problems} >= {"uncited_claim", "runbook_not_in_catalog", "runbook_missing"}
    assert first.judge is None                       # the judge is not paid to read a report the rules reject
    assert second.accepted and second.judge["score"] == 5
    assert inv.usage["by_role"]["judge"] == 1
    assert "[it-incident-response-runbook]" in inv.report
    revision_prompt = [r for r in llm.requests if "Reviewer feedback:" in r.messages[-1].text][0].messages[-1].text
    assert "it-db-index-rebuild-runbook] does not exist" in revision_prompt


def _always_first_draft(req: CompletionRequest) -> str:
    stripped = req.model_copy(update={"messages": [req.messages[0], req.messages[1].model_copy(
        update={"content": req.messages[1].text.split("\n\nPrevious draft:")[0]})]})
    return scripted_writer(stripped)


def test_a_writer_that_does_not_improve_stops_on_plateau_and_cannot_be_published(make_service):
    service = make_service(scripted_llm(writer=_always_first_draft))
    inv = service.investigate(MAIN, "oncall-logistics")
    assert inv.status is Status.NEEDS_REVISION
    assert inv.detail == "plateau after round 2" and len(inv.rounds) == 2
    assert inv.publish_run_id is None
    with pytest.raises(InvalidState):
        service.decide(inv.id, "ic-logistics", approve=True)


def test_judge_failure_is_advisory_once_the_dod_passes(make_service):
    unconvinced = lambda req: {"reasoning": "not convinced", "score": 3, "flagged": ["weak cause"]}  # noqa: E731
    inv = make_service(scripted_llm(judge=unconvinced), max_revisions=3).investigate(MAIN, "oncall-logistics")
    assert inv.status is Status.AWAITING_APPROVAL and inv.judge_passed is False
    assert all(not r.accepted for r in inv.rounds)
    assert inv.detail.startswith("plateau")


def test_a_separate_judge_model_shares_the_investigation_call_ceiling(settings, kb, telemetry):
    from agentkit import InMemoryEventStore
    from incident_agent.adapters.channel import InMemoryChannel
    from incident_agent.adapters.store import InMemoryInvestigationStore
    from incident_agent.service import IncidentService

    def make(**overrides):
        return IncidentService(settings.model_copy(update=overrides), llm=scripted_llm(), judge_llm=scripted_llm(),
                               kb=kb, telemetry=telemetry, channel=InMemoryChannel(),
                               store=InMemoryInvestigationStore(), event_store=InMemoryEventStore())

    inv = make().investigate(MAIN, "oncall-logistics")
    assert inv.usage["by_role"]["judge"] == 1 and inv.usage["model_calls"] == 17   # judge calls are counted
    capped = make(max_llm_calls=16).investigate(MAIN, "oncall-logistics")         # the judge is call 17
    assert capped.status is Status.FAILED and "budget of 16" in capped.detail
