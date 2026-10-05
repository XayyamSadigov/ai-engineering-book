# path: book/projects/p5-incident-agent/tests/test_replay.py
"""Replay tests: re-run recorded work without touching the outside world."""
from __future__ import annotations

import incident_agent.agent as agent_module
from aie_core.llm.providers import FakeLLM
from aie_core.llm.types import ToolCall
from agentkit import DefinitionOfDone, TerminationReason, citations_grounded, replay
from incident_agent.adapters.cassette import RecordingLLM, ReplayLLM
from incident_agent.adapters.scripted import scripted_llm
from incident_agent.domain.models import Status

from .conftest import MAIN


def test_every_step_replays_identically_without_executing_tools(service, telemetry):
    inv = service.investigate(MAIN, "oncall-logistics")
    before = telemetry.queries
    for step in inv.steps:
        report = replay(service.event_store.load(step.run_id))
        assert report.identical, (step.run_id, report.summary())
    assert telemetry.queries == before


def test_counterfactual_replay_reports_divergence_and_misses(service):
    inv = service.investigate(MAIN, "oncall-logistics")
    search = next(s for s in inv.steps if s.step.tool == "search_incidents")
    new_planner = FakeLLM(responses=[[ToolCall(id="x", name="search_incidents", arguments={"query": "dns outage"})],
                                     "no evidence: nothing relevant"])
    report = replay(service.event_store.load(search.run_id), new_planner)
    assert report.first_divergence == 0
    assert report.misses and report.misses[0]["arguments"] == {"query": "dns outage"}


def test_a_stricter_verifier_would_have_rejected_a_recorded_answer(service):
    inv = service.investigate(MAIN, "oncall-logistics")
    step = next(s for s in inv.steps if s.step.tool == "search_incidents")
    report = replay(service.event_store.load(step.run_id), dod=DefinitionOfDone(citations_grounded(3)))
    assert report.original_stop is TerminationReason.COMPLETED
    assert report.result.stop_reason is not TerminationReason.COMPLETED


def test_cassette_replays_the_whole_investigation_offline(make_service, tmp_path):
    tape = tmp_path / "tape.jsonl"
    recorded = make_service(RecordingLLM(scripted_llm(), tape)).investigate(MAIN, "oncall-logistics", inv_id="inv-a")
    replayer = ReplayLLM(tape)
    replayed = make_service(replayer).investigate(MAIN, "oncall-logistics", inv_id="inv-b")
    assert replayer.misses == []
    assert replayed.trajectory() == recorded.trajectory()
    assert replayed.report == recorded.report and replayed.status is Status.AWAITING_APPROVAL


def test_cassette_detects_a_prompt_change(make_service, tmp_path, monkeypatch):
    tape = tmp_path / "tape.jsonl"
    make_service(RecordingLLM(scripted_llm(), tape)).investigate(MAIN, "oncall-logistics")
    monkeypatch.setattr(agent_module, "WRITER_SYSTEM", agent_module.WRITER_SYSTEM + " Be brief.")
    replayer = ReplayLLM(tape)
    inv = make_service(replayer).investigate(MAIN, "oncall-logistics")
    assert inv.status is Status.FAILED and "CassetteMiss" in inv.detail
    assert len(replayer.misses) == 1                 # planning and execution replayed; the writer call is new
