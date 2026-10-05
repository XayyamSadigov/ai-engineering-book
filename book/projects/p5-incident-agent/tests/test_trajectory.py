# path: book/projects/p5-incident-agent/tests/test_trajectory.py
"""Trajectory tests: assert the path the agent took, not only the text it produced."""
from __future__ import annotations

from typing import Any

import pytest

from aie_core.llm.types import CompletionRequest
from incident_agent.adapters.metering import role_of
from incident_agent.adapters.scripted import planner as scripted_planner
from incident_agent.adapters.scripted import scripted_llm
from incident_agent.domain.models import Status
from incident_agent.service import NotAllowed

from .conftest import MAIN, SECOND

EXPECTED_MAIN = [
    ("query_service_metrics", "trackline"),
    ("query_service_metrics", "pg-logi-prod"),       # inserted by the replan
    ("get_recent_deploys", "pg-logi-prod"),          # inserted by the replan
    ("get_recent_deploys", "trackline"),
    ("search_incidents", "trackline latency degradation"),
    ("search_runbooks", "production incident response stabilise"),
]


def test_main_alert_replans_once_to_examine_the_anomalous_dependency(service):
    inv = service.investigate(MAIN, "oncall-logistics")
    assert inv.status is Status.AWAITING_APPROVAL
    assert inv.trajectory() == EXPECTED_MAIN
    assert len(inv.plans) == 2 and len(inv.deviations) == 1
    assert inv.deviations[0].startswith("dependency pg-logi-prod of trackline is anomalous")
    assert [s.ok for s in inv.steps] == [True] * 6
    assert "deploy:CHG-2026-0907" in inv.evidence and "[deploy:CHG-2026-0907]" in inv.report


def test_each_step_agent_sees_exactly_one_tool_and_never_the_publish_tool(make_service):
    llm = scripted_llm()
    make_service(llm).investigate(MAIN, "oncall-logistics")
    executor_tools = [[t.name for t in r.tools or []] for r in llm.requests if role_of(r) == "executor"]
    assert executor_tools and all(len(names) == 1 for names in executor_tools)
    assert not any("post_report" in names for names in executor_tools)
    assert all(r.tools is None for r in llm.requests if role_of(r) in ("planner", "writer", "judge"))


def test_second_alert_takes_a_shorter_path_without_replanning(service):
    inv = service.investigate(SECOND, "oncall-logistics")
    assert inv.status is Status.AWAITING_APPROVAL
    assert [t for t, _ in inv.trajectory()] == ["query_service_metrics", "get_recent_deploys", "search_incidents",
                                                "search_runbooks"]
    assert len(inv.plans) == 1
    assert "[deploy:CHG-2026-0909]" in inv.report and "CHG-2026-0907" not in inv.report


def test_replan_budget_exhaustion_fails_closed(make_service):
    inv = make_service(max_replans=0).investigate(MAIN, "oncall-logistics")
    assert inv.status is Status.FAILED and inv.detail.startswith("replan budget exhausted")
    assert inv.report is None and inv.publish_run_id is None and len(inv.steps) == 1


def test_invalid_plan_is_repaired_once_with_concrete_errors(make_service):
    seen: list[str] = []

    def sloppy_planner(req: CompletionRequest) -> dict[str, Any]:
        seen.append(req.messages[-1].text)
        plan = scripted_planner(req)
        if len(seen) == 1:
            plan["steps"][0]["target"] = "trackline-api"          # not a known service
        return plan

    llm = scripted_llm(planner=sloppy_planner)
    inv = make_service(llm).investigate(MAIN, "oncall-logistics")
    assert inv.status is Status.AWAITING_APPROVAL
    assert "unknown service 'trackline-api'" in seen[1]


def test_a_plan_that_never_validates_fails_the_investigation(make_service):
    llm = scripted_llm(planner=lambda r: {"steps": [{"id": "s1", "tool": "search_incidents", "target": "latency",
                                                       "objective": "Find similar incidents"}]})
    inv = make_service(llm).investigate(MAIN, "oncall-logistics")
    assert inv.status is Status.FAILED and "search_runbooks" in inv.detail


def test_global_model_call_ceiling_stops_the_investigation(make_service):
    inv = make_service(max_llm_calls=5).investigate(MAIN, "oncall-logistics")
    assert inv.status is Status.FAILED
    assert inv.usage["model_calls"] == 5


def test_users_cannot_investigate_another_tenants_alerts(service):
    with pytest.raises(NotAllowed):
        service.investigate(MAIN, "oncall-retail")
    with pytest.raises(NotAllowed):
        service.investigate(MAIN, "mallory")
