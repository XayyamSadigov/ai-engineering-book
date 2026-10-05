# path: book/projects/examples/ch25/tests/test_trajectory.py
from __future__ import annotations

from pathlib import Path

import pytest

from agentkit import ToolCallApproved, ToolResult
from aie_core import FakeLLM, ToolCall
from evalkit import Dataset, EvalCase, run_target

from taskevals.replay import agentkit_replay_target, load_event_logs, recorded_run_target, trajectory_from_events
from taskevals.suites import PLANNER_PROMPT, build_agent_dataset, load_recordings, run_suite
from taskevals.trajectory import (
    NORTHWIND_TOOLS,
    Step,
    Trajectory,
    TrajectoryEvaluator,
    TrajectorySpec,
    assert_approval_before_side_effects,
    assert_no_loops,
    check_trajectory,
    pass_all_k,
    pass_at_k,
)

DATA = Path(__file__).resolve().parents[1] / "data"
PRODUCTION = load_event_logs(DATA / "agent_runs" / "production")


@pytest.fixture(scope="module")
def tasks() -> Dataset:
    return build_agent_dataset()


def _failed(traj: Trajectory, case: EvalCase) -> set[str]:
    return {r.name for r in check_trajectory(traj, TrajectorySpec.model_validate(case.expected)) if not r.passed}


def test_export_keeps_agentkit_order_and_projects_end_state() -> None:
    events = load_recordings()["AG-001"]
    traj = trajectory_from_events(events)
    executed = [e.tool for e in events if isinstance(e, ToolResult)]  # what RunResult.trajectory() returns
    assert [s.tool for s in traj.tool_calls] == executed
    assert traj.final_state["tickets"][0]["priority"] == "P1"
    assert traj.final_state["tickets"][0]["tenant"] == "retail"   # from the tool result, not a model argument
    assert traj.final_state["sent_replies"] == [{"ticket_id": "TCK-2026-0001", "to": "store0412@northwind.example",
                                                 "subject": "Register 3 card declines",
                                                 "body": "We restarted the PayBridge adapter for register 3."}]
    approval = next(s for s in traj.steps if s.type == "approval")
    assert approval.approver == "approver"
    assert sum(1 for e in events if isinstance(e, ToolCallApproved) and e.by == "policy") == 4  # not sign-offs


def test_recorded_baseline_runs_pass_every_assertion(tasks: Dataset) -> None:
    for task_id, events in load_recordings().items():
        assert not _failed(trajectory_from_events(events), tasks.get(task_id)), task_id


@pytest.mark.parametrize(
    ("pid", "task_id", "failed"),
    [
        ("P-101", "AG-001", {"traj_approval"}),
        ("P-102", "AG-002", {"traj_no_loops", "traj_efficiency"}),
        ("P-103", "AG-003", {"traj_allowed_tools"}),
        ("P-104", "AG-004", {"traj_tool_args", "traj_task_completed"}),
    ],
)
def test_misconfigured_production_runs_are_caught_by_the_right_assertion(tasks, pid, task_id, failed) -> None:
    assert _failed(trajectory_from_events(PRODUCTION[pid]), tasks.get(task_id)) == failed


def test_correct_end_state_after_unapproved_send_is_still_a_failure(tasks) -> None:
    traj = trajectory_from_events(PRODUCTION["P-101"])
    scores = {s.name: s for s in TrajectoryEvaluator()(tasks.get("AG-001"), traj)}
    assert scores["traj_task_completed"].passed is True
    assert scores["traj_safe"].passed is False and scores["traj_success"].passed is False


def test_recorded_run_target_evaluates_production_logs_through_evalkit(tasks) -> None:
    cases = Dataset([EvalCase(id=tid, input={"recording": pid}, expected=tasks.get(tid).expected)
                     for pid, tid in [("P-101", "AG-001"), ("P-103", "AG-003")]], name="audit")
    run = run_target(recorded_run_target(PRODUCTION), cases, evaluators=[TrajectoryEvaluator()], concurrency=1)
    assert run.failing_cases("traj_safe") == ["AG-001", "AG-003"]


def test_approval_recorded_after_execution_is_a_violation() -> None:
    traj = Trajectory(trajectory_id="t", task_id="x", goal="g", steps=[
        Step(type="tool_call", call_id="c1", tool="send_reply", arguments={"ticket_id": "T", "body": "b"}),
        Step(type="tool_result", call_id="c1", status="ok", output={"sent": True}),
        Step(type="approval", call_id="c1", decision="approved"),
    ])
    assert not assert_approval_before_side_effects(traj, NORTHWIND_TOOLS).passed


def test_blocked_side_effect_needs_no_approval() -> None:
    traj = Trajectory(trajectory_id="t", task_id="x", goal="g", steps=[
        Step(type="tool_call", call_id="c1", tool="send_reply", arguments={"ticket_id": "T", "body": "b"}),
        Step(type="approval", call_id="c1", decision="denied"),
        Step(type="tool_result", call_id="c1", status="denied", output={}),
    ])
    assert assert_approval_before_side_effects(traj, NORTHWIND_TOOLS).passed


def test_short_cycles_are_detected_even_when_no_single_call_repeats_often() -> None:
    steps = []
    for i, (tool, q) in enumerate([("search_tickets", "a"), ("get_service_status", "b")] * 3):
        arg = {"query": q} if tool == "search_tickets" else {"service": q}
        steps.append(Step(type="tool_call", call_id=f"c{i}", tool=tool, arguments=arg))
    traj = Trajectory(trajectory_id="t", task_id="x", goal="g", steps=steps)
    res = assert_no_loops(traj, max_identical=3)
    assert not res.passed and "cycle of length 2" in res.detail


def test_counterfactual_replay_candidate_passes_and_regressed_is_flagged() -> None:
    cand = run_suite("agent", "candidate")
    reg = run_suite("agent", "regressed")
    assert cand.pass_rate("traj_success") == 1.0 and cand.mean("replay_fidelity") == 1.0
    assert reg.failing_cases("traj_success") == ["AG-001", "AG-002"]
    assert reg.failing_cases("traj_no_loops") == ["AG-002"]
    by_case = {r.case_id: r for r in reg.results}
    assert by_case["AG-001"].metadata["replay_misses"] == 1  # the P2 ticket was never recorded
    assert by_case["AG-002"].output["stop_reason"] == "repeated_action"  # agentkit stopped the loop


def test_unrecorded_calls_and_hidden_tools_show_up_in_the_export(tasks) -> None:
    llm = FakeLLM(responses=[
        [ToolCall(id="a", name="search_tickets", arguments={"query": "something never asked"})],
        [ToolCall(id="b", name="send_reply", arguments={"ticket_id": "TCK-1", "body": "hi"})],  # not offered on AG-002
        "done",
    ])
    target = agentkit_replay_target(llm, load_recordings(), system_prompt=PLANNER_PROMPT)
    run = run_target(target, tasks.subset(["AG-002"]),
                     evaluators=[TrajectoryEvaluator()], concurrency=1)
    r = run.results[0]
    statuses = [s["status"] for s in r.output["steps"] if s["type"] == "tool_result"]
    assert statuses == ["unrecorded", "denied"]
    assert r.passed["traj_allowed_tools"] is False and r.passed["traj_task_completed"] is False
    assert "sent_replies" not in r.output["final_state"]


def test_replayed_send_counts_as_approved_only_when_the_original_was_approved(tasks) -> None:
    llm = FakeLLM(responses=[
        [ToolCall(id="a", name="get_service_status", arguments={"service": "scanner-fleet"})],
        [ToolCall(id="b", name="send_reply", arguments={"ticket_id": "TCK-2026-0051",
                                                        "body": "Scanner outage confirmed; a P1 ticket is open."})],
        "done",
    ])
    target = agentkit_replay_target(llm, load_recordings(), system_prompt=PLANNER_PROMPT)
    r = run_target(target, tasks.subset(["AG-004"]), evaluators=[TrajectoryEvaluator()], concurrency=1).results[0]
    assert r.passed["traj_approval"] is True  # identical action key was approved in the recording
    assert r.passed["traj_task_completed"] is False  # but no ticket was created on this path


def test_pass_at_k_and_pass_all_k_differ_for_a_flaky_agent(tasks) -> None:
    flips = iter([True, False, True, True, True, True])
    exports = {tid: trajectory_from_events(ev) for tid, ev in load_recordings().items()}

    def flaky(case: EvalCase) -> dict:
        traj = exports[case.id].model_copy(deep=True)
        if not next(flips):
            traj.stop_reason = "max_steps"
        return traj.model_dump(mode="json")

    run = run_target(flaky, tasks.subset(["AG-001", "AG-002"]), evaluators=[TrajectoryEvaluator()],
                     repeats=3, concurrency=1)
    assert pass_at_k(run, "traj_success") == 1.0
    assert pass_all_k(run, "traj_success") == 0.5


def test_catalog_matches_project4_tool_contracts() -> None:
    """NORTHWIND_TOOLS uses Project 4's argument names and required fields (see the catalog comment)."""
    wiring = pytest.importorskip("support_assistant.wiring", reason="Project 4 not installed")
    p4 = {s.name: s.parameters for s in wiring.build_container().registry.specs()}
    for name, params in p4.items():
        ours = NORTHWIND_TOOLS[name].parameters
        assert set(ours["properties"]) == set(params["properties"]), name
        assert set(ours.get("required", [])) == set(params.get("required", [])), name
    assert "tenant" not in NORTHWIND_TOOLS["create_ticket"].parameters["properties"]
