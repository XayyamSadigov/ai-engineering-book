# path: book/projects/examples/ch20/test_patterns.py
"""Offline tests for the nine patterns: the happy path and the failure mode each one is known for."""
from __future__ import annotations

from typing import Any

from aie_core.llm.providers import FakeLLM
from aie_core.llm.types import CompletionRequest
from agentkit import Budget, DefinitionOfDone, TerminationReason, citations_grounded

from compare import PLAN, TASK, demo_llm, delegate_all, investigator, run_all
from patterns import (
    AgentRouter, Branch, ChainContext, CriticCheck, EvaluatorOptimizer, Evaluation, Meter, PlannerExecutor, Rule,
    SpawnBudget, Specialist, Supervisor, Team, Worker, agent_step, code_step, fan_out, react, reflective_agent,
    role_of, run_chain, run_hierarchy,
)
from patterns.fixtures import tools
from patterns.script import Script, goal_of, one_tool_then_report, tc

ALL = ("get_service_status", "query_metrics", "search_incidents", "search_runbooks")


def roles(llm: FakeLLM) -> list[str]:
    return [role_of(r) for r in llm.requests]


# ----------------------------------------------------------------------------- comparison harness
def test_every_pattern_completes_on_the_demo_task():
    rows = run_all()
    assert [r.pattern for r, _ in rows] == ["react", "router", "planner_executor", "supervisor", "reflection",
                                           "evaluator_optimizer", "parallel", "sequential", "hierarchical"]
    assert all(r.ok for r, _ in rows), [(r.pattern, r.detail) for r, _ in rows if not r.ok]
    calls = {r.pattern: m["calls"] for r, m in rows}
    assert calls["react"] < calls["supervisor"] < calls["hierarchical"]   # structure has a price


def test_meter_counts_calls_per_role():
    meter = Meter(demo_llm())
    react(meter, TASK, tools(*ALL))
    assert meter.snapshot()["by_role"] == {"react": 5}
    assert meter.snapshot()["tokens"] > 0


# ----------------------------------------------------------------------------------------- ReAct
def test_react_gathers_evidence_and_cites_it():
    llm = FakeLLM(handler=Script(react=investigator))
    result = react(llm, TASK, tools(*ALL), dod=DefinitionOfDone(citations_grounded(3)))
    assert result.ok
    assert result.trajectory() == list(ALL)
    assert "[inc-2026-02-tracking-latency]" in result.answer


def test_react_wandering_is_stopped_by_the_harness():
    llm = FakeLLM(handler=Script(react=lambda r: tc("search_runbooks", query="vpn login")))
    result = react(llm, TASK, tools("search_runbooks"), budget=Budget(max_steps=10))
    assert not result.ok
    assert result.runs[0].stop_reason is TerminationReason.REPEATED_ACTION
    assert result.tool_calls == 2


# ---------------------------------------------------------------------------------------- Router
def _router(llm: FakeLLM, **kw: Any) -> AgentRouter:
    return AgentRouter(llm, [
        Specialist("incident", "service degradation", tools(*ALL), "Investigate."),
        Specialist("hr", "leave and expenses", tools("search_policies"), "Answer from policy."),
    ], rules=[Rule("hr", r"\bpto\b")], **kw)


def test_router_rule_short_circuits_the_model():
    llm = FakeLLM(handler=Script(specialist=one_tool_then_report))
    result = _router(llm).run("How many PTO days can I carry over?")
    assert result.data == {"route": "hr", "decided_by": "rule", "confidence": 1.0}
    assert "router" not in roles(llm)
    assert result.runs[0].trajectory() == ["search_policies"]


def test_router_low_confidence_goes_to_human_without_running_an_agent():
    llm = FakeLLM(handler=Script(router=lambda r: {"route": "incident", "confidence": 0.3}))
    result = _router(llm).run("Something feels off with the thing")
    assert not result.ok and result.runs == []
    assert result.data["decided_by"] == "fallback:low_confidence"


def test_router_rejects_labels_outside_the_closed_set():
    llm = FakeLLM(handler=Script(router=lambda r: {"route": "finance", "confidence": 0.99}))
    result = _router(llm).run("Approve my invoice")
    assert result.data["route"] == "human" and result.data["decided_by"] == "fallback:malformed"
    assert roles(llm).count("router") == 3          # first try plus two repairs, then fail closed


# ------------------------------------------------------------------------------ Planner-executor
def test_planner_executor_gives_each_step_only_its_tools():
    llm = FakeLLM(handler=Script(planner=lambda r: PLAN, executor=one_tool_then_report,
                                 synthesizer=lambda r: "Answer [status:trackline]"))
    result = PlannerExecutor(llm, tools(*ALL)).run(TASK)
    assert result.ok and result.data["replans"] == 0
    assert result.trajectory() == ["get_service_status", "query_metrics", "search_incidents"]
    exec_tools = [{t.name for t in r.tools or []} for r in llm.requests if role_of(r) == "executor"]
    assert all(len(ts) == 1 for ts in exec_tools)
    assert [r.run_id for r in result.runs] == ["pe.s1-1", "pe.s2-2", "pe.s3-3"]


def _replanning_planner(req: CompletionRequest) -> dict[str, Any]:
    if "Deviation detected" not in goal_of(req):
        return {"steps": [{"id": "s1", "objective": "Search runbooks for zzz quantum flux", "tools": ["search_runbooks"]}]}
    return {"steps": [{"id": "s1", "objective": "Search runbooks for zzz quantum flux", "tools": ["search_runbooks"]},
                      {"id": "s2", "objective": "Find past incidents about tracking latency", "tools": ["search_incidents"]}]}


def test_planner_executor_replans_on_empty_evidence():
    llm = FakeLLM(handler=Script(planner=_replanning_planner, executor=one_tool_then_report,
                                 synthesizer=lambda r: "done"))
    result = PlannerExecutor(llm, tools(*ALL)).run(TASK)
    assert result.data["replans"] == 1
    assert result.trajectory() == ["search_runbooks", "search_incidents"]
    assert roles(llm).count("planner") == 2
    assert "found no evidence" in llm.requests[[role_of(r) for r in llm.requests].index("planner", 1)].messages[-1].text


def test_planner_executor_stops_when_replan_budget_is_spent():
    stuck = {"steps": [{"id": "s1", "objective": "Search runbooks for zzz quantum flux", "tools": ["search_runbooks"]},
                       {"id": "s2", "objective": "Search runbooks for yyy warp core", "tools": ["search_runbooks"]}]}
    llm = FakeLLM(handler=Script(planner=lambda r: stuck, executor=one_tool_then_report))
    result = PlannerExecutor(llm, tools(*ALL), max_replans=1).run(TASK)
    assert not result.ok and "replan budget exhausted" in result.detail


def test_planner_executor_rejects_plans_with_unknown_tools():
    bad = {"steps": [{"id": "s1", "objective": "Restart the database now", "tools": ["restart_db"]}]}
    llm = FakeLLM(handler=Script(planner=lambda r: bad))
    result = PlannerExecutor(llm, tools(*ALL)).run(TASK)
    assert not result.ok and "unknown tools" in result.detail and result.runs == []


# ---------------------------------------------------------------------- Supervisor / hierarchical
def _workers() -> list[Worker]:
    return [Worker("incident_analyst", "health and metrics", tools("get_service_status", "query_metrics"), "Diagnose."),
            Worker("runbook_finder", "past incidents and runbooks", tools("search_incidents", "search_runbooks"), "Find.")]


def test_supervisor_delegates_with_a_ledger_and_derived_run_ids():
    llm = FakeLLM(handler=Script(supervisor=delegate_all, worker=investigator))
    result = Supervisor(llm, _workers()).run(TASK)
    assert result.ok
    assert [e["member"] for e in result.data["ledger"]] == ["incident_analyst", "runbook_finder"]
    assert [r.run_id for r in result.runs] == ["supervisor", "supervisor.incident_analyst-1",
                                              "supervisor.runbook_finder-1"]
    assert result.runs[1].events[0].metadata["parent_run_id"] == "supervisor"
    assert "[status:trackline]" in result.answer             # citations survive the hand-offs


def test_supervisor_spawn_budget_refuses_extra_agents():
    llm = FakeLLM(handler=Script(supervisor=delegate_all, worker=investigator))
    result = Supervisor(llm, _workers(), spawn=SpawnBudget(max_agents=1)).run(TASK)
    assert result.data["agents_spawned"] == 1
    denied = [o for o in result.runs[0].state.observations if not o.ok]
    assert denied and "agent budget of 1 exhausted" in denied[0].content


def test_supervisor_caps_delegations_per_worker():
    def nagging(req: CompletionRequest) -> Any:
        n = sum(1 for m in req.messages if m.role.value == "tool")
        return tc("delegate_incident_analyst", task=f"Check trackline health, attempt {n}")
    llm = FakeLLM(handler=Script(supervisor=nagging, worker=investigator))
    result = Supervisor(llm, _workers(), max_delegations_per_member=2).run(TASK)
    assert len(result.data["ledger"]) == 2
    assert not result.ok


def _team() -> Team:
    return Team("ops", "incident lead", [
        Team("diagnostics", "health and metrics", [
            Worker("status_checker", "health", tools("get_service_status"), "Check."),
            Worker("metrics_reader", "metrics", tools("query_metrics"), "Read."),
        ]),
        Worker("knowledge", "incidents and runbooks", tools("search_incidents", "search_runbooks"), "Find."),
    ])


def test_hierarchy_rolls_up_ledger_and_run_ids_encode_the_path():
    llm = FakeLLM(handler=Script(supervisor=delegate_all, worker=investigator))
    result = run_hierarchy(llm, _team(), TASK)
    assert result.ok and result.data["max_depth_reached"] == 2
    ids = {r.run_id for r in result.runs}
    assert "ops.diagnostics-1.status_checker-1" in ids
    assert result.data["agents_spawned"] == 4


def test_hierarchy_depth_limit_blocks_grandchildren():
    llm = FakeLLM(handler=Script(supervisor=delegate_all, worker=investigator))
    result = run_hierarchy(llm, _team(), TASK, spawn=SpawnBudget(max_agents=10, max_depth=1))
    assert result.data["max_depth_reached"] == 1
    assert not any(r.run_id.count(".") == 2 for r in result.runs)


# ------------------------------------------------------------------------------------ Reflection
def test_reflection_objective_checks_run_before_the_critic():
    calls: list[str] = []

    def critic(req: CompletionRequest) -> dict[str, Any]:
        calls.append("critic")
        return {"score": 5}

    def needs_runbook(answer: str, state: Any) -> list[str]:
        return [] if "it-incident-response-runbook" in answer or "Revised" in answer else ["no runbook cited"]

    llm = FakeLLM(handler=Script(reflective=investigator, critic=critic))
    check = CriticCheck(llm, "cause and fix", objective=[needs_runbook])
    result = reflective_agent(llm, TASK, tools("get_service_status", "search_incidents"), check)
    assert result.ok
    assert check.history[0] == {"kind": "objective", "problems": ["no runbook cited"]}
    assert calls == ["critic"]                    # the known-bad first draft cost no critic call


def test_reflection_is_bounded_by_max_revisions():
    llm = FakeLLM(handler=Script(reflective=investigator, critic=lambda r: {"score": 2, "fix": "more"}))
    result = reflective_agent(llm, TASK, tools("get_service_status"), CriticCheck(llm, "x"), max_revisions=1)
    assert result.runs[0].stop_reason is TerminationReason.VERIFICATION_FAILED
    assert len(result.data["rejected_drafts"]) == 2


# --------------------------------------------------------------------------- Evaluator-optimizer
def _gen(drafts: list[str]):
    def generate(previous, feedback, round_no):
        return drafts[min(round_no, len(drafts)) - 1], None
    return generate


def test_evaluator_optimizer_stops_on_plateau_and_keeps_the_best():
    scores = {"a": 0.5, "b": 0.6, "c": 0.4}
    loop = EvaluatorOptimizer(_gen(["a", "b", "c", "d"]), lambda c: Evaluation(passed=False, score=scores.get(c, 0)),
                              max_rounds=5, min_improvement=0.2)
    result = loop.run()
    assert not result.ok and result.detail.startswith("plateau after round 2")
    assert result.answer == "b"


def test_evaluator_optimizer_accepts_when_evaluator_passes():
    loop = EvaluatorOptimizer(_gen(["bad", "good"]), lambda c: Evaluation(passed=c == "good", score=0.9 if c == "good" else 0.1))
    result = loop.run()
    assert result.ok and result.answer == "good" and len(result.data["rounds"]) == 2


# -------------------------------------------------------------------------------------- Parallel
def _branches(fail_metrics: bool = False) -> list[Branch]:
    return [Branch("status", "Check trackline health", tools("get_service_status")),
            Branch("metrics", "Read pg-logi-prod scan metrics", tools("query_metrics"),
                   budget=Budget(max_steps=1) if fail_metrics else Budget(max_steps=4)),
            Branch("history", "Find past incidents about tracking latency", tools("search_incidents"))]


def test_parallel_require_all_refuses_partial_results():
    llm = FakeLLM(handler=Script(branch=one_tool_then_report, aggregator=lambda r: "merged"))
    result = fan_out(llm, TASK, _branches(fail_metrics=True), require="all")
    assert not result.ok and result.data["missing"] == ["metrics"]
    assert "aggregator" not in roles(llm)


def test_parallel_quorum_answers_and_names_the_gap():
    llm = FakeLLM(handler=Script(branch=one_tool_then_report, aggregator=lambda r: r.messages[0].text))
    result = fan_out(llm, TASK, _branches(fail_metrics=True), require="quorum")
    assert result.ok and "['metrics']" in result.answer
    assert [r.run_id for r in result.runs] == ["par.status", "par.metrics", "par.history"]


# ------------------------------------------------------------------------------------ Sequential
def test_sequential_gate_stops_the_chain_before_the_agent():
    llm = FakeLLM(handler=Script(investigate=investigator))
    ctx = ChainContext()
    result = run_chain([
        code_step("triage", lambda s: {**s, "category": "hr"},
                  gate=lambda s: None if s["category"] == "incident" else "not an incident"),
        agent_step(llm, ctx, "investigate", tools(*ALL), "Investigate.", lambda s: s["request"], "answer"),
    ], {"request": "How much PTO do I have?"}, ctx)
    assert not result.ok and result.data["path"] == ["triage"] and llm.requests == []


def test_sequential_agent_failure_is_a_step_failure():
    llm = FakeLLM(handler=Script(investigate=lambda r: tc("search_runbooks", query="vpn login")))
    ctx = ChainContext()
    result = run_chain([agent_step(llm, ctx, "investigate", tools("search_runbooks"), "Investigate.",
                                   lambda s: s["request"], "answer")], {"request": TASK}, ctx)
    assert not result.ok and "repeated_action" in result.detail and len(result.runs) == 1
