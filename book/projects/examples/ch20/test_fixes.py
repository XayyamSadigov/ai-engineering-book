"""Edge cases behind the patterns' guarantees: plan ordering, shared limits, empty or broken parts."""
from __future__ import annotations

from aie_core.llm.providers import FakeLLM


from compare import TASK, delegate_all, investigator
from patterns import Supervisor, Team, Worker, code_step, fan_out, run_chain
from patterns.evaluator_optimizer import checks_then_judge
from patterns.fixtures import tools
from patterns.hierarchical import build_tree
from patterns.planner_executor import Plan, PlanStep, validate_plan
from patterns.script import Script

TOOLS = {"query_metrics"}


def _plan(*deps: tuple[str, list[str]]) -> Plan:
    return Plan(steps=[PlanStep(id=i, objective="check one thing", tools=["query_metrics"], depends_on=d)
                       for i, d in deps])


def test_plan_dependencies_must_run_earlier():
    assert validate_plan(_plan(("s1", []), ("s2", ["s1"])), TOOLS) == []
    assert validate_plan(_plan(("s1", ["s2"]), ("s2", ["s1"])), TOOLS)   # cycle
    assert validate_plan(_plan(("s3", ["s3"])), TOOLS)                   # self-dependency


def test_a_replan_may_depend_on_completed_steps():
    plan = _plan(("s2", ["s1"]))
    assert validate_plan(plan, TOOLS)                       # s1 unknown without context
    assert validate_plan(plan, TOOLS, completed={"s1"}) == []


def test_limits_and_store_are_shared_by_the_whole_tree():
    llm = FakeLLM(responses=[])
    leaf = Team("leaf", "d", [Worker("w", "d", tools("query_metrics"), "Read.")])
    root = build_tree(llm, Team("root", "d", [Team("mid", "d", [leaf])]))
    mid = root.members["mid"]
    assert mid.members["leaf"].store is root.store and mid.members["leaf"].spawn is root.spawn


def test_a_reused_supervisor_starts_each_run_fresh():
    workers = [Worker("incident_analyst", "health", tools("get_service_status", "query_metrics"), "Diagnose."),
               Worker("runbook_finder", "runbooks", tools("search_incidents", "search_runbooks"), "Find.")]
    sup = Supervisor(FakeLLM(handler=Script(supervisor=delegate_all, worker=investigator)), workers)
    first = sup.run(TASK, run_id="r1")
    second = sup.run(TASK, run_id="r2")
    assert first.ok and second.ok
    assert second.data["agents_spawned"] == 2 and len(second.data["ledger"]) == 2


def test_a_broken_judge_fails_the_round_instead_of_crashing():
    evaluate = checks_then_judge([], FakeLLM(handler=lambda req: "not json"), "be good")
    result = evaluate("a candidate")
    assert not result.passed and "judge unavailable" in result.feedback[0]


def test_a_chain_without_an_answer_fails():
    result = run_chain([code_step("noop", lambda s: s)], {"x": 1})
    assert not result.ok and "no 'answer'" in result.detail


def test_zero_branches_is_not_success():
    assert not fan_out(FakeLLM(responses=[]), TASK, [], require="all").ok
