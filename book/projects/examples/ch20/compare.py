# path: book/projects/examples/ch20/compare.py
"""Run all nine patterns on one Northwind task with a scripted model and print their cost shape.

    python compare.py

The numbers are model calls, agent runs, tool calls, and tokens counted by the scripted
FakeLLM. They show the *shape* of each pattern's cost (how many calls a structure implies),
not real prices or real quality. Swap `demo_llm()` for `aie_core.make_llm_client()` to measure
a real provider on the same harness.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aie_core.llm.providers import FakeLLM  # noqa: E402
from aie_core.llm.types import CompletionRequest  # noqa: E402
from agentkit import DefinitionOfDone, citations_grounded  # noqa: E402

from patterns import (  # noqa: E402
    AgentRouter, Branch, ChainContext, CriticCheck, EvaluatorOptimizer, Meter, PatternResult, PlannerExecutor,
    Rule, Specialist, Supervisor, Team, Worker, agent_generator, agent_step, checks_then_judge, fan_out, llm_step,
    react, reflective_agent, run_chain, run_hierarchy,
)
from patterns.fixtures import tools  # noqa: E402
from patterns.script import Script, citations, goal_of, harness_notes, one_tool_then_report, tc, tool_texts  # noqa: E402

TASK = ("Trackline tracking lookups are slow for the logistics tenant. "
        "Find the likely cause and the documented fix.")

STEPS = [("get_service_status", {"service": "trackline"}, "status:trackline"),
         ("query_metrics", {"metric": "pg-logi-prod.seq_scans_per_s"}, "metric:pg-logi-prod"),
         ("search_incidents", {"query": "index migration latency"}, "inc-2026-02"),
         ("search_runbooks", {"query": "incident stabilise"}, "it-incident-response-runbook")]


def investigator(req: CompletionRequest) -> Any:
    """Calls each available tool it has not used yet, in a sensible order, then answers with citations."""
    names = {t.name for t in req.tools or []}
    seen = " ".join(tool_texts(req))
    for name, args, marker in STEPS:
        if name in names and marker not in seen:
            return tc(name, **args)
    cited = " ".join(f"[{c}]" for c in dict.fromkeys(citations(seen)))
    revised = " Revised after review: the index must be rebuilt concurrently." if harness_notes(req) else ""
    return f"Likely cause: a dropped index on pg-logi-prod causing sequential scans; fix per past incident. {cited}{revised}"


def delegate_all(req: CompletionRequest) -> Any:
    """Supervisor behavior: delegate once to each member in order, then combine."""
    done = len(tool_texts(req))
    if req.tools and done < len(req.tools):
        return tc(req.tools[done].name, task=f"Sub-task for {req.tools[done].name}: {goal_of(req)[:160]}")
    return "Combined: " + " ".join(f"[{c}]" for c in dict.fromkeys(citations(" ".join(tool_texts(req)))))


def echo(req: CompletionRequest) -> str:
    text = req.messages[-1].text
    return "Answer: " + " ".join(f"[{c}]" for c in dict.fromkeys(citations(text)))


def sequence_of(*values: Any) -> Callable[[CompletionRequest], Any]:
    items = list(values)
    return lambda req: items.pop(0) if len(items) > 1 else items[0]


PLAN = {"steps": [
    {"id": "s1", "objective": "Check the current health of trackline", "tools": ["get_service_status"], "target": "trackline"},
    {"id": "s2", "objective": "Read database scan metrics for pg-logi-prod", "tools": ["query_metrics"], "target": "pg-logi-prod"},
    {"id": "s3", "objective": "Find past incidents about tracking latency after a migration", "tools": ["search_incidents"]},
]}


def demo_llm() -> FakeLLM:
    return FakeLLM(handler=Script(
        react=investigator, specialist=investigator, generator=investigator, reflective=investigator,
        worker=investigator, branch=one_tool_then_report, executor=one_tool_then_report, investigate=investigator,
        router=lambda r: {"route": "incident", "confidence": 0.9, "reason": "service degradation"},
        planner=lambda r: PLAN, synthesizer=echo, aggregator=echo, write=echo,
        triage=lambda r: {"category": "incident", "service": "trackline"},
        supervisor=delegate_all,
        critic=sequence_of({"score": 3, "problems": ["fix lacks how"], "fix": "say how to rebuild the index"},
                           {"score": 5}),
        judge=sequence_of({"score": 3, "feedback": ["name the runbook"]}, {"score": 5}),
    ))


def run_all(make_llm: Callable[[], Any] = demo_llm) -> list[tuple[PatternResult, dict[str, Any]]]:
    grounded = DefinitionOfDone(citations_grounded(2))
    out: list[tuple[PatternResult, dict[str, Any]]] = []

    def measure(fn: Callable[[Meter], PatternResult]) -> None:
        meter = Meter(make_llm())
        out.append((fn(meter), meter.snapshot()))

    all_tools = tools("get_service_status", "query_metrics", "search_incidents", "search_runbooks")
    measure(lambda m: react(m, TASK, all_tools, dod=grounded))
    measure(lambda m: AgentRouter(m, [
        Specialist("incident", "service degradation and outages", all_tools, "Investigate and cite sources.", dod=grounded),
        Specialist("hr", "leave, expenses, benefits", tools("search_policies"), "Answer from policies."),
    ], rules=[Rule("hr", r"\b(pto|expense|leave)\b")]).run(TASK))
    measure(lambda m: PlannerExecutor(m, all_tools).run(TASK))
    measure(lambda m: Supervisor(m, [
        Worker("incident_analyst", "checks service health and metrics", tools("get_service_status", "query_metrics"),
               "Diagnose with the tools; cite sources."),
        Worker("runbook_finder", "finds past incidents and runbooks", tools("search_incidents", "search_runbooks"),
               "Find documented fixes; cite sources."),
    ], dod=grounded).run(TASK))
    measure(lambda m: reflective_agent(m, TASK, all_tools, CriticCheck(m, "names cause and how to fix it")))
    measure(lambda m: EvaluatorOptimizer(
        agent_generator(m, TASK, all_tools, instructions="Investigate; cite facts as [source-id]."),
        checks_then_judge([lambda c: None if re.search(r"\[inc-", c) else "cite a past incident"], m,
                          "cause supported by evidence; fix is actionable")).run())
    measure(lambda m: fan_out(m, TASK, [
        Branch("status", "Check trackline health", tools("get_service_status")),
        Branch("metrics", "Read pg-logi-prod scan metrics", tools("query_metrics")),
        Branch("history", "Find past incidents about tracking latency", tools("search_incidents")),
    ]))

    def chain(m: Meter) -> PatternResult:
        ctx = ChainContext()
        return run_chain([
            llm_step(m, "triage", "Classify the request.", lambda s: s["request"], "triage",
                     schema=_Triage, gate=lambda s: None if s["triage"].category == "incident" else "not an incident"),
            agent_step(m, ctx, "investigate", all_tools, "Investigate; cite sources.", lambda s: s["request"],
                       "findings", dod=grounded),
            llm_step(m, "write", "Write a short summary from the findings.", lambda s: s["findings"], "answer"),
        ], {"request": TASK}, ctx)

    measure(chain)
    measure(lambda m: run_hierarchy(m, Team("ops", "incident response lead", [
        Team("diagnostics", "service health and metrics", [
            Worker("status_checker", "service health", tools("get_service_status"), "Check health; cite."),
            Worker("metrics_reader", "metrics", tools("query_metrics"), "Read metrics; cite."),
        ]),
        Worker("knowledge", "past incidents and runbooks", tools("search_incidents", "search_runbooks"),
               "Find documented fixes; cite."),
    ]), TASK))
    return out


from pydantic import BaseModel  # noqa: E402


class _Triage(BaseModel):
    category: str
    service: str


def main() -> None:
    rows = run_all()
    print(f"{'pattern':<20}{'ok':<5}{'model calls':>12}{'agent runs':>12}{'tool calls':>12}{'tokens':>9}")
    for result, meter in rows:
        print(f"{result.pattern:<20}{str(result.ok):<5}{meter['calls']:>12}{len(result.runs):>12}"
              f"{result.tool_calls:>12}{meter['tokens']:>9}")


if __name__ == "__main__":
    main()
