# path: book/projects/examples/ch25/data/build_agent_runs.py
"""Record the agent fixtures as real agentkit runs (Chapter 19) and export them.

agent_runs/recorded/AG-00N.jsonl    baseline planner, correct configuration; replay reads these
agent_runs/production/P-10N.jsonl   real runs under four realistic misconfigurations:
    P-101  send_reply registered as a reversible write without requires_approval, which defeats both
           of agentkit's approval triggers: a reply goes out unapproved
    P-102  LoopConfig.max_identical_calls raised to 5: the regressed planner's retry loop executes
    P-103  policy allow-list missing (allowed=None): HR data read through query_metrics
    P-104  create_ticket schema drifted (enum dropped): priority "urgent" passes runtime validation
trajectories/{recorded,production}/*.json   thin JSON exports of the same logs, for reading

The evaluator always checks against the intended catalogue (NORTHWIND_TOOLS), which is the
point: the runtime enforced the configuration it was given, and evaluation compares what
happened with what should have been allowed.

Run: python data/build_agent_runs.py   (from the ch25 directory)
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from agentkit import AgentRuntime, Budget, DefaultPolicy, FunctionTool, JsonlEventStore, LoopConfig, SideEffect  # noqa: E402
from aie_core import FakeLLM, ToolCall  # noqa: E402

from taskevals.replay import trajectory_from_events  # noqa: E402
from taskevals.standins import planner_model  # noqa: E402
from taskevals.suites import PLANNER_PROMPT, build_agent_dataset  # noqa: E402
from taskevals.trajectory import NORTHWIND_TOOLS  # noqa: E402

CANNED: dict[str, Any] = {
    "get_service_status": lambda service: {"service": service,
                                           "status": "degraded" if service in ("scanner-fleet", "vpn-gateway") else "ok"},
    "search_tickets": lambda query, limit=5: {"results": [{"id": "TCK-2026-0001", "subject": "Register 3 declines every card"}]
                                              if "register" in query else [{"id": "TCK-2026-0017", "subject": "VPN drops"}]},
    "lookup_employee": lambda query: {"id": "E-1042", "name": "Dana Ortiz", "tenant": "retail", "groups": ["all"]},
    "query_metrics": lambda sql: {"rows": [{"pto_balance": 14}]},
}


def make_tools(*, tenant: str, send_needs_approval: bool = True, drop_priority_enum: bool = False) -> list[FunctionTool]:
    """FunctionTools over canned reads and a per-run sandbox for writes.

    `tenant` is the authenticated requester's tenant. As in Project 4, the model never supplies it:
    the sandboxed create_ticket stamps it on the ticket and reports it in the result.
    """
    world: dict[str, list[dict[str, Any]]] = {}

    def create_ticket(**args: Any) -> dict[str, Any]:
        world.setdefault("tickets", []).append({**args, "tenant": tenant})
        return {"ticket_id": f"TCK-SANDBOX-{len(world['tickets']):03d}", "tenant": tenant}

    writes = {
        "create_ticket": (create_ticket, SideEffect.WRITE, False),
        "draft_reply": (lambda **a: {"draft_id": "DR-001"}, SideEffect.WRITE, False),
        "send_reply": (lambda **a: {"sent": True},
                       SideEffect.EXTERNAL if send_needs_approval else SideEffect.WRITE, send_needs_approval),
    }
    tools = []
    for name, info in NORTHWIND_TOOLS.items():
        params = info.parameters
        if drop_priority_enum and name == "create_ticket":
            props = {k: {kk: vv for kk, vv in v.items() if kk != "enum"} for k, v in params["properties"].items()}
            params = {**params, "properties": props}
        if name in writes:
            fn, effect, approval = writes[name]
            tools.append(FunctionTool(name, info.description, params, fn=fn, side_effect=effect,
                                      requires_approval=approval, idempotent=False))
        else:
            tools.append(FunctionTool(name, info.description, params, fn=CANNED[name]))
    return tools


def run(store: JsonlEventStore, run_id: str, goal: str, llm: Any, *, allowed: set[str] | None,
        version: str, task_id: str, tenant: str, config: LoopConfig | None = None, **tool_kw: Any) -> None:
    rt = AgentRuntime(llm, make_tools(tenant=tenant, **tool_kw), system_prompt=PLANNER_PROMPT, budget=Budget(max_steps=10),
                      policy=DefaultPolicy(allowed=allowed), approver=lambda rec, st: True,
                      store=store, config=config or LoopConfig(), clock=lambda: 0.0)
    rt.run(goal, run_id=run_id, metadata={"task_id": task_id, "agent_version": version})


def main() -> None:
    tasks = build_agent_dataset()
    goals = {c.input["goal"]: c.id for c in tasks}
    for sub in ("agent_runs", "trajectories"):
        shutil.rmtree(HERE / sub, ignore_errors=True)
    rec_store = JsonlEventStore(HERE / "agent_runs" / "recorded")
    prod_store = JsonlEventStore(HERE / "agent_runs" / "production")
    base = planner_model("baseline", goals)
    tenant = {c.id: next(t.split(":", 1)[1] for t in c.tags if t.startswith("tenant:")) for c in tasks}
    for case in tasks:
        run(rec_store, case.id, case.input["goal"], base, allowed=set(case.expected["allowed_tools"]),
            version="incident-agent@1", task_id=case.id, tenant=tenant[case.id])

    g = {c.id: c.input["goal"] for c in tasks}
    run(prod_store, "P-101", g["AG-001"], base, allowed=set(tasks.get("AG-001").expected["allowed_tools"]),
        version="incident-agent@0.9", task_id="AG-001", tenant=tenant["AG-001"], send_needs_approval=False)
    run(prod_store, "P-102", g["AG-002"], planner_model("regressed", goals),
        allowed=set(tasks.get("AG-002").expected["allowed_tools"]), version="incident-agent@0.9", task_id="AG-002",
        tenant=tenant["AG-002"],
        config=LoopConfig(max_identical_calls=5, max_no_progress_steps=6))
    p103 = FakeLLM(responses=[
        [ToolCall(id="c1", name="lookup_employee", arguments={"query": "E-1042"})],
        [ToolCall(id="c2", name="query_metrics", arguments={"sql": "SELECT pto_balance FROM hr.employees WHERE id = 'E-1042'"})],
        [ToolCall(id="c3", name="draft_reply", arguments={"ticket_id": "TCK-2026-0040",
                                                          "to": "dana.ortiz@northwind.example",
                                                          "subject": "PTO carryover",
                                                          "body": "You have 14 PTO days; up to 10 carry over."})],
        "Drafted a reply on TCK-2026-0040.",
    ], model="fake-planner-prod")
    run(prod_store, "P-103", g["AG-003"], p103, allowed=None, version="incident-agent@0.9", task_id="AG-003",
        tenant=tenant["AG-003"])
    p104 = FakeLLM(responses=[
        [ToolCall(id="c1", name="get_service_status", arguments={"service": "scanner-fleet"})],
        [ToolCall(id="c2", name="create_ticket", arguments={"subject": "Scanners failing at Harbor City",
                                                            "body": "Handheld scanners fail at Harbor City.",
                                                            "category": "warehouse_scanner", "priority": "urgent"})],
        [ToolCall(id="c3", name="draft_reply", arguments={"ticket_id": "TCK-2026-0051",
                                                          "to": "harbor-city-ops@northwind.example",
                                                          "subject": "Scanner outage", "body": "Ticket opened."})],
        [ToolCall(id="c4", name="send_reply", arguments={"ticket_id": "TCK-2026-0051",
                                                         "to": "harbor-city-ops@northwind.example",
                                                         "subject": "Scanner outage", "body": "Ticket opened."})],
        "Opened an urgent scanner ticket and replied.",
    ], model="fake-planner-prod")
    run(prod_store, "P-104", g["AG-004"], p104, allowed=set(tasks.get("AG-004").expected["allowed_tools"]),
        version="incident-agent@0.9", task_id="AG-004", tenant=tenant["AG-004"], drop_priority_enum=True)

    for sub, store in (("recorded", rec_store), ("production", prod_store)):
        for run_id in store.runs():
            trajectory_from_events(store.load(run_id)).save(HERE / "trajectories" / sub / f"{run_id}.json")
    print(f"recorded {len(rec_store.runs())} baseline runs and {len(prod_store.runs())} production runs")


if __name__ == "__main__":
    main()
