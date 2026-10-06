# path: book/projects/agentkit/examples/northwind_incident.py
"""Northwind Assist incident research, end to end, offline.

The on-call engineer asks why Trackline tracking lookups are slow. The agent searches the
shared runbooks and incident reports (respecting document ACLs), checks service status and
database metrics, proposes a follow-up ticket (which pauses for human approval), and answers
with citations that the Definition of Done verifies against what the agent actually read.
The run is persisted as JSONL and then replayed without executing any tool.

    python examples/northwind_incident.py            # uses ./.agent-runs or $AGENTKIT_EVENT_DIR

The model is a scripted FakeLLM so the example is deterministic; swap in
`aie_core.make_llm_client()` to drive it with a real provider.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from aie_core.llm.providers import FakeLLM
from aie_core.llm.types import CompletionRequest, Role, ToolCall
from aie_core.observability import InMemoryTracer
from agentkit import (
    AgentRuntime, Budget, DefinitionOfDone, FunctionTool, JsonlEventStore, SideEffect, ToolContext, ToolOutput,
    citations_grounded, contains_all, replay, tool_was_called,
)

DOCS_DIR = Path(__file__).resolve().parents[2] / "shared-data" / "docs"
GOAL = ("Alert trackline-p95-latency fired for the logistics tenant. Tracking lookups are slow. "
        "Find the most likely cause, the fix from our runbooks or past incidents, and open a follow-up ticket.")


# ------------------------------------------------------------------------------ tools
def _load_docs() -> list[dict[str, Any]]:
    docs = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        front, _, body = raw.partition("\n---\n")
        meta = dict(re.findall(r"^(\w+):\s*(.+)$", front, re.MULTILINE))
        groups = re.findall(r'"([^"]+)"', meta.get("acl_groups", ""))
        docs.append({"id": meta.get("id", path.stem), "title": meta.get("title", "").strip('"'),
                     "tenant": meta.get("tenant", "shared"), "groups": groups, "body": body})
    return docs


DOCS = _load_docs()


def search_docs(ctx: ToolContext, query: str, limit: int = 3) -> ToolOutput:
    """Keyword search with ACL and tenant filtering from the trusted principal, never from the model."""
    groups = set(ctx.principal.get("groups", [])) | {"all"}
    tenant = ctx.principal.get("tenant")
    terms = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 2]
    scored = []
    for d in DOCS:
        if not groups & set(d["groups"]) or d["tenant"] not in (tenant, "shared"):
            continue
        text = (d["title"] + " " + d["body"]).lower()
        score = sum(text.count(t) for t in terms)
        if score:
            line = next((ln.strip() for ln in d["body"].splitlines() if any(t in ln.lower() for t in terms)), "")
            scored.append((score, f"[{d['id']}] {d['title']}: {line[:240]}"))
    hits = [s for _, s in sorted(scored, reverse=True)[:limit]]
    return ToolOutput(content="\n".join(hits) or "no results", data={"hits": len(hits)})


STATUS = {
    "trackline": {"state": "degraded", "p95_ms": 4200, "error_rate": 0.028, "since": "06:02 UTC"},
    "pg-logi-prod": {"state": "degraded", "cpu": 0.95, "replication_lag_s": 1.2},
}


def get_service_status(service: str) -> dict[str, Any]:
    if service not in STATUS:
        raise FileNotFoundError(f"no service named {service!r}; known: {sorted(STATUS)}")
    return {"service": service, **STATUS[service]}


def query_metrics(metric: str) -> dict[str, Any]:
    data = {"pg-logi-prod.seq_scans_per_s": [12, 15, 480, 950], "trackline.webhook_retries_per_s": [3, 4, 210, 640]}
    if metric not in data:
        raise FileNotFoundError(f"unknown metric {metric!r}")
    return {"metric": metric, "last_4_intervals": data[metric]}


TICKETS: list[dict[str, Any]] = []


def create_ticket(ctx: ToolContext, summary: str, priority: str) -> ToolOutput:
    # Idempotent by key: a resumed run that re-executes this call does not open a second ticket.
    existing = next((t for t in TICKETS if t["idempotency_key"] == ctx.idempotency_key), None)
    ticket = existing or {"id": f"TCK-2026-{9000 + len(TICKETS)}", "summary": summary, "priority": priority,
                          "idempotency_key": ctx.idempotency_key}
    if existing is None:
        TICKETS.append(ticket)
    return ToolOutput(content=f"created {ticket['id']}", artifacts={"ticket_id": ticket["id"]})


def obj(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


TOOLS = [
    FunctionTool("search_docs", "Search Northwind runbooks, product docs and incident reports.",
                 obj({"query": {"type": "string", "minLength": 3},
                      "limit": {"type": "integer", "minimum": 1, "maximum": 5}}, ["query"]),
                 search_docs, pass_context=True),
    FunctionTool("get_service_status", "Current health of a service or database cluster.",
                 obj({"service": {"type": "string"}}, ["service"]), get_service_status),
    FunctionTool("query_metrics", "Recent values of a named metric (read-only).",
                 obj({"metric": {"type": "string"}}, ["metric"]), query_metrics),
    FunctionTool("create_ticket", "Open a follow-up ticket for the owning team.",
                 obj({"summary": {"type": "string"}, "priority": {"type": "string", "enum": ["P1", "P2", "P3"]}},
                     ["summary", "priority"]),
                 create_ticket, side_effect=SideEffect.WRITE, requires_approval=True, idempotent=True,
                 pass_context=True),
]


# ------------------------------------------------------------------------- scripted model
def scripted_planner(req: CompletionRequest) -> str | list[ToolCall]:
    """Stands in for a model: decides from what is already in the transcript."""
    seen = " ".join(m.text for m in req.messages if m.role is Role.TOOL)
    n = sum(1 for m in req.messages if m.role is Role.ASSISTANT)

    def tc(name: str, **args: Any) -> list[ToolCall]:
        return [ToolCall(id=f"c{n}", name=name, arguments=args)]

    if "trackline" not in seen:
        return tc("get_service_status", service="trackline")
    if "seq_scans" not in seen:
        return tc("query_metrics", metric="pg-logi-prod.seq_scans_per_s")
    if "inc-2026-02-tracking-latency" not in seen:
        return tc("search_docs", query="trackline latency sequential scan missing index migration")
    if "created TCK" not in seen and "DENIED" not in seen:
        return tc("create_ticket", summary="Trackline p95 latency: check shipment_events composite index",
                  priority="P2")
    return ("Likely cause: event-history queries on pg-logi-prod fell back to sequential scans "
            "(seq scans rose from 12/s to 950/s while Trackline p95 is 4.2 s), the same pattern as "
            "INC-2026-0217, where a migration dropped the composite index on (tracking_id, occurred_at) "
            "[inc-2026-02-tracking-latency]. Fix: compare pg_indexes with the pre-migration snapshot, "
            "recreate the missing index concurrently, and pause webhook retry workers while it builds. "
            "Follow-up ticket opened.")


# ------------------------------------------------------------------------------- main
def main(event_dir: str | None = None, out=sys.stdout) -> dict[str, Any]:
    store = JsonlEventStore(event_dir or os.environ.get("AGENTKIT_EVENT_DIR", ".agent-runs"))
    tracer = InMemoryTracer()
    dod = DefinitionOfDone(
        tool_was_called("get_service_status", "search_docs"),
        citations_grounded(min_citations=1),
        contains_all("index"),
        description="Done means: status checked, evidence searched, every [doc-id] cited was actually read.",
    )
    runtime = AgentRuntime(
        FakeLLM(handler=scripted_planner), TOOLS, budget=Budget(max_steps=8, max_tool_calls=8),
        dod=dod, store=store, tracer=tracer,
        principal={"user": "oncall-logistics", "tenant": "logistics", "groups": ["it-oncall"]},
    )
    run_id = f"incident-{len(store.runs()) + 1:03d}"
    first = runtime.run(GOAL, run_id=run_id)
    print(f"[{run_id}] paused: {first.stop_reason.value if first.stop_reason else None}: {first.detail}", file=out)
    pending = first.state.pending_approval
    if pending is not None:
        print(f"  approval needed for {pending.tool}({json.dumps(pending.arguments)})", file=out)
    # Bind the decision to the call the reviewer saw; a stale decision for another call is refused.
    final = runtime.resume(run_id, approve=True, reason="on-call lead approved",
                           request_id=pending.request_id if pending else None)
    print(f"[{run_id}] {final.stop_reason.value if final.stop_reason else None} after "
          f"{final.state.usage.steps} steps, trajectory={final.trajectory()}", file=out)
    print(f"  answer: {final.final_answer}", file=out)

    report = replay(store.load(run_id))
    print(f"  replay: {report.summary()}", file=out)
    return {"run_id": run_id, "first": first, "final": final, "replay": report, "spans": tracer.spans,
            "tickets": list(TICKETS)}


if __name__ == "__main__":
    main()
