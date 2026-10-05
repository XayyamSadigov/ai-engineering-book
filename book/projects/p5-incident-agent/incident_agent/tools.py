# path: book/projects/p5-incident-agent/incident_agent/tools.py
"""The agent's tools, as agentkit FunctionTools.

Four read-only research tools are built per investigation, bound to the alert's time so the
model cannot ask about a different window. Every observation starts each fact with its
citation key, and returns the same keys in `data["sources"]` so the orchestrator can build
the evidence ledger without parsing text. Retrieved document text is wrapped as untrusted
data (Chapter 26): it informs the model but never instructs it.

The publish tool is EXTERNAL. agentkit's DefaultPolicy therefore requires approval for it,
and it reads the report body from the investigation store by id: the model never retypes
the report, so what the human approved is exactly what gets posted.
"""
from __future__ import annotations

from typing import Any, Callable

from agentkit import ErrorClass, FunctionTool, SideEffect, ToolContext, ToolOutput

from .adapters.channel import Channel
from .adapters.corpus import KnowledgeBase
from .adapters.telemetry import Telemetry
from .domain.models import Alert, Evidence, Investigation


def _obj(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


def _evidence_output(lines: list[str], items: list[Evidence], extra: dict[str, Any] | None = None) -> ToolOutput:
    return ToolOutput(content="\n".join(lines) or "no results",
                      data={"sources": [e.model_dump() for e in items], **(extra or {})})


def research_tools(kb: KnowledgeBase, telemetry: Telemetry, alert: Alert, *, k: int = 3) -> list[FunctionTool]:
    as_of = alert.fired_at

    def search(kind: str) -> Callable[..., ToolOutput]:
        def run(ctx: ToolContext, query: str) -> ToolOutput:
            hits = kb.search(kind, query, ctx.principal, k)         # ACL from the trusted principal
            lines, items = [], []
            for h in hits:
                lines.append(f"[{h.doc_id}] {h.title} > {h.section}\n<untrusted_data source=\"{h.doc_id}\">"
                             f"{h.text[:600]}</untrusted_data>")
                items.append(Evidence(id=h.doc_id, kind=kind, title=h.title, text=h.text[:600]))  # type: ignore[arg-type]
            return _evidence_output(lines, items)
        return run

    def query_service_metrics(service: str) -> ToolOutput:
        if not telemetry.known_service(service):
            return ToolOutput.failure(f"unknown service {service!r}; known: {sorted(telemetry.services)}",
                                      ErrorClass.VALIDATION)
        lines, items, anomalous = [], [], []
        for r in telemetry.read(service, as_of):
            text = r.describe(as_of)
            lines.append(f"[{r.source_id}] {text}")
            items.append(Evidence(id=r.source_id, kind="metric", title=r.name, text=text))
            if r.anomalous:
                anomalous.append(r.name)
        bad_deps = []
        for dep in telemetry.dependencies(service):
            dep_bad = [r.name for r in telemetry.read(dep, as_of) if r.anomalous]
            lines.append(f"dependency {dep}: " + (f"ANOMALOUS ({', '.join(dep_bad)}); query it for detail"
                                                  if dep_bad else "normal"))
            if dep_bad:
                bad_deps.append(dep)
        return _evidence_output(lines, items, {"service": service, "anomalous": anomalous,
                                               "anomalous_dependencies": bad_deps})

    def get_recent_deploys(service: str, hours: int = 24) -> ToolOutput:
        if not telemetry.known_service(service):
            return ToolOutput.failure(f"unknown service {service!r}", ErrorClass.VALIDATION)
        lines, items = [], []
        for d in telemetry.recent_deploys(service, as_of, hours):
            text = f"{d['at_dt']:%Y-%m-%d %H:%M} UTC {d['service']} {d['kind']}: {d['summary']} (by {d['author']})"
            lines.append(f"[deploy:{d['id']}] {text}")
            items.append(Evidence(id=f"deploy:{d['id']}", kind="deploy", title=d["id"], text=text))
        if not lines:
            lines.append(f"no deploys to {service} in the {hours} h before {as_of:%H:%M} UTC")
        return ToolOutput(content="\n".join(lines), data={"sources": [e.model_dump() for e in items]})

    q = _obj({"query": {"type": "string", "minLength": 3, "maxLength": 200}}, ["query"])
    svc = {"service": {"type": "string", "minLength": 2}}
    return [
        FunctionTool("search_runbooks", "Search Northwind operational runbooks (BM25). Returns [doc-id] excerpts.",
                     q, search("runbook"), pass_context=True),
        FunctionTool("search_incidents", "Search past incident reports and postmortems. Returns [doc-id] excerpts.",
                     q, search("incident"), pass_context=True),
        FunctionTool("query_service_metrics", "Metrics of one service at alert time, with anomaly flags and the "
                     "health of its dependencies.", _obj(svc, ["service"]), query_service_metrics),
        FunctionTool("get_recent_deploys", "Deploys and migrations to one service before the alert.",
                     _obj({**svc, "hours": {"type": "integer", "minimum": 1, "maximum": 72}}, ["service"]),
                     get_recent_deploys),
    ]


def publish_tool(channel: Channel, load: Callable[[str], Investigation | None]) -> FunctionTool:
    def post_report(ctx: ToolContext, investigation_id: str, channel_name: str) -> ToolOutput:
        inv = load(investigation_id)
        if inv is None or inv.report is None:
            return ToolOutput.failure(f"no report for investigation {investigation_id!r}", ErrorClass.IMPOSSIBLE)
        msg_id = channel.post(channel_name, f"Incident report {inv.alert.id}: {inv.alert.rule}", inv.report,
                              idempotency_key=ctx.idempotency_key)
        return ToolOutput(content=f"posted {msg_id} to {channel_name}", artifacts={"message_id": msg_id})

    return FunctionTool(
        "post_report", "Post the approved incident report to a team channel.",
        _obj({"investigation_id": {"type": "string"}, "channel_name": {"type": "string", "pattern": "^#"}},
             ["investigation_id", "channel_name"]),
        post_report, side_effect=SideEffect.EXTERNAL, requires_approval=True, idempotent=True, pass_context=True)


__all__ = ["research_tools", "publish_tool"]
