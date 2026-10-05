# path: book/projects/examples/ch20/patterns/fixtures.py
"""A tiny, self-contained Northwind world for the pattern examples: four tools over in-memory data.
Project 5 replaces these with real retrieval over shared-data; the patterns do not change."""
from __future__ import annotations

from typing import Any

from agentkit import FunctionTool

from .common import obj

RUNBOOKS = {
    "it-vpn-access-runbook": "VPN login loops: clear the cached NorthGate profile, then re-enroll the device certificate.",
    "it-database-failover-runbook": "Failover: confirm replica lag under 5 s, then run the Patroni switchover.",
    "it-incident-response-runbook": "Incidents: acknowledge the page, open the incident, stabilise first, update on cadence.",
    "it-password-reset-runbook": "Password reset: verify identity with the manager, then reset in Northwind ID.",
}
POLICIES = {
    "hr-pto-policy": "PTO carryover is capped at 10 days per year (policy v3.0, effective 2026-01-01).",
    "hr-expense-policy": "Meals are reimbursed up to the daily per-diem with itemised receipts.",
}
INCIDENTS = {
    "inc-2026-02-tracking-latency": "INC-2026-0217: a migration dropped the (tracking_id, occurred_at) index; "
                                    "fixed by CREATE INDEX CONCURRENTLY and pausing webhook retries.",
}
STATUS = {
    "trackline": "degraded: p95 4.2 s, error rate 2.8% since 06:02 UTC",
    "pg-logi-prod": "degraded: CPU 95%, sequential scans 950/s",
    "northgate-vpn": "healthy",
}
METRICS = {
    "pg-logi-prod.seq_scans_per_s": [12, 15, 480, 950],
    "trackline.p95_ms": [240, 260, 2100, 4200],
}


def _search(corpus: dict[str, str], query: str) -> str:
    words = [w for w in query.lower().split() if len(w) > 2]
    hits = [f"[{doc_id}] {text}" for doc_id, text in corpus.items() if any(w in text.lower() for w in words)]
    return "\n".join(hits[:3]) or "no results"


def search_runbooks(query: str) -> str:
    return _search(RUNBOOKS, query)


def search_policies(query: str) -> str:
    return _search(POLICIES, query)


def search_incidents(query: str) -> str:
    return _search(INCIDENTS, query)


def get_service_status(service: str) -> str:
    return f"[status:{service}] {STATUS.get(service, 'unknown service')}"


def query_metrics(metric: str) -> dict[str, Any] | str:
    if metric not in METRICS:
        return f"unknown metric {metric!r}; known: {sorted(METRICS)}"
    return {"source": f"metric:{metric}", "last_4_intervals": METRICS[metric], "cite": f"[metric:{metric}]"}


_Q = obj({"query": {"type": "string", "minLength": 3}}, ["query"])
TOOLS: dict[str, FunctionTool] = {
    "search_runbooks": FunctionTool("search_runbooks", "Search IT runbooks by keywords.", _Q, search_runbooks),
    "search_policies": FunctionTool("search_policies", "Search HR and finance policies.", _Q, search_policies),
    "search_incidents": FunctionTool("search_incidents", "Search past incident reports.", _Q, search_incidents),
    "get_service_status": FunctionTool("get_service_status", "Current health of a service.",
                                       obj({"service": {"type": "string"}}, ["service"]), get_service_status),
    "query_metrics": FunctionTool("query_metrics", "Recent values of a named metric.",
                                  obj({"metric": {"type": "string"}}, ["metric"]), query_metrics),
}


def tools(*names: str) -> list[FunctionTool]:
    """Least privilege by construction: each agent gets only the tools it names."""
    return [TOOLS[n] for n in names]


__all__ = ["RUNBOOKS", "POLICIES", "INCIDENTS", "STATUS", "METRICS", "TOOLS", "tools"]
