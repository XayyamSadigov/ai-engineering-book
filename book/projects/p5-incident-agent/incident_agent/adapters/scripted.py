# path: book/projects/p5-incident-agent/incident_agent/adapters/scripted.py
"""A deterministic stand-in model that plays every role of Project 5, so the whole system runs
offline (tests, `LLM_PROVIDER=fake`, CI). It reads only what the prompts give it.

Its first report draft is deliberately flawed (an invented runbook id and an uncited claim) so
that the evaluator-optimizer loop and the deterministic Definition of Done are exercised on
every offline run. A real model makes such mistakes less predictably; the loop is the same.
"""
from __future__ import annotations

import json
import re
from typing import Any

from aie_core.llm.providers import FakeLLM
from aie_core.llm.types import CompletionRequest, Role, ToolCall

from ..domain.report import cited, parse_sections
from .metering import role_of

_LEDGER = re.compile(r"^- \[([^\]]+)\] \((\w+)\) ([^:]*): (.*)$", re.MULTILINE)


def _user(req: CompletionRequest) -> str:
    return next((m.text for m in req.messages if m.role is Role.USER), "")


def _field(text: str, name: str) -> str:
    m = re.search(rf"^- {name}: (.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else ""


# ------------------------------------------------------------------------------- planner
def planner(req: CompletionRequest) -> dict[str, Any]:
    text = "\n".join(m.text for m in req.messages if m.role is Role.USER)
    service = _field(text, "service")
    if "Deviation:" not in text:
        return {"hypothesis": f"a recent change to {service} or one of its dependencies",
                "steps": [
                    {"id": "s1", "tool": "query_service_metrics", "target": service,
                     "objective": f"Read {service} metrics and dependency health at alert time"},
                    {"id": "s2", "tool": "get_recent_deploys", "target": service,
                     "objective": f"List deploys to {service} in the previous 24 hours"},
                    {"id": "s3", "tool": "search_incidents", "target": f"{service} latency degradation",
                     "objective": "Find past incidents with the same symptoms"},
                    {"id": "s4", "tool": "search_runbooks", "target": "production incident response stabilise",
                     "objective": "Find the runbook the on-call engineer should follow"}]}
    ids = [int(i) for i in re.findall(r"\bs(\d+)\b", text)]
    nxt = max(ids, default=0) + 1
    remaining = [{"id": m.group(1), "tool": m.group(2), "target": m.group(3), "objective": m.group(4)}
                 for m in re.finditer(r"^- (s\d+) (\w+)\((.+?)\): (.+)$",
                                      text.split("Remaining steps in the current plan:")[-1].split("Deviation:")[0],
                                      re.MULTILINE)]
    dep = re.search(r"dependency (\S+) of", text)
    if dep:
        d = dep.group(1)
        new = [{"id": f"s{nxt}", "tool": "query_service_metrics", "target": d,
                "objective": f"Read {d} metrics at alert time"},
               {"id": f"s{nxt + 1}", "tool": "get_recent_deploys", "target": d,
                "objective": f"List deploys and migrations to {d} in the previous 24 hours"}]
        return {"hypothesis": f"a change to {d} degraded {service}", "steps": new + remaining}
    return {"hypothesis": "broaden the search", "steps": remaining + [
        {"id": f"s{nxt}", "tool": "search_incidents", "target": f"{service} incident",
         "objective": "Search past incidents with broader terms"}]}


# ------------------------------------------------------------------------------ executor
def executor(req: CompletionRequest) -> Any:
    goal = _user(req)
    tool = re.search(r"^Tool: (\S+)", goal, re.MULTILINE).group(1)  # type: ignore[union-attr]
    target = re.search(r"^Target: (.+)$", goal, re.MULTILINE).group(1).strip()  # type: ignore[union-attr]
    results = [m.text for m in req.messages if m.role is Role.TOOL]
    if not results:
        args = {"query": target} if tool.startswith("search_") else {"service": target}
        return [ToolCall(id=f"{tool}-1", name=tool, arguments=args)]
    lines = results[-1].splitlines()
    facts = [ln for ln in lines if ln.startswith("[")][:4]
    deps = [ln for ln in lines if ln.startswith("dependency ")]
    if not facts:
        return f"no evidence: {lines[0] if lines else 'empty result'}"
    return "Finding: " + "; ".join(facts + deps)


# -------------------------------------------------------------------------------- writer
def _rows(text: str) -> list[dict[str, str]]:
    return [{"id": m.group(1), "kind": m.group(2), "title": m.group(3), "text": m.group(4)}
            for m in _LEDGER.finditer(text)]


def _ratio(text: str) -> float:
    m = re.search(r"\(x([0-9.]+)\)", text)
    return float(m.group(1)) if m else 0.0


def writer(req: CompletionRequest) -> str:
    text = _user(req)
    revising = "Reviewer feedback:" in text
    rows = _rows(text.split("Step findings:")[0])
    service, fired, summary = _field(text, "service"), _field(text, "fired_at"), _field(text, "summary")
    metrics = [r for r in rows if r["kind"] == "metric" and "anomalous" in r["text"]]
    own = sorted((r for r in metrics if r["id"].startswith(f"metric:{service}.")), key=lambda r: -_ratio(r["text"]))
    deps = sorted((r for r in metrics if r not in own), key=lambda r: -_ratio(r["text"]))
    deploys = sorted((r for r in rows if r["kind"] == "deploy"), key=lambda r: r["text"])
    migration = next((r for r in deploys if " migration:" in r["text"]), None)
    release = next((r for r in deploys if f" {service} release:" in r["text"]), None)
    incident = next((r for r in rows if r["kind"] == "incident"
                     and (service in r["text"].lower() or (migration and "migration" in r["text"].lower()))), None)
    runbooks = [r for r in rows if r["kind"] == "runbook"]
    runbook = next((r for r in runbooks if "incident-response" in r["id"]), runbooks[0] if runbooks else None)
    lead = own[0] if own else (metrics[0] if metrics else None)

    if migration and deps:
        cause = (f"The migration {migration['title']} preceded the degradation [deploy:{migration['title']}]. "
                 f"Database signals moved with it: {deps[0]['title']} {deps[0]['text']} [{deps[0]['id']}]."
                 + (f" This matches a past incident in which a migration dropped a composite index and queries "
                    f"fell back to sequential scans [{incident['id']}]." if incident else "")
                 + f" Confidence is moderate: the index loss is inferred, not yet confirmed [{deps[0]['id']}].")
        short = f"a database migration on a dependency [deploy:{migration['title']}]"
        steps = ["Compare pg_indexes on the database with the pre-migration snapshot.",
                 "Recreate any missing index with CREATE INDEX CONCURRENTLY rather than rolling back.",
                 "Pause or throttle webhook retry workers while the index builds."]
    elif release:
        cause = (f"Release {release['title']} preceded the degradation [deploy:{release['title']}]. "
                 f"{lead['title']} {lead['text']} [{lead['id']}]." if lead else "")
        short = f"the {service} release [deploy:{release['title']}]"
        steps = [f"Disable the change behind its feature flag or roll back {release['title']}.",
                 "Confirm p95 recovers for 30 minutes before resolving."]
    else:
        cause = f"No deploy explains the degradation; signals: {lead['title']} [{lead['id']}]." if lead else ""
        short = "unknown"
        steps = ["Escalate to the owning team."]

    impact = [f"- {r['title']}: {r['text']} [{r['id']}]" for r in own[:3]]
    timeline = [f"- {r['text'].split(' UTC ')[0]} UTC: {r['title']} ({r['text'].split(': ', 1)[-1]}) [{r['id']}]"
                for r in deploys]
    timeline += [f"- {r['title']} {r['text'].split(', ')[-1]} [{r['id']}]" for r in own[:1] + deps[:1]]
    rb = "it-db-index-rebuild-runbook" if not revising else (runbook["id"] if runbook else "")
    if not revising and impact:
        impact[0] = impact[0].rsplit(" [", 1)[0]          # first draft: one claim loses its citation
    report = [
        f"# Incident report: {summary}", "",
        "## Summary",
        f"{summary} at {fired} [{lead['id']}]. The likely cause is {short}." if lead else summary, "",
        "## Impact", *impact, "",
        "## Timeline", *timeline, "",
        "## Likely cause", cause, "",
        "## Recommended runbook",
        f"Follow [{rb}]: stabilise first, keep the incident timeline, and post updates on the SEV2 cadence.", "",
        "## Next steps", *[f"{i}. {s}" for i, s in enumerate(steps, 1)],
    ]
    return "\n".join(report)


# --------------------------------------------------------------------------------- judge
def judge(req: CompletionRequest) -> dict[str, Any]:
    text = _user(req)
    candidate = text.split("<candidate>")[-1].split("</candidate>")[0]
    cause = parse_sections(candidate).get("likely cause", "")
    ids = cited(cause)
    converging = any(i.startswith("deploy:") for i in ids) and any(i.startswith(("metric:", "inc-")) for i in ids)
    if converging:
        return {"reasoning": "The cause cites a change and an independent signal; steps are specific.",
                "score": 5, "flagged": []}
    return {"reasoning": "The cause is not supported by converging evidence.", "score": 3,
            "flagged": ["cause lacks an independent signal"]}


HANDLERS = {"planner": planner, "executor": executor, "writer": writer, "judge": judge}


def scripted_handler(req: CompletionRequest) -> Any:
    out = HANDLERS[role_of(req)](req)
    return json.dumps(out) if isinstance(out, dict) else out


def scripted_llm(**overrides: Any) -> FakeLLM:
    """FakeLLM playing every role; pass e.g. writer=my_writer to replace one role in a test."""
    handlers = {**HANDLERS, **overrides}

    def handler(req: CompletionRequest) -> Any:
        out = handlers[role_of(req)](req)
        return json.dumps(out) if isinstance(out, dict) else out

    return FakeLLM(handler=handler, model="scripted-incident-model")


__all__ = ["planner", "executor", "writer", "judge", "scripted_handler", "scripted_llm", "HANDLERS"]
