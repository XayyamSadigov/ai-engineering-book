# path: book/projects/agentkit/tests/test_example.py
"""The Northwind incident example runs offline, pauses for approval, completes, and replays."""
from __future__ import annotations

import importlib.util
import io
from pathlib import Path

from agentkit import TerminationReason

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "northwind_incident.py"


def test_northwind_incident_example(tmp_path):
    spec = importlib.util.spec_from_file_location("northwind_incident", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    out = io.StringIO()
    summary = module.main(str(tmp_path), out=out)

    assert summary["first"].stop_reason is TerminationReason.APPROVAL_REQUIRED
    final = summary["final"]
    assert final.ok, out.getvalue()
    assert final.trajectory() == ["get_service_status", "query_metrics", "search_docs", "create_ticket"]
    assert "[inc-2026-02-tracking-latency]" in final.final_answer
    assert len(summary["tickets"]) == 1
    assert summary["replay"].identical, summary["replay"].summary()
    assert {s.name for s in summary["spans"]} >= {"agent.run", "agent.step", "agent.tool"}
