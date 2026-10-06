# path: book/projects/examples/ch25/tests/test_hardening.py
"""Edge cases behind the chapter's guarantees: approvals, evidence, synthetic ids, gate errors."""
from __future__ import annotations

import shutil
from pathlib import Path

import release_gate
from taskevals.extraction import evidence_status
from taskevals.synthetic import SourceDoc, validate_candidates
from taskevals.trajectory import NORTHWIND_TOOLS, Step, Trajectory, assert_approval_before_side_effects

from test_release_gate import BASELINES, GATES, runs  # noqa: F401  (module-scoped fixture)
from test_synthetic import DOCS, _generator


def _send(call_id: str, body: str) -> Step:
    return Step(type="tool_call", call_id=call_id, tool="send_reply", arguments={"ticket_id": "T", "body": body})


def test_a_reused_call_id_cannot_borrow_an_approval() -> None:
    traj = Trajectory(trajectory_id="t", task_id="x", goal="g", steps=[
        _send("5.0", "approved text"), Step(type="approval", call_id="5.0", decision="approved"),
        Step(type="tool_result", call_id="5.0", status="ok", output={"sent": True}),
        _send("5.0", "something else"), Step(type="tool_result", call_id="5.0", status="ok", output={"sent": True}),
    ])
    assert not assert_approval_before_side_effects(traj, NORTHWIND_TOOLS).passed


def test_an_unrecorded_replayed_send_without_approval_is_unsafe() -> None:
    traj = Trajectory(trajectory_id="t", task_id="x", goal="g", steps=[
        _send("c1", "new text"), Step(type="tool_result", call_id="c1", status="unrecorded", output={}),
    ])
    assert not assert_approval_before_side_effects(traj, NORTHWIND_TOOLS).passed


def test_evidence_must_contain_the_value_as_a_whole_token() -> None:
    doc = "Subtotal $11,488.00\nRef INV-1042\nTotal $1,488.00"
    assert evidence_status(doc, "Subtotal $11,488.00", 1488.0) == "wrong_location"
    assert evidence_status(doc, "Ref INV-1042", "INV-104") == "wrong_location"
    assert evidence_status(doc, "Total $1,488.00", 1488.0) == "correct"
    from taskevals.extraction import mentions
    assert mentions("INV-001, 1488.00", 1488.0) and mentions("No.5 qty", 5)


def test_synthetic_ids_do_not_collide_across_batches() -> None:
    gen, _ = _generator()
    candidates = gen.generate([SourceDoc(id="hr-pto-policy", text=DOCS["hr-pto-policy"])])
    batches = [validate_candidates([c], DOCS)[0] for c in candidates]   # one candidate per batch
    ids = [cases[0].id for cases in batches if cases]
    assert len(ids) >= 2 and len(set(ids)) == len(ids)


def test_an_unreadable_baseline_is_a_pipeline_error(runs, tmp_path) -> None:  # noqa: F811
    baselines = tmp_path / "baselines"
    shutil.copytree(BASELINES, baselines)
    (baselines / "agent.json").write_text("{not json")
    code = release_gate.main(["--config", str(GATES), "--runs", str(runs["candidate"]), "--baselines", str(baselines),
                              "--out", str(tmp_path / "out")])
    assert code == release_gate.EXIT_ERROR and (tmp_path / "out" / "summary.md").exists()
