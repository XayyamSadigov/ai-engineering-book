# path: book/projects/examples/ch38/test_ch38_coding_harness.py
"""The coding harness: patching, sandboxed tests, scope rules, and a Definition of Done
that runs the tests itself instead of believing the transcript."""
from __future__ import annotations

from typing import Any

import pytest

from aie_core.llm.providers import FakeLLM
from aie_core.llm.types import ToolCall
from agentkit import AgentRuntime, AgentState, Budget, FinalAnswer, Note, ToolResult

from coding_harness import (
    SEED_REPO, TASK, CodingTools, PatchError, Workspace, apply_hunks, coding_dod, parse_unified_diff,
)

FIX_P2 = """--- a/northwind_sla/sla.py
+++ b/northwind_sla/sla.py
@@ -2,5 +2,5 @@
 from datetime import datetime, timedelta

-RESPONSE_HOURS = {"P1": 1, "P2": 4, "P3": 24}
+RESPONSE_HOURS = {"P1": 1, "P2": 8, "P3": 24}


"""

FIX_UNKNOWN = """--- a/northwind_sla/sla.py
+++ b/northwind_sla/sla.py
@@ -7,3 +7,5 @@
 def response_deadline(priority: str, opened_at: datetime) -> datetime:
-    hours = RESPONSE_HOURS.get(priority, 24)
+    if priority not in RESPONSE_HOURS:
+        raise ValueError(f"unknown priority {priority!r}")
+    hours = RESPONSE_HOURS[priority]
     return opened_at + timedelta(hours=hours)
"""

CHEAT = """--- a/tests/test_sla.py
+++ b/tests/test_sla.py
@@ -12,2 +12,2 @@
 def test_p2_is_eight_hours_after_policy_change():
-    assert response_deadline("P2", T0) == T0 + timedelta(hours=8)
+    assert response_deadline("P2", T0) == T0 + timedelta(hours=4)
"""


def call(name: str, i: int, **args: Any) -> list[ToolCall]:
    return [ToolCall(id=f"{name}-{i}", name=name, arguments=args)]


@pytest.fixture
def harness(tmp_path):
    ws = Workspace.from_files(tmp_path / "repo", SEED_REPO, allowed=("northwind_sla/*",))
    tools = CodingTools(ws)
    return ws, tools


def runtime(tools: CodingTools, responses: list[Any]) -> AgentRuntime:
    return AgentRuntime(FakeLLM(responses=responses), tools.as_tools(), dod=coding_dod(tools, max_changed_lines=20),
                        budget=Budget(max_steps=15))


def test_inspect_edit_test_repair_loop_reaches_done(harness):
    ws, tools = harness
    result = runtime(tools, [
        call("search_code", 1, pattern="RESPONSE_HOURS"),
        call("read_file", 2, path="northwind_sla/sla.py"),
        call("apply_patch", 3, diff=FIX_P2),
        call("run_tests", 4),                                 # one test still fails
        call("apply_patch", 5, diff=FIX_UNKNOWN),
        call("run_tests", 6),
        call("show_diff", 7),
        "P2 target is now 8 hours and unknown priorities raise ValueError. All tests pass.",
    ]).run(TASK)
    assert result.ok, result.detail
    assert result.trajectory() == ["search_code", "read_file", "apply_patch", "run_tests", "apply_patch",
                                   "run_tests", "show_diff"]
    first, second = [r for r in result.events_of(ToolResult) if r.tool == "run_tests"]
    assert "FAILED" in first.content and "1 failed" in first.content
    assert "PASSED" in second.content
    assert ws.changed_files() == ["northwind_sla/sla.py"]
    [final] = result.events_of(FinalAnswer)
    assert {c["name"] for c in final.checks} >= {"tests_pass", "diff_in_scope"}
    assert all(c["passed"] for c in final.checks)


def test_premature_done_claim_is_rejected_by_running_the_tests(harness):
    ws, tools = harness
    result = runtime(tools, [
        call("apply_patch", 1, diff=FIX_P2),
        call("run_tests", 2),
        call("show_diff", 3),
        "Done, everything passes.",                           # it does not: the unknown-priority test fails
        call("apply_patch", 4, diff=FIX_UNKNOWN),
        call("run_tests", 5),
        "Fixed both requirements; tests pass.",
    ]).run(TASK)
    assert result.ok
    rejected = [n for n in result.events_of(Note) if n.kind == "dod_rejected"]
    assert len(rejected) == 1 and "tests fail" in rejected[0].text


def test_protected_test_file_cannot_be_edited_to_pass(harness):
    ws, tools = harness
    out = tools.apply_patch(CHEAT)
    assert not out.ok and out.error_class.value == "permission"
    assert ws.changed_files() == []


def test_path_traversal_and_out_of_scope_paths_are_refused(harness):
    ws, tools = harness
    evil = "--- /dev/null\n+++ b/../escape.py\n@@ -0,0 +1 @@\n+print('x')\n"
    assert not tools.apply_patch(evil).ok
    readme = "--- a/README.md\n+++ b/README.md\n@@ -1,1 +1,1 @@\n-# northwind-sla\n+# hacked\n"
    out = tools.apply_patch(readme)
    assert not out.ok and "outside the paths" in out.content


def test_stale_patch_fails_with_the_actual_lines(harness):
    ws, tools = harness
    stale = FIX_P2.replace('"P2": 4', '"P2": 6')
    out = tools.apply_patch(stale)
    assert not out.ok and "does not match" in out.content and "RESPONSE_HOURS" in out.content


def test_scope_and_size_checks_fail_closed(harness):
    ws, tools = harness
    (ws.root / "tests" / "test_sla.py").write_text("def test_ok():\n    assert True\n")   # edited behind the harness
    verdicts = {v.name: v for v in coding_dod(tools).verify("done", AgentState()).verdicts}
    assert not verdicts["diff_in_scope"].passed
    assert "tests/test_sla.py" in verdicts["diff_in_scope"].reason


def test_unified_diff_parser_handles_new_files_and_multiple_hunks():
    patches = parse_unified_diff("--- /dev/null\n+++ b/pkg/new.py\n@@ -0,0 +1,2 @@\n+a = 1\n+b = 2\n")
    assert patches[0].old_path is None and apply_hunks([], patches[0].hunks) == ["a = 1", "b = 2"]
    original = [f"line {i}" for i in range(1, 21)]
    two = parse_unified_diff("--- a/f\n+++ b/f\n@@ -2,1 +2,2 @@\n-line 2\n+line 2a\n+line 2b\n"
                             "@@ -15,1 +16,1 @@\n-line 15\n+line 15!\n")
    out = apply_hunks(original, two[0].hunks)
    assert out[1:3] == ["line 2a", "line 2b"] and out[15] == "line 15!" and len(out) == 21
    with pytest.raises(PatchError):
        parse_unified_diff("just some text")


def test_sandbox_runs_on_a_copy_and_cannot_touch_the_workspace(harness):
    ws, tools = harness
    (ws.root / "tests" / "test_side_effect.py").write_text(
        "from pathlib import Path\n\ndef test_write():\n    Path('northwind_sla/sla.py').write_text('broken')\n")
    ws.baseline = ws.snapshot()
    tools.pytest("tests/test_side_effect.py")
    assert "RESPONSE_HOURS" in (ws.root / "northwind_sla" / "sla.py").read_text()
