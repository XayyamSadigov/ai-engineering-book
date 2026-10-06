# path: book/projects/examples/ch38/coding_harness.py
"""A minimal coding-agent harness on agentkit.

Narrow tools instead of a shell: search_code, read_file, apply_patch (unified diff),
run_tests (pytest in Chapter 16's SandboxRunner, on a throwaway copy of the workspace), and
show_diff. A deterministic Definition of Done decides when the task is finished: the verifier
runs the tests itself, checks that the diff stays inside the allowed paths, that protected
files are untouched, and that the change is small enough to review.

The workspace is a plain directory (no git needed); the baseline snapshot taken at start is
what `show_diff` and the scope checks compare against.
"""
from __future__ import annotations

import difflib
import fnmatch
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentkit import (
    AgentState, Check, DefinitionOfDone, ErrorClass, FunctionTool, SideEffect, ToolOutput, tool_was_called,
)
from toolkit import SandboxLimits, SandboxRunner

MAX_FILE_BYTES = 200_000
TEXT_SUFFIXES = {".py", ".md", ".txt", ".toml", ".cfg", ".ini", ".json", ".yaml", ".yml"}


class PatchError(ValueError):
    """The patch does not apply. The message tells the model what the file actually contains."""


# --------------------------------------------------------------------------- unified diff
@dataclass
class Hunk:
    old_start: int
    old_len: int
    new_start: int
    new_len: int
    lines: list[str] = field(default_factory=list)     # each starts with ' ', '-', or '+'


@dataclass
class FilePatch:
    old_path: str | None      # None for a new file
    new_path: str | None      # None for a deletion
    hunks: list[Hunk] = field(default_factory=list)


HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _strip_prefix(raw: str) -> str | None:
    path = raw.split("\t")[0].strip()
    if path == "/dev/null":
        return None
    return path[2:] if path.startswith(("a/", "b/")) else path


def parse_unified_diff(text: str) -> list[FilePatch]:
    patches: list[FilePatch] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
            current = FilePatch(_strip_prefix(line[4:]), _strip_prefix(lines[i + 1][4:]))
            patches.append(current)
            i += 2
            continue
        m = HUNK_RE.match(line)
        if m:
            if not patches:
                raise PatchError("hunk before any '--- a/path' / '+++ b/path' header")
            hunk = Hunk(int(m.group(1)), int(m.group(2) or 1), int(m.group(3)), int(m.group(4) or 1))
            i += 1
            while i < len(lines) and not lines[i].startswith(("@@", "--- ")):
                body = lines[i]
                if body.startswith("\\"):              # "\ No newline at end of file"
                    i += 1
                    continue
                if body == "":
                    body = " "                          # editors strip the space on blank context lines
                if body[0] not in " +-":
                    raise PatchError(f"bad hunk line {body!r}: lines must start with ' ', '-' or '+'")
                hunk.lines.append(body)
                i += 1
            patches[-1].hunks.append(hunk)
            continue
        i += 1
    if not patches:
        raise PatchError("no file headers found; send a unified diff with '--- a/path' and '+++ b/path'")
    return patches


def apply_hunks(original: list[str], hunks: list[Hunk], *, fuzz: int = 3, path: str = "") -> list[str]:
    """Apply hunks in order. Each hunk's context and removed lines must match exactly; the
    position may be off by up to `fuzz` lines (earlier edits shift line numbers)."""
    result = list(original)
    offset = 0
    for h in hunks:
        old = [ln[1:] for ln in h.lines if ln[0] in " -"]
        new = [ln[1:] for ln in h.lines if ln[0] in " +"]
        want = max(h.old_start - 1 + offset, 0) if h.old_len else h.old_start + offset
        found = None
        for delta in sorted(range(-fuzz, fuzz + 1), key=abs):
            pos = want + delta
            if 0 <= pos <= len(result) - len(old) and result[pos:pos + len(old)] == old:
                found = pos
                break
        if found is None:
            lo = max(want - 2, 0)
            actual = "\n".join(f"{n + 1:4d}| {t}" for n, t in enumerate(result[lo:lo + len(old) + 4], start=lo))
            raise PatchError(f"hunk @@ -{h.old_start},{h.old_len} @@ does not match {path or 'the file'}. "
                             f"Lines there are:\n{actual}\nRe-read the file and regenerate the patch.")
        result[found:found + len(old)] = new
        offset += len(new) - len(old)
    return result


# --------------------------------------------------------------------------- workspace
class Workspace:
    """A repository directory plus a frozen baseline. All paths are confined to `root`;
    `protected` globs may be read but never patched. Test configuration (conftest.py,
    pytest.ini, pyproject.toml, setup.cfg, tox.ini) and anything named like pytest itself are
    protected by default: each can change what "the tests pass" means without touching a test."""

    def __init__(self, root: str | Path, *,
                 protected: tuple[str, ...] = ("tests/*", "*.lock", ".env*", "conftest.py", "*/conftest.py",
                                               "pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini",
                                               "pytest.py", "pytest/*", "_pytest*", "py.py"),
                 allowed: tuple[str, ...] = ("*",)) -> None:
        self.root = Path(root).resolve()
        self.protected = protected
        self.allowed = allowed
        self.baseline = self.snapshot()

    @classmethod
    def from_files(cls, root: str | Path, files: dict[str, str], **kwargs: Any) -> "Workspace":
        root = Path(root)
        for rel, content in files.items():
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        return cls(root, **kwargs)

    def resolve(self, rel: str) -> Path:
        if rel.startswith("/") or "\\" in rel:
            raise PermissionError(f"path must be relative to the repository root: {rel!r}")
        target = (self.root / rel).resolve()
        if not target.is_relative_to(self.root):
            raise PermissionError(f"path escapes the repository: {rel!r}")
        return target

    @staticmethod
    def tracked(rel: str) -> bool:
        """Files the harness snapshots and diffs. The agent may write only these: a change the
        diff cannot see is a change the scope check cannot judge."""
        parts = Path(rel).parts
        return Path(rel).suffix in TEXT_SUFFIXES and not any(p.startswith(".") or p == "__pycache__" for p in parts)

    def snapshot(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for p in sorted(self.root.rglob("*")):
            rel = p.relative_to(self.root).as_posix()
            if p.is_file() and self.tracked(rel) and p.stat().st_size <= MAX_FILE_BYTES:
                out[rel] = p.read_text(encoding="utf-8")
        return out

    def changed_files(self) -> list[str]:
        now = self.snapshot()
        return sorted(f for f in set(now) | set(self.baseline) if now.get(f) != self.baseline.get(f))

    def diff(self) -> str:
        now = self.snapshot()
        chunks = []
        for f in self.changed_files():
            a = self.baseline.get(f, "").splitlines(keepends=True)
            b = now.get(f, "").splitlines(keepends=True)
            chunks.append("".join(difflib.unified_diff(a, b, f"a/{f}", f"b/{f}")))
        return "".join(chunks)

    def changed_line_count(self) -> int:
        return sum(1 for ln in self.diff().splitlines()
                   if ln[:1] in "+-" and not ln.startswith(("+++", "---")))

    def is_protected(self, rel: str) -> bool:
        return any(fnmatch.fnmatch(rel, g) for g in self.protected)

    def is_allowed(self, rel: str) -> bool:
        return any(fnmatch.fnmatch(rel, g) for g in self.allowed)


# --------------------------------------------------------------------------- tools
class CodingTools:
    def __init__(self, ws: Workspace, *, sandbox: SandboxRunner | None = None, max_hits: int = 30) -> None:
        self.ws = ws
        self.max_hits = max_hits
        self.sandbox = sandbox or SandboxRunner(SandboxLimits(timeout_s=60, cpu_s=60, memory_mb=2048,
                                                              max_output_bytes=20_000), collect_outputs=())
        self.test_runs: list[dict[str, Any]] = []

    def search_code(self, pattern: str, glob: str = "*") -> ToolOutput:
        try:
            rx = re.compile(pattern)
        except re.error as exc:
            return ToolOutput.failure(f"invalid regex {pattern!r}: {exc}", ErrorClass.VALIDATION)
        hits = []
        for rel, text in self.ws.snapshot().items():
            if not fnmatch.fnmatch(rel, glob):
                continue
            for n, line in enumerate(text.splitlines(), start=1):
                if rx.search(line):
                    hits.append(f"{rel}:{n}: {line.strip()[:160]}")
        more = f"\n... {len(hits) - self.max_hits} more; narrow the pattern" if len(hits) > self.max_hits else ""
        return ToolOutput(content="\n".join(hits[: self.max_hits]) + more if hits else "no matches",
                          data={"hits": len(hits)})

    def read_file(self, path: str, start: int = 1, end: int = 200) -> ToolOutput:
        target = self.ws.resolve(path)
        if not target.is_file():
            return ToolOutput.failure(f"no such file: {path}", ErrorClass.IMPOSSIBLE)
        lines = target.read_text(encoding="utf-8").splitlines()
        body = "\n".join(f"{n:4d}| {t}" for n, t in enumerate(lines[start - 1:end], start=start))
        return ToolOutput(content=f"{path} (lines {start}-{min(end, len(lines))} of {len(lines)})\n{body}")

    def apply_patch(self, diff: str) -> ToolOutput:
        try:
            patches = parse_unified_diff(diff)
            staged: dict[Path, str | None] = {}
            for fp in patches:
                raw = fp.new_path or fp.old_path
                assert raw is not None
                target = self.ws.resolve(raw)
                rel = target.relative_to(self.ws.root).as_posix()   # judge the normalized path, not the raw one
                if not self.ws.tracked(rel):
                    return ToolOutput.failure(f"{raw} is a hidden or non-text path; the agent may not write it",
                                              ErrorClass.PERMISSION)
                if self.ws.is_protected(rel):
                    return ToolOutput.failure(f"{rel} is protected and cannot be modified by the agent",
                                              ErrorClass.PERMISSION)
                if not self.ws.is_allowed(rel):
                    return ToolOutput.failure(f"{rel} is outside the paths this task may change: "
                                              f"{list(self.ws.allowed)}", ErrorClass.PERMISSION)
                if fp.new_path is None:
                    return ToolOutput.failure("file deletion is not allowed through apply_patch",
                                              ErrorClass.PERMISSION)
                if target in staged:            # a second patch to the same file applies on top of the first
                    original = (staged[target] or "").splitlines()
                else:
                    original = [] if fp.old_path is None else target.read_text(encoding="utf-8").splitlines()
                if fp.old_path is None and (target.exists() or target in staged):
                    raise PatchError(f"{rel} already exists; patch it instead of creating it")
                staged[target] = "\n".join(apply_hunks(original, fp.hunks, path=rel)) + "\n"
        except PatchError as exc:
            return ToolOutput.failure(str(exc), ErrorClass.VALIDATION)
        except (PermissionError, FileNotFoundError) as exc:
            return ToolOutput.failure(str(exc), ErrorClass.PERMISSION)
        for target, content in staged.items():          # all-or-nothing: write only after every hunk applied
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content or "", encoding="utf-8")
        names = [str(t.relative_to(self.ws.root)) for t in staged]
        return ToolOutput(content=f"patched {names}; {self.ws.changed_line_count()} changed lines in total",
                          artifacts={"changed_files": self.ws.changed_files()})

    def run_tests(self, selector: str = "tests") -> ToolOutput:
        if selector.startswith("-") or ".." in selector:
            return ToolOutput.failure("selector must be a test path or node id, not an option", ErrorClass.VALIDATION)
        report = self.pytest(selector)
        self.test_runs.append(report)
        status = "PASSED" if report["ok"] else "FAILED"
        return ToolOutput(content=f"tests {status}: {report['summary']}\n{report['tail']}", ok=True,
                          data=report, artifacts={"last_test_status": status})

    def show_diff(self) -> ToolOutput:
        diff = self.ws.diff()
        return ToolOutput(content=diff or "no changes", artifacts={"diff_lines": self.ws.changed_line_count()})

    def pytest(self, selector: str = "tests") -> dict[str, Any]:
        """Run pytest on a fresh copy of the workspace, with a scrubbed environment and the
        sandbox limits. The copy keeps ordinary writes away from the workspace, but this is
        not filesystem isolation: test code could still reach the workspace by absolute path.
        Real isolation needs a container or microVM (Chapter 16)."""
        res = self.sandbox.run([sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                                "-o", "addopts=", "--rootdir=.", selector], files=self.ws.snapshot())
        out = (res.stdout + res.stderr).strip()
        summary = next((ln for ln in reversed(out.splitlines()) if re.search(r"passed|failed|error", ln)),
                       "timed out" if res.timed_out else f"exit code {res.exit_code}")
        tail = "\n".join(out.splitlines()[-25:])
        return {"ok": res.ok, "summary": summary.strip("= "), "tail": tail, "exit_code": res.exit_code,
                "timed_out": res.timed_out, "duration_ms": res.duration_ms}

    def as_tools(self) -> list[FunctionTool]:
        s = {"type": "object", "additionalProperties": False}
        return [
            FunctionTool("search_code", "Regex search over repository text files; returns path:line: text.",
                         {**s, "properties": {"pattern": {"type": "string", "minLength": 1},
                                              "glob": {"type": "string"}}, "required": ["pattern"]},
                         fn=self.search_code),
            FunctionTool("read_file", "Read a file with line numbers.",
                         {**s, "properties": {"path": {"type": "string"}, "start": {"type": "integer", "minimum": 1},
                                              "end": {"type": "integer", "minimum": 1}}, "required": ["path"]},
                         fn=self.read_file),
            FunctionTool("apply_patch", "Apply a unified diff (--- a/path, +++ b/path, @@ hunks). All files or none.",
                         {**s, "properties": {"diff": {"type": "string", "minLength": 10}}, "required": ["diff"]},
                         fn=self.apply_patch, side_effect=SideEffect.WRITE, idempotent=False),
            FunctionTool("run_tests", "Run pytest in a sandbox on a copy of the repository.",
                         {**s, "properties": {"selector": {"type": "string"}}, "required": []},
                         fn=self.run_tests),
            FunctionTool("show_diff", "Show the full diff against the starting state.",
                         {**s, "properties": {}, "required": []}, fn=self.show_diff),
        ]


# --------------------------------------------------------------------------- Definition of Done
def coding_dod(tools: CodingTools, *, max_changed_lines: int = 60, test_selector: str = "tests") -> DefinitionOfDone:
    ws = tools.ws

    def tests_pass(answer: str, state: AgentState) -> tuple[bool, str]:
        report = tools.pytest(test_selector)       # the verifier runs the tests; it does not trust the transcript
        return report["ok"], "" if report["ok"] else f"tests fail: {report['summary']}"

    def scoped(answer: str, state: AgentState) -> tuple[bool, str]:
        bad = [f for f in ws.changed_files() if ws.is_protected(f) or not ws.is_allowed(f)]
        return not bad, f"changes outside the allowed scope: {bad}" if bad else ""

    def has_change(answer: str, state: AgentState) -> tuple[bool, str]:
        return bool(ws.changed_files()), "no file was changed"

    def small(answer: str, state: AgentState) -> tuple[bool, str]:
        n = ws.changed_line_count()
        return n <= max_changed_lines, f"diff has {n} changed lines; limit is {max_changed_lines}"

    return DefinitionOfDone(
        Check("has_change", has_change),
        Check("tests_pass", tests_pass),
        Check("diff_in_scope", scoped),
        Check(f"diff<={max_changed_lines}_lines", small),
        tool_was_called("run_tests", "show_diff"),
        description="The task is done only when all of these hold (checked by the harness, not by you):",
    )


SEED_REPO: dict[str, str] = {
    "README.md": "# northwind-sla\nResponse-time targets for Northwind support tickets.\n",
    "northwind_sla/__init__.py": "",
    "northwind_sla/sla.py": '''"""Response-time SLA for Northwind support tickets."""
from datetime import datetime, timedelta

RESPONSE_HOURS = {"P1": 1, "P2": 4, "P3": 24}


def response_deadline(priority: str, opened_at: datetime) -> datetime:
    hours = RESPONSE_HOURS.get(priority, 24)
    return opened_at + timedelta(hours=hours)


def is_breached(priority: str, opened_at: datetime, now: datetime) -> bool:
    return now > response_deadline(priority, opened_at)
''',
    "tests/test_sla.py": '''from datetime import datetime, timedelta

import pytest

from northwind_sla.sla import is_breached, response_deadline

T0 = datetime(2026, 3, 2, 9, 0)


def test_p1_is_one_hour():
    assert response_deadline("P1", T0) == T0 + timedelta(hours=1)


def test_p2_is_eight_hours_after_policy_change():
    assert response_deadline("P2", T0) == T0 + timedelta(hours=8)


def test_unknown_priority_is_rejected():
    with pytest.raises(ValueError):
        response_deadline("P9", T0)


def test_breach():
    assert is_breached("P1", T0, T0 + timedelta(hours=2))
''',
}

TASK = ("SLA policy v4 changed the P2 response target to 8 hours, and unknown priorities must raise "
        "ValueError instead of silently getting 24 hours. Make tests/test_sla.py pass without editing tests.")


__all__ = [
    "PatchError", "Hunk", "FilePatch", "parse_unified_diff", "apply_hunks", "Workspace", "CodingTools",
    "coding_dod", "SEED_REPO", "TASK",
]
