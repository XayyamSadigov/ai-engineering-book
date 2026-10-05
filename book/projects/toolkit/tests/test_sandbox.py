# path: book/projects/toolkit/tests/test_sandbox.py
from __future__ import annotations

import os
import sys

import pytest

from toolkit import SandboxError, SandboxLimits, SandboxRunner, make_python_tool

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX sandbox")


def test_runs_code_in_temp_dir_and_collects_outputs():
    r = SandboxRunner().run_python("import os\nprint(os.getcwd())\nopen('out.txt','w').write('42')")
    assert r.ok and "sbx-" in r.stdout and r.files == {"out.txt": "42"}


def test_wall_clock_timeout_kills_process():
    r = SandboxRunner(SandboxLimits(timeout_s=0.5, cpu_s=5)).run_python("while True: pass")
    assert r.timed_out and not r.ok and r.duration_ms < 5000


def test_environment_secrets_are_not_inherited(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    r = SandboxRunner().run_python("import os\nprint(sorted(os.environ))\nprint(os.environ.get('OPENAI_API_KEY'))")
    assert "sk-should-not-leak" not in r.stdout and "None" in r.stdout


def test_output_is_capped():
    r = SandboxRunner(SandboxLimits(max_output_bytes=1000)).run_python("print('a' * 100000)")
    assert r.truncated and len(r.stdout) == 1000


def test_file_paths_cannot_escape():
    with pytest.raises(ValueError):
        SandboxRunner().run(["true"], files={"../escape.txt": "x"})


def test_file_size_limit_stops_large_writes():
    r = SandboxRunner(SandboxLimits(file_size_mb=1)).run_python(
        "f = open('big.bin','wb')\nfor _ in range(4): f.write(b'0' * 1024 * 1024)\nf.close()\nprint('done')")
    assert not r.ok and "done" not in r.stdout


def test_network_deny_requires_platform_support():
    if SandboxRunner.network_isolation_available():
        pytest.skip("platform supports unshare; covered elsewhere")
    with pytest.raises(SandboxError):
        SandboxRunner(network="deny")


def test_python_tool_wrapper(ctx):
    from aie_core.llm.types import ToolCall

    from toolkit import ToolExecutor, ToolRegistry

    tool = make_python_tool(SandboxRunner(), required_permission=None)
    ex = ToolExecutor(ToolRegistry([tool]))
    r = ex.execute(ToolCall(id="1", name="run_python", arguments={"code": "print(6*7)"}), ctx)
    assert r.ok and r.data["stdout"].strip() == "42"
    ex.close()
