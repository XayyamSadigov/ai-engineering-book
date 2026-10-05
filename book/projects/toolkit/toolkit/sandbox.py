# path: book/projects/toolkit/toolkit/sandbox.py
"""Run untrusted code in a child process with a wall-clock timeout, resource limits,
a throwaway working directory, and a scrubbed environment.

This is a *process* sandbox: it bounds time, CPU, memory, file size, open files, and
what environment the child inherits. It does not isolate the filesystem (the child can
read anything the parent user can) and it does not block the network unless you ask for
`network="deny"` on a Linux host with `unshare`. For hostile code use a container or
microVM with no network, a read-only root, and a dropped-privilege user; keep this class
as the interface and swap the implementation.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .registry import SideEffect, Tool

try:  # POSIX only
    import resource
except ImportError:  # pragma: no cover - Windows
    resource = None  # type: ignore[assignment]


class SandboxLimits(BaseModel):
    timeout_s: float = 5.0          # wall clock, enforced by the parent
    cpu_s: int = 2                  # RLIMIT_CPU, enforced by the kernel
    memory_mb: int = 512            # RLIMIT_AS (ignored by some kernels, e.g. macOS)
    file_size_mb: int = 8           # RLIMIT_FSIZE: largest file the child may write
    open_files: int = 64            # RLIMIT_NOFILE
    max_output_bytes: int = 64_000  # stdout/stderr returned to the caller


class SandboxResult(BaseModel):
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    truncated: bool = False
    duration_ms: float = 0.0
    signal: str | None = None
    files: dict[str, str] = Field(default_factory=dict)  # small text outputs the code wrote

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


class SandboxError(RuntimeError):
    pass


SAFE_ENV_KEYS = ("PATH", "LANG", "LC_ALL", "TZ")


class SandboxRunner:
    def __init__(self, limits: SandboxLimits | None = None, *, env_allowlist: tuple[str, ...] = SAFE_ENV_KEYS,
                 python_executable: str = sys.executable, network: Literal["inherit", "deny"] = "inherit",
                 collect_outputs: tuple[str, ...] = ("*.txt", "*.csv", "*.json")) -> None:
        self.limits = limits or SandboxLimits()
        self.env_allowlist = env_allowlist
        self.python_executable = python_executable
        self.network = network
        self.collect_outputs = collect_outputs
        if network == "deny" and not self.network_isolation_available():
            raise SandboxError("network='deny' needs Linux with unprivileged 'unshare'; use a container instead")

    @staticmethod
    def network_isolation_available() -> bool:
        return sys.platform.startswith("linux") and shutil.which("unshare") is not None

    # -------------------------------------------------------------------- running
    def run(self, argv: list[str], *, stdin: str | None = None, files: dict[str, str] | None = None) -> SandboxResult:
        """Run `argv` inside a fresh temporary directory that is deleted afterwards."""
        if not argv:
            raise ValueError("argv must not be empty")
        with tempfile.TemporaryDirectory(prefix="sbx-") as root:
            work = Path(root) / "work"
            work.mkdir()
            for name, content in (files or {}).items():
                target = (work / name).resolve()
                if not target.is_relative_to(work.resolve()):
                    raise ValueError(f"file path escapes the sandbox: {name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
            if self.network == "deny":
                argv = ["unshare", "--user", "--map-root-user", "--net", "--", *argv]
            out_path, err_path = Path(root) / "stdout", Path(root) / "stderr"
            start = time.monotonic()
            with out_path.open("wb") as out, err_path.open("wb") as err:
                proc = subprocess.Popen(
                    argv, cwd=work, env=self._env(work), stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                    stdout=out, stderr=err, preexec_fn=self._apply_limits if resource else None,
                    start_new_session=True,  # own process group, so we can kill children too
                    close_fds=True,
                )
                timed_out = False
                try:
                    if stdin is not None:
                        assert proc.stdin is not None
                        proc.stdin.write(stdin.encode())
                        proc.stdin.close()
                    proc.wait(timeout=self.limits.timeout_s)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    self._kill_group(proc)
                    proc.wait()
            duration = (time.monotonic() - start) * 1000
            stdout, t1 = self._read_capped(out_path)
            stderr, t2 = self._read_capped(err_path)
            rc = proc.returncode
            sig = signal.Signals(-rc).name if rc is not None and rc < 0 else None
            return SandboxResult(exit_code=rc, stdout=stdout, stderr=stderr, timed_out=timed_out,
                                 truncated=t1 or t2, duration_ms=round(duration, 1), signal=sig,
                                 files=self._collect(work))

    def run_python(self, code: str, *, stdin: str | None = None, files: dict[str, str] | None = None) -> SandboxResult:
        """Run a Python snippet in isolated mode (-I: no user site, no PYTHON* env vars)."""
        payload = dict(files or {})
        payload["main.py"] = code
        return self.run([self.python_executable, "-I", "-B", "main.py"], stdin=stdin, files=payload)

    # ------------------------------------------------------------------ internals
    def _env(self, work: Path) -> dict[str, str]:
        env = {k: os.environ[k] for k in self.env_allowlist if k in os.environ}
        env.update({"HOME": str(work), "TMPDIR": str(work), "PYTHONDONTWRITEBYTECODE": "1"})
        return env  # nothing else: no API keys, no cloud credentials, no proxy tokens

    def _apply_limits(self) -> None:  # runs in the child between fork and exec
        lim = self.limits

        def setl(which: int, value: int) -> None:
            try:
                resource.setrlimit(which, (value, value))
            except (ValueError, OSError):
                pass  # not supported on this kernel; the wall-clock timeout still applies

        setl(resource.RLIMIT_CPU, lim.cpu_s)
        setl(resource.RLIMIT_FSIZE, lim.file_size_mb * 1024 * 1024)
        setl(resource.RLIMIT_NOFILE, lim.open_files)
        setl(resource.RLIMIT_CORE, 0)
        if hasattr(resource, "RLIMIT_AS"):
            setl(resource.RLIMIT_AS, lim.memory_mb * 1024 * 1024)

    @staticmethod
    def _kill_group(proc: subprocess.Popen[Any]) -> None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()

    def _read_capped(self, path: Path) -> tuple[str, bool]:
        cap = self.limits.max_output_bytes
        with path.open("rb") as f:
            data = f.read(cap + 1)
        return data[:cap].decode("utf-8", errors="replace"), len(data) > cap

    def _collect(self, work: Path) -> dict[str, str]:
        out: dict[str, str] = {}
        for pattern in self.collect_outputs:
            for p in sorted(work.rglob(pattern)):
                if p.is_file() and p.stat().st_size <= 16_000:
                    out[str(p.relative_to(work))] = p.read_text(encoding="utf-8", errors="replace")
        return out


class RunPythonArgs(BaseModel):
    code: str = Field(description="A complete Python 3 program. Print results to stdout. No network access.",
                      max_length=20_000)
    stdin: str | None = Field(default=None, description="Optional text passed on standard input.", max_length=100_000)


def make_python_tool(runner: SandboxRunner, *, name: str = "run_python",
                     required_permission: str | None = "code:execute") -> Tool:
    """Expose the sandbox as a tool. Side effects stay inside a deleted temp dir, so the class
    is READ for the outside world; the permission still gates who may run code at all."""

    def handler(args: RunPythonArgs, ex: Any) -> dict[str, Any]:
        r = runner.run_python(args.code, stdin=args.stdin)
        return {"exit_code": r.exit_code, "stdout": r.stdout, "stderr": r.stderr[-4000:],
                "timed_out": r.timed_out, "truncated": r.truncated, "files": r.files}

    return Tool(name=name, description="Run a short Python program in an isolated sandbox and return its output. "
                "Use for calculations and data transformations, not for network access.",
                args_model=RunPythonArgs, handler=handler, side_effect=SideEffect.READ,
                required_permission=required_permission, timeout_s=runner.limits.timeout_s + 5,
                max_result_chars=8000, tags=frozenset({"compute"}))


__all__ = ["SandboxLimits", "SandboxResult", "SandboxError", "SandboxRunner", "RunPythonArgs",
           "make_python_tool", "SAFE_ENV_KEYS"]
