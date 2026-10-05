# path: book/projects/examples/ch18/mcp_client.py
"""A minimal MCP-style client that spawns a server as a subprocess and speaks
JSON-RPC 2.0 over its stdin/stdout.

The client is protocol plumbing only. It does not decide which tools a model may see or
call; that is the host's job (host_adapter.py). It keeps an audit list of every message
it sent, which the security tests use to prove that a denied call never left the host.
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import jsonrpc as rpc  # noqa: E402

CLIENT_SUPPORTED_VERSIONS: tuple[str, ...] = ("2025-06-18",)  # illustrative, see server


class McpClientError(Exception):
    """The server answered with a JSON-RPC error object."""

    def __init__(self, code: int, message: str, data: Any | None = None) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.data = data


class McpTimeout(McpClientError):
    def __init__(self, method: str, timeout_s: float) -> None:
        super().__init__(-1, f"timeout after {timeout_s}s waiting for {method}")


class ProtocolMismatch(McpClientError):
    def __init__(self, offered: str) -> None:
        super().__init__(-2, f"server chose unsupported protocol version {offered!r}")


class StdioMcpClient:
    def __init__(
        self,
        command: list[str],
        *,
        timeout_s: float = 5.0,
        client_name: str = "northwind-assist",
        client_version: str = "0.1.0",
        supported_versions: tuple[str, ...] = CLIENT_SUPPORTED_VERSIONS,
        env: dict[str, str] | None = None,
    ) -> None:
        self.command = command
        self.timeout_s = timeout_s
        self.client_info = {"name": client_name, "version": client_version}
        self.supported_versions = supported_versions
        # Credential scoping starts here: the child gets an explicit environment, not ours.
        self.env = {"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8", **(env or {})}
        self.sent: list[dict[str, Any]] = []
        self.notifications: list[dict[str, Any]] = []
        self.stderr_lines: list[str] = []  # server logs; drained so a chatty server cannot block
        self.server_info: dict[str, Any] = {}
        self.server_capabilities: dict[str, Any] = {}
        self.protocol_version: str | None = None
        self.instructions: str | None = None
        self._next_id = 0
        self._inbox: dict[int | str | None, queue.Queue[dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None

    # ------------------------------------------------------------ process lifecycle
    def start(self) -> "StdioMcpClient":
        self._proc = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=self.env,
            bufsize=1,
        )
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._stderr_reader = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_reader.start()
        return self

    def close(self) -> None:
        if self._proc is None:
            return
        if self._proc.stdin:
            self._proc.stdin.close()  # EOF on stdin is the polite shutdown signal for stdio servers
        try:
            self._proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        for thread in (self._reader, self._stderr_reader):
            if thread:
                thread.join(timeout=1)  # readers end at EOF once the process has exited
        self._proc = None

    def __enter__(self) -> "StdioMcpClient":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _read_loop(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            try:
                msg = rpc.decode(line)
            except rpc.JsonRpcError:
                continue  # a non-protocol line on stdout: a server bug; drop it
            if rpc.is_response(msg):
                self._box(msg.get("id")).put(msg)
            else:
                self.notifications.append(msg)  # notifications and server-initiated requests

    def _drain_stderr(self) -> None:
        assert self._proc and self._proc.stderr
        for line in self._proc.stderr:
            self.stderr_lines.append(line.rstrip("\n"))

    def _box(self, id_: int | str | None) -> queue.Queue[dict[str, Any]]:
        with self._lock:
            return self._inbox.setdefault(id_, queue.Queue())

    # ------------------------------------------------------------ raw messaging
    def _send(self, msg: dict[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("client not started")
        self.sent.append(msg)
        self._proc.stdin.write(rpc.encode(msg))
        self._proc.stdin.flush()

    def _wait(self, id_: int | str | None, method: str) -> dict[str, Any]:
        try:
            return self._box(id_).get(timeout=self.timeout_s)
        except queue.Empty as exc:
            raise McpTimeout(method, self.timeout_s) from exc

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            self._next_id += 1
            id_ = self._next_id
        self._send(rpc.request(id_, method, params))
        msg = self._wait(id_, method)
        if "error" in msg:
            err = msg["error"]
            raise McpClientError(err.get("code", 0), err.get("message", ""), err.get("data"))
        return msg["result"]

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self._send(rpc.notification(method, params))

    def send_raw_line(self, line: str) -> dict[str, Any]:
        """Test helper: write an arbitrary line and wait for the id-less error reply."""
        assert self._proc and self._proc.stdin
        self._proc.stdin.write(line if line.endswith("\n") else line + "\n")
        self._proc.stdin.flush()
        return self._wait(None, "raw")

    # ------------------------------------------------------------ protocol operations
    def initialize(self, requested_version: str | None = None) -> dict[str, Any]:
        """Capability exchange. Rejects a server that picks a version we do not speak."""
        result = self.request(
            "initialize",
            {
                "protocolVersion": requested_version or self.supported_versions[0],
                "capabilities": {},  # this client offers no optional features to the server
                "clientInfo": self.client_info,
            },
        )
        chosen = result.get("protocolVersion", "")
        if chosen not in self.supported_versions:
            raise ProtocolMismatch(chosen)
        self.protocol_version = chosen
        self.server_capabilities = result.get("capabilities", {})
        self.server_info = result.get("serverInfo", {})
        self.instructions = result.get("instructions")
        self.notify("notifications/initialized")
        return result

    def supports(self, feature: str) -> bool:
        """Only call a feature family the server declared during the capability exchange."""
        return feature in self.server_capabilities

    def _list_all(self, method: str, key: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            page = self.request(method, {"cursor": cursor} if cursor else {})
            items.extend(page.get(key, []))
            cursor = page.get("nextCursor")
            if not cursor:
                return items

    def list_tools(self) -> list[dict[str, Any]]:
        return self._list_all("tools/list", "tools")

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.request("tools/call", {"name": name, "arguments": arguments})

    def list_resources(self) -> list[dict[str, Any]]:
        return self._list_all("resources/list", "resources")

    def read_resource(self, uri: str) -> dict[str, Any]:
        return self.request("resources/read", {"uri": uri})

    def list_prompts(self) -> list[dict[str, Any]]:
        return self._list_all("prompts/list", "prompts")

    def get_prompt(self, name: str, arguments: dict[str, str] | None = None) -> dict[str, Any]:
        return self.request("prompts/get", {"name": name, "arguments": arguments or {}})


def northwind_command(tenant: str, *extra: str) -> list[str]:
    """Command line that launches this chapter's server with the current interpreter."""
    return [sys.executable, str(HERE / "northwind_server.py"), "--tenant", tenant, *extra]


__all__ = [
    "StdioMcpClient",
    "McpClientError",
    "McpTimeout",
    "ProtocolMismatch",
    "CLIENT_SUPPORTED_VERSIONS",
    "northwind_command",
]
