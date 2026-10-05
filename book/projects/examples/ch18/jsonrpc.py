# path: book/projects/examples/ch18/jsonrpc.py
"""JSON-RPC 2.0 message helpers and newline-delimited framing for a stdio transport.

MCP messages are JSON-RPC 2.0 objects. Over stdio each message is one line of UTF-8 JSON
with no embedded newlines; stdout carries protocol messages only, stderr carries logs.
Nothing in this module knows about tools or resources: it is the wire layer.
"""
from __future__ import annotations

import json
from typing import Any, TextIO

JSONRPC = "2.0"

# Standard JSON-RPC 2.0 error codes.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

RequestId = int | str


class JsonRpcError(Exception):
    """A protocol-level error. The server turns it into an error response."""

    def __init__(self, code: int, message: str, data: Any | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data

    def to_dict(self) -> dict[str, Any]:
        err: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            err["data"] = self.data
        return err


def request(id_: RequestId, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    msg: dict[str, Any] = {"jsonrpc": JSONRPC, "id": id_, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def notification(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """A request without an id. The receiver must not reply."""
    msg: dict[str, Any] = {"jsonrpc": JSONRPC, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def result(id_: RequestId | None, value: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": JSONRPC, "id": id_, "result": value}


def error(id_: RequestId | None, err: JsonRpcError) -> dict[str, Any]:
    return {"jsonrpc": JSONRPC, "id": id_, "error": err.to_dict()}


def is_notification(msg: dict[str, Any]) -> bool:
    return "method" in msg and "id" not in msg


def is_response(msg: dict[str, Any]) -> bool:
    return "id" in msg and ("result" in msg or "error" in msg) and "method" not in msg


def encode(msg: dict[str, Any]) -> str:
    """One message, one line. json.dumps escapes newlines inside strings, so the frame is safe."""
    return json.dumps(msg, ensure_ascii=False, separators=(",", ":")) + "\n"


def write_message(stream: TextIO, msg: dict[str, Any]) -> None:
    stream.write(encode(msg))
    stream.flush()


def decode(line: str) -> dict[str, Any]:
    """Parse one frame. Raises JsonRpcError(PARSE_ERROR / INVALID_REQUEST) on bad input."""
    try:
        msg = json.loads(line)
    except json.JSONDecodeError as exc:
        raise JsonRpcError(PARSE_ERROR, "parse error", str(exc)) from exc
    if isinstance(msg, list):
        # JSON-RPC allows batches; this implementation (like recent MCP revisions) does not.
        raise JsonRpcError(INVALID_REQUEST, "batches are not supported")
    if not isinstance(msg, dict) or msg.get("jsonrpc") != JSONRPC:
        raise JsonRpcError(INVALID_REQUEST, "not a JSON-RPC 2.0 object")
    return msg


__all__ = [
    "JSONRPC",
    "PARSE_ERROR",
    "INVALID_REQUEST",
    "METHOD_NOT_FOUND",
    "INVALID_PARAMS",
    "INTERNAL_ERROR",
    "JsonRpcError",
    "request",
    "notification",
    "result",
    "error",
    "is_notification",
    "is_response",
    "encode",
    "write_message",
    "decode",
]
