# path: book/projects/examples/ch32/northwind_triage/record_replay.py
"""Record real provider HTTP exchanges once; replay them in tests forever after.

RecordReplayTransport is an httpx transport, so it plugs into any adapter that accepts
`transport=` (aie_core's OpenAICompatibleClient and AnthropicClient do). It sits below the
adapter, which means replayed tests exercise the real request encoding, response decoding,
and error mapping, the code a FakeLLM skips.

Modes:
  replay  serve from the cassette; a request with no recording raises CassetteMiss
  record  forward to the upstream transport and append to the cassette
  auto    replay when recorded, otherwise record (convenient locally, never in CI)

Requests are matched by method, path, and a hash of the canonical JSON body with volatile
fields removed. Credentials in headers are never written. A `scrub` callback can remove PII
from bodies before they reach disk; cassettes are committed to git and reviewed like code.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Callable, Literal

import httpx

Mode = Literal["replay", "record", "auto"]
SENSITIVE_HEADERS = frozenset({"authorization", "x-api-key", "api-key", "cookie", "set-cookie",
                               "openai-organization", "x-request-id"})
DROP_RESPONSE_HEADERS = frozenset({"content-encoding", "content-length", "transfer-encoding",
                                   "date", "connection"})


class CassetteMiss(LookupError):
    pass


def mode_from_env(default: Mode = "replay") -> Mode:
    value = os.environ.get("CASSETTE_MODE", default)
    if value not in ("replay", "record", "auto"):
        raise ValueError(f"CASSETTE_MODE must be replay|record|auto, got {value!r}")
    return value  # type: ignore[return-value]


class RecordReplayTransport(httpx.BaseTransport):
    def __init__(
        self,
        cassette_path: str | Path,
        mode: Mode = "replay",
        upstream: httpx.BaseTransport | None = None,
        ignore_body_fields: tuple[str, ...] = ("user", "metadata", "stream_options"),
        scrub: Callable[[str], str] | None = None,
    ) -> None:
        self.path = Path(cassette_path)
        self.mode = mode
        self.upstream = upstream
        self.ignore_body_fields = ignore_body_fields
        self.scrub = scrub or (lambda s: s)
        self._lock = threading.Lock()
        self._cursor: dict[str, int] = {}
        self._entries: dict[str, list[dict[str, Any]]] = self._load()
        self._rerecorded: set[str] = set()   # keys replaced in this session (record mode)

    # ------------------------------------------------------------------ storage
    def _load(self) -> dict[str, list[dict[str, Any]]]:
        if not self.path.exists():
            return {}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return {k: v for k, v in data.get("interactions", {}).items()}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "interactions": self._entries}
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                             encoding="utf-8")

    # ------------------------------------------------------------------ matching
    def _canonical_body(self, request: httpx.Request) -> str:
        raw = request.content.decode("utf-8", errors="replace") if request.content else ""
        try:
            body = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            return raw
        if isinstance(body, dict):
            body = {k: v for k, v in body.items() if k not in self.ignore_body_fields}
        return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def key_for(self, request: httpx.Request) -> str:
        material = f"{request.method}\n{request.url.path}\n{self._canonical_body(request)}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]

    # ------------------------------------------------------------------ transport
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        key = self.key_for(request)
        with self._lock:
            recorded = self._entries.get(key)
            if recorded and self.mode in ("replay", "auto"):
                i = self._cursor.get(key, 0)
                self._cursor[key] = i + 1
                return self._to_response(recorded[min(i, len(recorded) - 1)], request)
        if self.mode == "replay":
            raise CassetteMiss(
                f"no recording for {request.method} {request.url.path} (key {key}) in {self.path}. "
                f"The request changed (prompt, model, parameters) or was never recorded. "
                f"Re-record deliberately with CASSETTE_MODE=record and review the diff.")
        return self._record(key, request)

    def _record(self, key: str, request: httpx.Request) -> httpx.Response:
        upstream = self.upstream or httpx.HTTPTransport()
        response = upstream.handle_request(request)
        content = response.read()
        entry = {
            "request": {
                "method": request.method,
                "path": request.url.path,
                "body": self.scrub(self._canonical_body(request)),
            },
            "response": {
                "status": response.status_code,
                "headers": {k: v for k, v in response.headers.items()
                            if k.lower() not in SENSITIVE_HEADERS | DROP_RESPONSE_HEADERS},
                "body": self.scrub(content.decode("utf-8", errors="replace")),
            },
        }
        with self._lock:
            if self.mode == "record" and key not in self._rerecorded:
                # Re-recording replaces the old exchange; appending would let replay keep serving it.
                self._entries[key] = []
                self._rerecorded.add(key)
            self._entries.setdefault(key, []).append(entry)
            self._save()
        # read() already decoded gzip/br, so the encoding headers no longer describe the body.
        headers = {k: v for k, v in response.headers.items() if k.lower() not in DROP_RESPONSE_HEADERS}
        return httpx.Response(response.status_code, headers=headers, content=content, request=request)

    @staticmethod
    def _to_response(entry: dict[str, Any], request: httpx.Request) -> httpx.Response:
        r = entry["response"]
        return httpx.Response(r["status"], headers=r["headers"], content=r["body"].encode("utf-8"),
                              request=request)

    # ------------------------------------------------------------------ audit
    def unused_keys(self) -> list[str]:
        """Recordings no test replayed: stale fixtures that should be deleted."""
        return sorted(k for k in self._entries if k not in self._cursor)
