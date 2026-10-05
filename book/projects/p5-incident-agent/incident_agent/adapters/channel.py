# path: book/projects/p5-incident-agent/incident_agent/adapters/channel.py
"""Where approved reports are posted. Posting is EXTERNAL: it leaves our boundary and cannot be
unsent, so it is idempotent by key: the same key always returns the same message."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Protocol


class Channel(Protocol):
    def post(self, channel: str, title: str, body: str, *, idempotency_key: str) -> str: ...

    def messages(self) -> list[dict[str, Any]]: ...


class InMemoryChannel:
    def __init__(self) -> None:
        self._messages: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def post(self, channel: str, title: str, body: str, *, idempotency_key: str) -> str:
        with self._lock:
            for m in self._messages:
                if m["idempotency_key"] == idempotency_key:
                    return str(m["id"])
            msg = {"id": f"MSG-{len(self._messages) + 1:05d}", "channel": channel, "title": title, "body": body,
                   "idempotency_key": idempotency_key}
            self._messages.append(msg)
            return str(msg["id"])

    def messages(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._messages)


class JsonlChannel(InMemoryChannel):
    """Same semantics, persisted, so a second process sees what the first one posted."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            self._messages = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    def post(self, channel: str, title: str, body: str, *, idempotency_key: str) -> str:
        before = len(self._messages)
        msg_id = super().post(channel, title, body, idempotency_key=idempotency_key)
        if len(self._messages) > before:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(self._messages[-1]) + "\n")
        return msg_id


__all__ = ["Channel", "InMemoryChannel", "JsonlChannel"]
