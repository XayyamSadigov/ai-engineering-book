# path: book/projects/aie_core/tests/_helpers.py
"""Shared fixtures. Every test runs offline: HTTP goes through httpx.MockTransport, time is faked."""
from __future__ import annotations

import json
from typing import Any, Callable

import httpx


def sse_body(events: list[tuple[str | None, Any]]) -> bytes:
    """Build an SSE response body from (event_name, payload) pairs. Payload may be a dict or raw str."""
    lines: list[str] = []
    for name, payload in events:
        if name:
            lines.append(f"event: {name}")
        data = payload if isinstance(payload, str) else json.dumps(payload)
        lines.append(f"data: {data}")
        lines.append("")
    return ("\n".join(lines) + "\n").encode("utf-8")


class RecordingTransport(httpx.MockTransport):
    """MockTransport that keeps every request so tests can assert on the wire payload."""

    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        self.requests: list[httpx.Request] = []

        def wrapped(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return handler(request)

        super().__init__(wrapped)

    def last_json(self) -> dict[str, Any]:
        return json.loads(self.requests[-1].content)


class FakeClock:
    """Manual monotonic clock; `sleep` advances it instead of waiting."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    async def asleep(self, seconds: float) -> None:
        self.sleep(seconds)

    def advance(self, seconds: float) -> None:
        self.now += seconds

