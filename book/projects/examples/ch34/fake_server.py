# path: book/projects/examples/ch34/fake_server.py
"""An in-process fake of an OpenAI-compatible streaming server for offline tests.

It is deliberately simple but has the two properties that make load tests interesting:
a per-token decode delay (so TTFT < E2E) and a capacity limit (so queue time, and therefore
TTFT, grows once more requests are active than the fake "batch" can hold). Plug it into an
``httpx.AsyncClient`` through ``httpx.MockTransport``.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx


@dataclass
class FakeServer:
    capacity: int = 4  # concurrent sequences the fake "GPU" decodes without queueing
    prefill_s: float = 0.01  # fixed prompt-processing delay
    tpot_s: float = 0.002  # per-token decode delay
    queue_slot_s: float = 0.02  # extra wait per request above capacity (crude queue model)
    fail_above: int | None = None  # return HTTP 429 when more than this many are active
    active: int = 0
    seen: list[dict] = field(default_factory=list)
    peak_active: int = 0

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    async def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        self.seen.append(body)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "fake-model"}]})
        if not request.url.path.endswith("/chat/completions"):
            return httpx.Response(404, json={"error": "unknown path"})
        if self.fail_above is not None and self.active >= self.fail_above:
            return httpx.Response(429, json={"error": {"message": "overloaded"}})
        if not body.get("stream"):
            return httpx.Response(400, json={"error": {"message": "this fake only streams"}})
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_SSEStream(self, int(body.get("max_tokens", 16)), body.get("model", "fake-model")),
        )


class _SSEStream(httpx.AsyncByteStream):
    def __init__(self, server: FakeServer, n_tokens: int, model: str) -> None:
        self.server, self.n_tokens, self.model = server, n_tokens, model

    async def __aiter__(self) -> AsyncIterator[bytes]:
        s = self.server
        s.active += 1
        s.peak_active = max(s.peak_active, s.active)
        try:
            over = max(0, s.active - s.capacity)
            await asyncio.sleep(s.prefill_s + over * s.queue_slot_s)  # queue + prefill => TTFT
            created = int(time.time())
            for i in range(self.n_tokens):
                if i:
                    await asyncio.sleep(s.tpot_s)
                yield _chunk(self.model, created, f"tok{i} ")
            usage = {"prompt_tokens": 10, "completion_tokens": self.n_tokens, "total_tokens": 10 + self.n_tokens}
            yield _chunk(self.model, created, None, finish="stop", usage=usage)
            yield b"data: [DONE]\n\n"
        finally:
            s.active -= 1

    async def aclose(self) -> None:  # pragma: no cover - nothing to release
        return None


def _chunk(model: str, created: int, text: str | None, finish: str | None = None, usage: dict | None = None) -> bytes:
    delta = {"content": text} if text is not None else {}
    payload = {
        "id": "chatcmpl-fake", "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if usage is not None:
        payload["usage"] = usage
    return f"data: {json.dumps(payload)}\n\n".encode()
