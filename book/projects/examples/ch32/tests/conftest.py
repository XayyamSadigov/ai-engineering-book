# path: book/projects/examples/ch32/tests/conftest.py
"""Shared fixtures: a fake OpenAI-dialect upstream, an in-memory tracer, a clean environment."""
from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from aie_core.observability import InMemoryTracer

from northwind_triage.domain import Ticket

TRIAGE_ENV_PREFIXES = ("TRIAGE_", "LLM_", "OPENAI_", "ANTHROPIC_", "EMBEDDING_", "TRACE_", "CASSETTE_")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Tests never inherit the developer's real keys or provider settings, and never read
    a stray .env file from the working directory."""
    import os

    for key in list(os.environ):
        if key.startswith(TRIAGE_ENV_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def tracer() -> InMemoryTracer:
    return InMemoryTracer()


@pytest.fixture
def ticket() -> Ticket:
    return Ticket(id="TCK-T-1", tenant="retail", subject="Register 3 declines every card",
                  body="Store 0412. Since 08:00 all cards are declined on register 3. Queue building.")


class FakeOpenAIUpstream:
    """Stands in for a provider during RECORDING. It answers chat-completion requests in the
    OpenAI wire format and counts calls, so tests can prove replay never reaches it."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0
        self.seen_auth: list[str | None] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        self.seen_auth.append(request.headers.get("authorization"))
        body: dict[str, Any] = json.loads(request.content)
        return httpx.Response(200, json={
            "id": f"chatcmpl-{self.calls}",
            "object": "chat.completion",
            "model": body["model"] + "-2026-01-15",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": self.content}}],
            "usage": {"prompt_tokens": 180, "completion_tokens": 30, "total_tokens": 210},
        }, headers={"x-request-id": "req_secret_123", "openai-organization": "org-secret"})

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)
