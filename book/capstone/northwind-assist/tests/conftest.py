# path: book/capstone/northwind-assist/tests/conftest.py
"""Offline fixtures: the demo FakeLLM, vocabulary FakeEmbeddings, in-memory stores and tracer."""
from __future__ import annotations

import json
import os
from typing import Any

import pytest

for _k in [k for k in os.environ if k.startswith("NA_")]:
    del os.environ[_k]
os.environ.update({"LLM_PROVIDER": "fake", "EMBEDDING_PROVIDER": "fake", "TRACE_SINK": "none"})

from fastapi.testclient import TestClient  # noqa: E402

from northwind_assist.api.app import create_app  # noqa: E402
from northwind_assist.config import Settings  # noqa: E402
from northwind_assist.container import Container, build_container  # noqa: E402
from northwind_assist.evaluation.suites import persona_ctx  # noqa: E402
from northwind_assist.observability.tracing import build_tracer  # noqa: E402
from northwind_assist.orchestrator import ChatRequest, TurnResult  # noqa: E402
from northwind_assist.security.auth import DEV_PERSONAS, issue_dev_token  # noqa: E402


def make(settings: Settings | None = None, **kw: Any) -> Container:
    return build_container(settings or Settings(environment="test"), tracer=build_tracer({"TRACE_SINK": "memory"}),
                           **kw)


@pytest.fixture
def container() -> Container:
    return make()


@pytest.fixture
def client(container: Container) -> TestClient:
    return TestClient(create_app(container))


def token_for(c: Container, persona: str, **override: Any) -> str:
    p = {**DEV_PERSONAS[persona], **override}
    return issue_dev_token(c.settings, sub=persona, tenant=p["tenant"], groups=p["groups"], roles=p["roles"])


def auth(c: Container, persona: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token_for(c, persona)}"}


def chat(c: Container, persona: str, message: str, *, session: str | None = "s1", idempotency_key: str | None = None,
         **req: Any) -> TurnResult:
    prepared = c.orchestrator.prepare(persona_ctx(persona), ChatRequest(message=message, session_id=session, **req),
                                      idempotency_key=idempotency_key)
    return c.orchestrator.run(prepared)


def parse_sse(text: str) -> list[dict[str, Any]]:
    events = []
    for frame in text.split("\n\n"):
        fields = dict(line.split(": ", 1) for line in frame.splitlines() if ": " in line and not line.startswith(":"))
        if "event" in fields:
            events.append({"id": int(fields["id"]), "event": fields["event"], "data": json.loads(fields["data"])})
    return events
