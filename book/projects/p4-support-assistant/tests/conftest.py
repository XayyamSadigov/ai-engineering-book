# path: book/projects/p4-support-assistant/tests/conftest.py
from __future__ import annotations

import pytest
from aie_core.llm.providers import FakeLLM
from aie_core.llm.types import ToolCall

from support_assistant.assistant import SupportAssistant
from support_assistant.config import AssistantSettings
from support_assistant.wiring import build_container


def tc(name: str, call_id: str | None = None, **args) -> ToolCall:
    return ToolCall(id=call_id or f"call_{name}", name=name, arguments=args)


@pytest.fixture
def make_assistant():
    created = []

    def factory(responses, **overrides) -> SupportAssistant:
        settings = AssistantSettings(**overrides)
        container = build_container(settings, llm=FakeLLM(responses=responses), sleep=lambda s: None)
        created.append(container)
        return SupportAssistant(container)

    yield factory
    for c in created:
        c.executor.close()
