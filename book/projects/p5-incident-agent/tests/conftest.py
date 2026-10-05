# path: book/projects/p5-incident-agent/tests/conftest.py
"""Fixtures: one knowledge base per session (chunking is the slow part), fresh everything else."""
from __future__ import annotations

from typing import Any, Callable

import pytest

from agentkit import InMemoryEventStore
from incident_agent.adapters.channel import InMemoryChannel
from incident_agent.adapters.corpus import KnowledgeBase
from incident_agent.adapters.scripted import scripted_llm
from incident_agent.adapters.store import InMemoryInvestigationStore
from incident_agent.adapters.telemetry import Telemetry
from incident_agent.config import P5Settings
from incident_agent.service import IncidentService

MAIN = "ALR-2026-0914-01"        # Trackline latency; cause is a migration on a dependency
SECOND = "ALR-2026-0914-02"      # RoutePilot latency; cause is its own release


@pytest.fixture(scope="session")
def settings(tmp_path_factory: pytest.TempPathFactory) -> P5Settings:
    return P5Settings(state_dir=tmp_path_factory.mktemp("state"))


@pytest.fixture(scope="session")
def kb(settings: P5Settings) -> KnowledgeBase:
    return KnowledgeBase(settings.shared_data_dir / "docs", chunk_tokens=settings.chunk_tokens)


@pytest.fixture
def telemetry(settings: P5Settings) -> Telemetry:
    return Telemetry(settings.data_dir)


@pytest.fixture
def make_service(settings: P5Settings, kb: KnowledgeBase, telemetry: Telemetry) -> Callable[..., IncidentService]:
    def make(llm: Any = None, **overrides: Any) -> IncidentService:
        s = settings.model_copy(update=overrides)
        return IncidentService(s, llm=llm or scripted_llm(), kb=kb, telemetry=telemetry, channel=InMemoryChannel(),
                               store=InMemoryInvestigationStore(), event_store=InMemoryEventStore())
    return make


@pytest.fixture
def service(make_service: Callable[..., IncidentService]) -> IncidentService:
    return make_service()


ONCALL = {"user": "oncall-logistics", "tenant": "logistics", "groups": ["all", "it-oncall"]}
