# path: book/projects/p3-rag-assistant/tests/conftest.py
"""Offline fixtures: in-memory stores, FakeLLM with the extractive handler, vocabulary embeddings.

Every test builds its own container over a private copy of shared-data/docs, so tests that
edit, delete or re-ACL documents never affect each other or the shared fixture files.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

import pytest
from aie_core.embeddings import FakeEmbeddings
from aie_core.llm.providers import FakeLLM
from aie_core.observability import InMemoryTracer
from ragkit.retrieval import Principal

from rag_assistant.adapters.llm import corpus_vocabulary, extractive_handler
from rag_assistant.config import DEFAULT_DOCS_DIR, AssistantSettings
from rag_assistant.wiring import Container, build_container

PTO_Q = "How many unused PTO days can I carry over into next year, and by when must I use them?"


class CountingEmbeddings(FakeEmbeddings):
    """FakeEmbeddings that counts texts sent to the 'provider' (what an embedding bill counts)."""

    @property
    def texts_embedded(self) -> int:
        return sum(len(c) for c in self.calls)


@pytest.fixture(scope="session")
def vocabulary() -> list[str]:
    return corpus_vocabulary(DEFAULT_DOCS_DIR)


@pytest.fixture
def docs_dir(tmp_path: Path) -> Path:
    target = tmp_path / "docs"
    shutil.copytree(DEFAULT_DOCS_DIR, target)
    return target


@pytest.fixture
def make_container(docs_dir: Path, vocabulary: list[str]) -> Callable[..., Container]:
    def factory(*, ingest: bool = True, llm=None, **overrides) -> Container:  # type: ignore[no-untyped-def]
        settings = AssistantSettings(docs_dir=docs_dir, **overrides)
        container = build_container(settings, llm=llm or FakeLLM(handler=extractive_handler()),
                                    embeddings=CountingEmbeddings(vocabulary=vocabulary), tracer=InMemoryTracer())
        if ingest:
            container.ingestion.sync("folder")
            container.drain()
        return container

    return factory


@pytest.fixture
def container(make_container) -> Container:  # type: ignore[no-untyped-def]
    return make_container()


@pytest.fixture
def employee() -> Principal:
    return Principal(user_id="emp-1", tenant="retail", groups=["all"])


@pytest.fixture
def logistics_employee() -> Principal:
    return Principal(user_id="emp-2", tenant="logistics", groups=["all", "logistics"])


@pytest.fixture
def oncall() -> Principal:
    return Principal(user_id="oncall-1", tenant="retail", groups=["all", "it-oncall"])


def inner_embeddings(c: Container) -> CountingEmbeddings:
    return c.embeddings.inner  # type: ignore[return-value]


def edit_doc(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text, f"{old!r} not in {path.name}"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
