# path: book/projects/memorykit/tests/conftest.py
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aie_core.embeddings import FakeEmbeddings  # noqa: E402

from memorykit import InMemoryStore, Owner, SQLiteStore  # noqa: E402

VOCAB = [
    "vpn", "laptop", "docking", "station", "spanish", "language", "prefers", "warehouse", "office",
    "printer", "label", "refund", "pos", "register", "adapter", "restart", "payment", "outage",
    "berlin", "lisbon", "timezone", "manager", "night", "shift", "scanner", "replacement", "keyboard",
]


class FakeClock:
    def __init__(self, start: datetime | None = None) -> None:
        self.now = start or datetime(2026, 9, 1, 9, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs: float) -> None:
        self.now = self.now + timedelta(**kwargs)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture(params=["memory", "sqlite"])
def store(request: pytest.FixtureRequest):
    return InMemoryStore() if request.param == "memory" else SQLiteStore(":memory:")


@pytest.fixture
def embedder() -> FakeEmbeddings:
    return FakeEmbeddings(vocabulary=VOCAB)


@pytest.fixture
def ana() -> Owner:
    return Owner(tenant="retail", user="ana")


@pytest.fixture
def ben() -> Owner:
    return Owner(tenant="retail", user="ben")


@pytest.fixture
def ana_logistics() -> Owner:
    # Same user id in another tenant: must still be a different owner.
    return Owner(tenant="logistics", user="ana")
