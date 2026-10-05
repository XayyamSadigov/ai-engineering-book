# path: book/projects/p3-rag-assistant/rag_assistant/ingestion/registry.py
"""The document registry: the system of record for what is indexed, where, and since when.

Indexes are caches of the registry, not the other way round. The registry answers questions
the indexes cannot: which documents exist and in which state (active, deleting, deleted),
which chunk ids each one produced, which index versions hold it, when a change was observed
and when it became searchable (freshness), and the generation counter of every tenant scope.

Deleted documents keep their row as a tombstone (`status="deleted"`, `deleted_seq`). The
tombstone is what stops resurrection: a stale job, a retry, or a folder sync that still sees
the file all compare against it.
"""
from __future__ import annotations

import json
import threading
from typing import Iterable, Protocol, runtime_checkable

from ..domain.models import DocRecord, DocStatus


@runtime_checkable
class DocumentRegistry(Protocol):
    def next_seq(self) -> int: ...
    def get(self, doc_id: str) -> DocRecord | None: ...
    def put(self, record: DocRecord) -> None: ...
    def all(self, status: DocStatus | None = None) -> list[DocRecord]: ...
    def bump(self, scopes: Iterable[str]) -> dict[str, int]: ...
    def generations(self, scopes: Iterable[str] | None = None) -> dict[str, int]: ...
    def get_state(self, key: str) -> str | None: ...
    def set_state(self, key: str, value: str | None) -> None: ...


class InMemoryRegistry:
    def __init__(self) -> None:
        self._docs: dict[str, DocRecord] = {}
        self._gens: dict[str, int] = {}
        self._state: dict[str, str] = {}
        self._seq = 0
        self._lock = threading.RLock()

    def next_seq(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

    def get(self, doc_id: str) -> DocRecord | None:
        with self._lock:
            rec = self._docs.get(doc_id)
            return rec.model_copy(deep=True) if rec else None

    def put(self, record: DocRecord) -> None:
        with self._lock:
            self._docs[record.doc_id] = record.model_copy(deep=True)

    def all(self, status: DocStatus | None = None) -> list[DocRecord]:
        with self._lock:
            return [r.model_copy(deep=True) for r in self._docs.values() if status is None or r.status == status]

    def bump(self, scopes: Iterable[str]) -> dict[str, int]:
        with self._lock:
            for s in set(scopes):
                self._gens[s] = self._gens.get(s, 0) + 1
            return dict(self._gens)

    def generations(self, scopes: Iterable[str] | None = None) -> dict[str, int]:
        with self._lock:
            if scopes is None:
                return dict(self._gens)
            return {s: self._gens.get(s, 0) for s in scopes}

    def get_state(self, key: str) -> str | None:
        with self._lock:
            return self._state.get(key)

    def set_state(self, key: str, value: str | None) -> None:
        with self._lock:
            if value is None:
                self._state.pop(key, None)
            else:
                self._state[key] = value


class SqlRegistry:
    """The same contract on any SQLAlchemy database (PostgreSQL in Docker Compose, SQLite in tests).

    Counters use `UPDATE ... SET value = value + 1 RETURNING value`, which is atomic in both,
    so several workers can bump generations and sequence numbers concurrently.
    """

    def __init__(self, url: str) -> None:
        from sqlalchemy import BigInteger, Column, Float, MetaData, String, Table, Text, create_engine

        if url.startswith("postgresql://"):
            url = "postgresql+psycopg://" + url[len("postgresql://"):]
        self.engine = create_engine(url, future=True, pool_pre_ping=True)
        md = MetaData()
        self.docs = Table(
            "rag_documents", md,
            Column("doc_id", String(256), primary_key=True),
            Column("tenant", String(64), nullable=False, index=True),
            Column("status", String(16), nullable=False, index=True),
            Column("updated_at", Float, nullable=False),
            Column("record", Text, nullable=False),
        )
        self.counters = Table("rag_counters", md, Column("name", String(128), primary_key=True),
                              Column("value", BigInteger, nullable=False))
        self.state = Table("rag_state", md, Column("key", String(128), primary_key=True),
                           Column("value", Text, nullable=True))
        md.create_all(self.engine)

    # ------------------------------------------------------------------ counters
    def _incr(self, conn, name: str) -> int:  # type: ignore[no-untyped-def]
        from sqlalchemy import insert, select, update

        row = conn.execute(update(self.counters).where(self.counters.c.name == name)
                           .values(value=self.counters.c.value + 1).returning(self.counters.c.value)).first()
        if row is not None:
            return int(row[0])
        try:
            with conn.begin_nested():
                conn.execute(insert(self.counters).values(name=name, value=1))
            return 1
        except Exception:  # another writer inserted first: increment theirs
            conn.execute(update(self.counters).where(self.counters.c.name == name)
                         .values(value=self.counters.c.value + 1))
            return int(conn.execute(select(self.counters.c.value).where(self.counters.c.name == name)).scalar_one())

    def next_seq(self) -> int:
        with self.engine.begin() as conn:
            return self._incr(conn, "seq")

    def bump(self, scopes: Iterable[str]) -> dict[str, int]:
        with self.engine.begin() as conn:
            for s in sorted(set(scopes)):
                self._incr(conn, f"gen:{s}")
        return self.generations()

    def generations(self, scopes: Iterable[str] | None = None) -> dict[str, int]:
        from sqlalchemy import select

        with self.engine.connect() as conn:
            rows = conn.execute(select(self.counters.c.name, self.counters.c.value)
                                .where(self.counters.c.name.like("gen:%"))).all()
        gens = {name[4:]: int(value) for name, value in rows}
        if scopes is None:
            return gens
        return {s: gens.get(s, 0) for s in scopes}

    # ------------------------------------------------------------------ documents
    def get(self, doc_id: str) -> DocRecord | None:
        from sqlalchemy import select

        with self.engine.connect() as conn:
            raw = conn.execute(select(self.docs.c.record).where(self.docs.c.doc_id == doc_id)).scalar_one_or_none()
        return DocRecord.model_validate_json(raw) if raw else None

    def put(self, record: DocRecord) -> None:
        from sqlalchemy import delete, insert

        values = dict(doc_id=record.doc_id, tenant=record.tenant, status=record.status,
                      updated_at=record.indexed_at or record.submitted_at, record=record.model_dump_json())
        with self.engine.begin() as conn:  # delete+insert in one transaction: portable upsert
            conn.execute(delete(self.docs).where(self.docs.c.doc_id == record.doc_id))
            conn.execute(insert(self.docs).values(**values))

    def all(self, status: DocStatus | None = None) -> list[DocRecord]:
        from sqlalchemy import select

        q = select(self.docs.c.record)
        if status is not None:
            q = q.where(self.docs.c.status == status)
        with self.engine.connect() as conn:
            return [DocRecord.model_validate_json(r) for (r,) in conn.execute(q).all()]

    # ------------------------------------------------------------------ state
    def get_state(self, key: str) -> str | None:
        from sqlalchemy import select

        with self.engine.connect() as conn:
            return conn.execute(select(self.state.c.value).where(self.state.c.key == key)).scalar_one_or_none()

    def set_state(self, key: str, value: str | None) -> None:
        from sqlalchemy import delete, insert

        with self.engine.begin() as conn:
            conn.execute(delete(self.state).where(self.state.c.key == key))
            if value is not None:
                conn.execute(insert(self.state).values(key=key, value=value))


def state_json(registry: DocumentRegistry, key: str, default: object = None) -> object:
    raw = registry.get_state(key)
    return json.loads(raw) if raw is not None else default


__all__ = ["DocumentRegistry", "InMemoryRegistry", "SqlRegistry", "state_json"]
