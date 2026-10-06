# path: book/projects/memorykit/memorykit/store.py
"""Memory storage: one protocol, two implementations (in-memory and SQLite).

Rules every implementation enforces, so callers cannot forget them:
- every read and delete is scoped by tenant; there is no method that reads across tenants;
- user scope is exact: a user reads their own records, plus tenant-wide records only when asked;
- expired records are invisible to reads before any purge job has run;
- delete is hard (the row is gone) and leaves a content-free tombstone;
- delete cascades to records derived from the deleted one (provenance "mem:<id>");
- writes can be guarded by optimistic concurrency (`expected_version`).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from .models import (
    MemoryKind,
    MemoryRecord,
    MemoryStatus,
    Owner,
    Sensitivity,
    Source,
    Tombstone,
    normalize_text,
    utcnow,
)


class VersionConflict(Exception):
    """Another writer changed the record since it was read."""


def fingerprint(record: MemoryRecord, secret: bytes) -> str:
    """Keyed hash of what a memory says.

    A plain hash of a phone number can be reversed by enumerating phone numbers, so
    tombstones use an HMAC with a per-deployment secret (rotate it like any other key).
    """
    basis = f"{record.kind.value}\x00{record.key or ''}\x00{normalize_text(record.value_text())}"
    return hmac.new(secret, basis.encode("utf-8"), hashlib.sha256).hexdigest()


@runtime_checkable
class MemoryStore(Protocol):
    def put(self, record: MemoryRecord, *, expected_version: int | None = None) -> MemoryRecord: ...

    def get(self, owner: Owner, record_id: str) -> MemoryRecord | None: ...

    def query(
        self,
        owner: Owner,
        *,
        kinds: Sequence[MemoryKind] | None = None,
        key: str | None = None,
        statuses: Sequence[MemoryStatus] = (MemoryStatus.ACTIVE,),
        include_shared: bool = False,
        include_expired: bool = False,
        now: datetime | None = None,
    ) -> list[MemoryRecord]: ...

    def delete(self, owner: Owner, record_id: str, *, reason: str, now: datetime | None = None) -> list[Tombstone]: ...

    def delete_owner(self, owner: Owner) -> int: ...

    def is_suppressed(self, record: MemoryRecord) -> bool: ...

    def tombstones(self, owner: Owner) -> list[Tombstone]: ...

    def purge_expired(self, now: datetime | None = None) -> int: ...

    def export(self, owner: Owner) -> dict[str, Any]: ...


def _visible(
    r: MemoryRecord,
    owner: Owner,
    kinds: Sequence[MemoryKind] | None,
    key: str | None,
    statuses: Sequence[MemoryStatus],
    include_shared: bool,
    include_expired: bool,
    now: datetime,
) -> bool:
    if r.owner.tenant != owner.tenant:
        return False
    if r.owner.user != owner.user and not (include_shared and r.owner.user is None):
        return False
    if kinds is not None and r.kind not in kinds:
        return False
    if key is not None and r.key != key:
        return False
    if r.status not in statuses:
        return False
    if not include_expired and r.is_expired(now):
        return False
    return True


def _export_payload(owner: Owner, records: Iterable[MemoryRecord], stones: Iterable[Tombstone]) -> dict[str, Any]:
    return {
        "owner": owner.model_dump(),
        "exported_at": utcnow().isoformat(),
        # Embeddings are derived data with no meaning to a person; content, value, and provenance are what they asked for.
        "records": [r.model_dump(mode="json", exclude={"embedding", "embedding_model"}) for r in records],
        "deleted": [{"record_id": t.record_id, "kind": t.kind.value, "key": t.key, "deleted_at": t.deleted_at.isoformat(), "reason": t.reason} for t in stones],
    }


class InMemoryStore:
    """Dict-backed store for tests and single-process prototypes."""

    def __init__(self, *, fingerprint_secret: bytes = b"dev-only-secret") -> None:
        self._records: dict[str, MemoryRecord] = {}
        self._tombstones: list[Tombstone] = []
        self._secret = fingerprint_secret
        self._lock = threading.Lock()

    def put(self, record: MemoryRecord, *, expected_version: int | None = None) -> MemoryRecord:
        with self._lock:
            current = self._records.get(record.id)
            if current is not None and current.owner != record.owner:
                raise PermissionError("record id belongs to another owner")
            if expected_version is not None:
                found = current.version if current else 0
                if found != expected_version:
                    raise VersionConflict(f"{record.id}: expected version {expected_version}, found {found}")
            self._records[record.id] = record.model_copy(deep=True)
            return record

    def get(self, owner: Owner, record_id: str) -> MemoryRecord | None:
        r = self._records.get(record_id)
        if r is None or r.owner != owner:
            return None
        return r.model_copy(deep=True)

    def query(
        self,
        owner: Owner,
        *,
        kinds: Sequence[MemoryKind] | None = None,
        key: str | None = None,
        statuses: Sequence[MemoryStatus] = (MemoryStatus.ACTIVE,),
        include_shared: bool = False,
        include_expired: bool = False,
        now: datetime | None = None,
    ) -> list[MemoryRecord]:
        now = now or utcnow()
        out = [
            r.model_copy(deep=True)
            for r in self._records.values()
            if _visible(r, owner, kinds, key, statuses, include_shared, include_expired, now)
        ]
        return sorted(out, key=lambda r: (r.created_at, r.id))

    def delete(self, owner: Owner, record_id: str, *, reason: str, now: datetime | None = None) -> list[Tombstone]:
        now = now or utcnow()
        with self._lock:
            root = self._records.get(record_id)
            if root is None or root.owner != owner:
                return []
            doomed = self._collect_derived(root)
            stones = []
            for r in doomed:
                del self._records[r.id]
                stone = Tombstone(
                    record_id=r.id, owner=r.owner, kind=r.kind, key=r.key,
                    fingerprint=fingerprint(r, self._secret), deleted_at=now,
                    reason=reason if r.id == record_id else f"derived from {record_id}: {reason}",
                )
                self._tombstones.append(stone)
                stones.append(stone)
            return stones

    def _collect_derived(self, root: MemoryRecord) -> list[MemoryRecord]:
        """The root plus everything in the same tenant whose provenance points at it, transitively."""
        found = {root.id: root}
        frontier = [root.id]
        while frontier:
            parent = frontier.pop()
            for r in self._records.values():
                if r.id not in found and r.owner.tenant == root.owner.tenant and parent in r.derived_from():
                    found[r.id] = r
                    frontier.append(r.id)
        return list(found.values())

    def delete_owner(self, owner: Owner) -> int:
        """Erase everything a user owns, with the same cascade into derived records as `delete`,
        then every tombstone the erasure produced or the user owned: fingerprints are derived from
        personal data. Each root is its own transaction, so a crash mid-way is safe to retry.
        Returns the number of the user's own records erased."""
        with self._lock:
            ids = [r.id for r in self._records.values() if r.owner == owner]
        stones = [t for i in ids for t in self.delete(owner, i, reason="account erased")]
        erased_ids = {t.record_id for t in stones}
        with self._lock:
            self._tombstones = [t for t in self._tombstones if t.owner != owner and t.record_id not in erased_ids]
        return sum(1 for t in stones if t.owner == owner)

    def is_suppressed(self, record: MemoryRecord) -> bool:
        fp = fingerprint(record, self._secret)
        return any(t.owner == record.owner and t.fingerprint == fp for t in self._tombstones)

    def tombstones(self, owner: Owner) -> list[Tombstone]:
        return [t for t in self._tombstones if t.owner == owner]

    def purge_expired(self, now: datetime | None = None) -> int:
        now = now or utcnow()
        with self._lock:
            expired = [i for i, r in self._records.items() if r.is_expired(now)]
            for i in expired:
                del self._records[i]
            return len(expired)

    def export(self, owner: Owner) -> dict[str, Any]:
        records = self.query(owner, statuses=tuple(MemoryStatus), include_expired=True)
        return _export_payload(owner, records, self.tombstones(owner))


# ------------------------------------------------------------------------------------ SQLite
_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    tenant TEXT NOT NULL,
    user_id TEXT NOT NULL,          -- '' means tenant-wide
    kind TEXT NOT NULL,
    key TEXT,
    content TEXT NOT NULL,
    value_json TEXT,
    source TEXT NOT NULL,
    provenance_json TEXT NOT NULL,
    confidence REAL NOT NULL,
    salience REAL NOT NULL,
    sensitivity TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    expires_at TEXT,
    version INTEGER NOT NULL,
    embedding_json TEXT,
    embedding_model TEXT
);
CREATE INDEX IF NOT EXISTS ix_mem_scope ON memories (tenant, user_id, kind, status);
CREATE INDEX IF NOT EXISTS ix_mem_expiry ON memories (expires_at);
CREATE TABLE IF NOT EXISTS tombstones (
    record_id TEXT NOT NULL,
    tenant TEXT NOT NULL,
    user_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    key TEXT,
    fingerprint TEXT NOT NULL,
    deleted_at TEXT NOT NULL,
    reason TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_tomb_scope ON tombstones (tenant, user_id, fingerprint);
"""


def _ts(dt: datetime | None) -> str | None:
    # One fixed format in UTC so string comparison in SQL equals time comparison.
    return None if dt is None else dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")


def _dt(s: str | None) -> datetime | None:
    return None if s is None else datetime.fromisoformat(s)


class SQLiteStore:
    """Durable single-node store. The same schema maps onto PostgreSQL (add a pgvector column
    for embeddings and row-level security on `tenant` for defense in depth)."""

    _COLUMNS = (
        "id, tenant, user_id, kind, key, content, value_json, source, provenance_json, confidence, salience, "
        "sensitivity, status, created_at, updated_at, expires_at, version, embedding_json, embedding_model"
    )

    def __init__(self, path: str = ":memory:", *, fingerprint_secret: bytes = b"dev-only-secret") -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._secret = fingerprint_secret
        self._lock = threading.Lock()

    # -------------------------------------------------------------- mapping
    @staticmethod
    def _row(r: MemoryRecord) -> tuple[Any, ...]:
        return (
            r.id, r.owner.tenant, r.owner.user or "", r.kind.value, r.key, r.content,
            json.dumps(r.value), r.source.value, json.dumps(r.provenance), r.confidence, r.salience,
            r.sensitivity.value, r.status.value, _ts(r.created_at), _ts(r.updated_at), _ts(r.expires_at),
            r.version, json.dumps(r.embedding) if r.embedding is not None else None, r.embedding_model,
        )

    @staticmethod
    def _record(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            owner=Owner(tenant=row["tenant"], user=row["user_id"] or None),
            kind=MemoryKind(row["kind"]),
            key=row["key"],
            content=row["content"],
            value=json.loads(row["value_json"]) if row["value_json"] is not None else None,
            source=Source(row["source"]),
            provenance=json.loads(row["provenance_json"]),
            confidence=row["confidence"],
            salience=row["salience"],
            sensitivity=Sensitivity(row["sensitivity"]),
            status=MemoryStatus(row["status"]),
            created_at=_dt(row["created_at"]),
            updated_at=_dt(row["updated_at"]),
            expires_at=_dt(row["expires_at"]),
            version=row["version"],
            embedding=json.loads(row["embedding_json"]) if row["embedding_json"] else None,
            embedding_model=row["embedding_model"],
        )

    # -------------------------------------------------------------- protocol
    def put(self, record: MemoryRecord, *, expected_version: int | None = None) -> MemoryRecord:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                cur = self._conn.execute("SELECT tenant, user_id, version FROM memories WHERE id = ?",
                                         (record.id,)).fetchone()
                if cur is not None and (cur["tenant"], cur["user_id"]) != (record.owner.tenant, record.owner.user or ""):
                    raise PermissionError("record id belongs to another owner")
                if expected_version is not None:
                    found = cur["version"] if cur else 0
                    if found != expected_version:
                        raise VersionConflict(f"{record.id}: expected version {expected_version}, found {found}")
                placeholders = ", ".join("?" * 19)
                self._conn.execute(f"INSERT OR REPLACE INTO memories ({self._COLUMNS}) VALUES ({placeholders})", self._row(record))
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
        return record

    def get(self, owner: Owner, record_id: str) -> MemoryRecord | None:
        row = self._conn.execute(
            f"SELECT {self._COLUMNS} FROM memories WHERE id = ? AND tenant = ? AND user_id = ?",
            (record_id, owner.tenant, owner.user or ""),
        ).fetchone()
        return self._record(row) if row else None

    def query(
        self,
        owner: Owner,
        *,
        kinds: Sequence[MemoryKind] | None = None,
        key: str | None = None,
        statuses: Sequence[MemoryStatus] = (MemoryStatus.ACTIVE,),
        include_shared: bool = False,
        include_expired: bool = False,
        now: datetime | None = None,
    ) -> list[MemoryRecord]:
        now = now or utcnow()
        # The tenant predicate is not optional and not built from caller-controlled SQL.
        sql = [f"SELECT {self._COLUMNS} FROM memories WHERE tenant = ?"]
        args: list[Any] = [owner.tenant]
        if include_shared and owner.user is not None:
            sql.append("AND user_id IN (?, '')")
        else:
            sql.append("AND user_id = ?")
        args.append(owner.user or "")
        if kinds is not None:
            sql.append(f"AND kind IN ({', '.join('?' * len(kinds))})")
            args.extend(k.value for k in kinds)
        if key is not None:
            sql.append("AND key = ?")
            args.append(key)
        sql.append(f"AND status IN ({', '.join('?' * len(statuses))})")
        args.extend(s.value for s in statuses)
        if not include_expired:
            sql.append("AND (expires_at IS NULL OR expires_at > ?)")
            args.append(_ts(now))
        sql.append("ORDER BY created_at, id")
        return [self._record(r) for r in self._conn.execute(" ".join(sql), args).fetchall()]

    def delete(self, owner: Owner, record_id: str, *, reason: str, now: datetime | None = None) -> list[Tombstone]:
        now = now or utcnow()
        with self._lock:
            root = self.get(owner, record_id)
            if root is None:
                return []
            doomed = self._collect_derived(root)
            stones: list[Tombstone] = []
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                for r in doomed:
                    stone = Tombstone(
                        record_id=r.id, owner=r.owner, kind=r.kind, key=r.key,
                        fingerprint=fingerprint(r, self._secret), deleted_at=now,
                        reason=reason if r.id == record_id else f"derived from {record_id}: {reason}",
                    )
                    self._conn.execute("DELETE FROM memories WHERE id = ? AND tenant = ?", (r.id, r.owner.tenant))
                    self._conn.execute(
                        "INSERT INTO tombstones VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (r.id, r.owner.tenant, r.owner.user or "", r.kind.value, r.key, stone.fingerprint, _ts(now), stone.reason),
                    )
                    stones.append(stone)
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            return stones

    def _collect_derived(self, root: MemoryRecord) -> list[MemoryRecord]:
        found = {root.id: root}
        frontier = [root.id]
        while frontier:
            parent = frontier.pop()
            rows = self._conn.execute(
                f"SELECT {self._COLUMNS} FROM memories WHERE tenant = ? AND provenance_json LIKE ?",
                (root.owner.tenant, f'%"mem:{parent}"%'),
            ).fetchall()
            for row in rows:
                r = self._record(row)
                if r.id not in found:
                    found[r.id] = r
                    frontier.append(r.id)
        return list(found.values())

    def delete_owner(self, owner: Owner) -> int:
        args = (owner.tenant, owner.user or "")
        ids = [row["id"] for row in self._conn.execute("SELECT id FROM memories WHERE tenant = ? AND user_id = ?", args)]
        stones = [t for i in ids for t in self.delete(owner, i, reason="account erased")]
        with self._lock:
            self._conn.execute("DELETE FROM tombstones WHERE tenant = ? AND user_id = ?", args)
            self._conn.executemany("DELETE FROM tombstones WHERE tenant = ? AND record_id = ?",
                                   [(owner.tenant, t.record_id) for t in stones])
        return sum(1 for t in stones if t.owner == owner)

    def is_suppressed(self, record: MemoryRecord) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM tombstones WHERE tenant = ? AND user_id = ? AND fingerprint = ? LIMIT 1",
            (record.owner.tenant, record.owner.user or "", fingerprint(record, self._secret)),
        ).fetchone()
        return row is not None

    def tombstones(self, owner: Owner) -> list[Tombstone]:
        rows = self._conn.execute(
            "SELECT * FROM tombstones WHERE tenant = ? AND user_id = ? ORDER BY deleted_at",
            (owner.tenant, owner.user or ""),
        ).fetchall()
        return [
            Tombstone(
                record_id=r["record_id"], owner=owner, kind=MemoryKind(r["kind"]), key=r["key"],
                fingerprint=r["fingerprint"], deleted_at=_dt(r["deleted_at"]), reason=r["reason"],
            )
            for r in rows
        ]

    def purge_expired(self, now: datetime | None = None) -> int:
        with self._lock:
            return self._conn.execute(
                "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?", (_ts(now or utcnow()),)
            ).rowcount

    def export(self, owner: Owner) -> dict[str, Any]:
        records = self.query(owner, statuses=tuple(MemoryStatus), include_expired=True)
        return _export_payload(owner, records, self.tombstones(owner))


__all__ = ["MemoryStore", "InMemoryStore", "SQLiteStore", "VersionConflict", "fingerprint"]
