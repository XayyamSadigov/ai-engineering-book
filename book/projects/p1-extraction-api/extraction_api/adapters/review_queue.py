# path: book/projects/p1-extraction-api/extraction_api/adapters/review_queue.py
"""Review queue adapters: in-memory for tests and demos, SQLite for a single-node service.

Resolution is a compare-and-set on status, so two reviewers clicking at the same moment
cannot both resolve one item. A Postgres adapter would use the same UPDATE ... WHERE
status = 'pending' and add SELECT ... FOR UPDATE SKIP LOCKED for claiming work.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from ..application.ports import AlreadyResolved, ReviewItem, ReviewNotFound, ReviewResolution, ReviewStatus


class InMemoryReviewQueue:
    def __init__(self) -> None:
        self._items: dict[str, ReviewItem] = {}
        self._lock = threading.Lock()

    def enqueue(self, item: ReviewItem) -> ReviewItem:
        with self._lock:
            self._items[item.review_id] = item
        return item

    def get(self, review_id: str) -> ReviewItem:
        try:
            return self._items[review_id]
        except KeyError:
            raise ReviewNotFound(review_id) from None

    def list(
        self, status: ReviewStatus | None = "pending", limit: int = 50, *, tenant: str | None = None
    ) -> list[ReviewItem]:
        items = sorted(self._items.values(), key=lambda i: i.created_at)
        return [i for i in items
                if (status is None or i.status == status) and (tenant is None or i.tenant == tenant)][:limit]

    def resolve(self, review_id: str, resolution: ReviewResolution) -> ReviewItem:
        with self._lock:
            item = self.get(review_id)
            if item.status != "pending":
                raise AlreadyResolved(review_id)
            updated = item.model_copy(update={"status": resolution.status, "resolution": resolution})
            self._items[review_id] = updated
            return updated

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for i in self._items.values():
            out[i.status] = out.get(i.status, 0) + 1
        return out


class SQLiteReviewQueue:
    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS review_items (
        review_id  TEXT PRIMARY KEY,
        status     TEXT NOT NULL,
        doc_type   TEXT NOT NULL,
        created_at TEXT NOT NULL,
        payload    TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS ix_review_status_created ON review_items(status, created_at);
    """

    def __init__(self, path: str | Path = "review.db") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # one connection guarded by a lock: FastAPI runs sync endpoints on a thread pool
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(self._SCHEMA)
        self._lock = threading.Lock()

    def enqueue(self, item: ReviewItem) -> ReviewItem:
        with self._lock:
            self._conn.execute(
                "INSERT INTO review_items (review_id, status, doc_type, created_at, payload) VALUES (?, ?, ?, ?, ?)",
                (item.review_id, item.status, item.doc_type.value, item.created_at.isoformat(), item.model_dump_json()),
            )
        return item

    def get(self, review_id: str) -> ReviewItem:
        with self._lock:
            row = self._conn.execute("SELECT payload FROM review_items WHERE review_id = ?", (review_id,)).fetchone()
        if row is None:
            raise ReviewNotFound(review_id)
        return ReviewItem.model_validate_json(row[0])

    def list(
        self, status: ReviewStatus | None = "pending", limit: int = 50, *, tenant: str | None = None
    ) -> list[ReviewItem]:
        # filters are bound parameters; "? IS NULL OR ..." keeps one statement for every combination
        # (tenant lives in the JSON payload; a Postgres adapter would give it an indexed column)
        with self._lock:
            rows = self._conn.execute(
                "SELECT payload FROM review_items"
                " WHERE (? IS NULL OR status = ?) AND (? IS NULL OR json_extract(payload, '$.tenant') = ?)"
                " ORDER BY created_at LIMIT ?",
                (status, status, tenant, tenant, limit)).fetchall()
        return [ReviewItem.model_validate_json(r[0]) for r in rows]

    def resolve(self, review_id: str, resolution: ReviewResolution) -> ReviewItem:
        item = self.get(review_id)
        updated = item.model_copy(update={"status": resolution.status, "resolution": resolution})
        with self._lock:
            cur = self._conn.execute(
                "UPDATE review_items SET status = ?, payload = ? WHERE review_id = ? AND status = 'pending'",
                (updated.status, updated.model_dump_json(), review_id),
            )
        if cur.rowcount == 0:
            raise AlreadyResolved(review_id)
        return updated

    def counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute("SELECT status, COUNT(*) FROM review_items GROUP BY status").fetchall()
        return {status: n for status, n in rows}

    def close(self) -> None:
        self._conn.close()


__all__ = ["InMemoryReviewQueue", "SQLiteReviewQueue"]
