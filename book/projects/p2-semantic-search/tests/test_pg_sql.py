# path: book/projects/p2-semantic-search/tests/test_pg_sql.py
"""Offline checks of the SQL the pgvector adapter generates (no database needed)."""
from __future__ import annotations

from pathlib import Path

import pytest

from semsearch.adapters.pg_store import PgVectorStore, build_where, render_schema
from semsearch.domain.models import SearchFilter

SCHEMA_FILE = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"


def test_where_is_parameterized():
    where, params = build_where(
        "knowledge:m:v1",
        SearchFilter(tenants=("retail", "shared"), acl_groups=("all",), tags_any=("api",), doc_ids=("d'; DROP TABLE x;--",)),
    )
    assert where == (
        "namespace = %(namespace)s AND tenant = ANY(%(tenants)s) AND acl_groups && %(acl_groups)s::text[] "
        "AND tags && %(tags_any)s::text[] AND doc_id = ANY(%(doc_ids)s)"
    )
    assert "DROP" not in where
    assert params["tenants"] == ["retail", "shared"]


def test_where_without_filter_still_scopes_namespace():
    where, params = build_where("ns:a:v1", None)
    assert where == "namespace = %(namespace)s" and params == {"namespace": "ns:a:v1"}


def test_schema_has_hnsw_and_filter_indexes():
    sql = render_schema(768, "chunks", m=24, ef_construction=128)
    assert "vector(768)" in sql
    assert "USING hnsw (embedding vector_cosine_ops)" in sql
    assert "m = 24, ef_construction = 128" in sql
    assert "USING gin (acl_groups)" in sql


def test_schema_rejects_unsafe_identifiers():
    with pytest.raises(ValueError):
        render_schema(64, "chunks; DROP TABLE users")


def test_checked_in_schema_matches_renderer():
    body = SCHEMA_FILE.read_text(encoding="utf-8").split("\n\n", 1)[1]
    assert body == render_schema(256)


class _RecordingConn:
    """Just enough of a psycopg connection to see which statements a search sends."""

    def __init__(self) -> None:
        self.sql: list[tuple[str, tuple]] = []

    def transaction(self):
        import contextlib

        return contextlib.nullcontext()

    def cursor(self):
        conn = self

        class Cur:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def execute(self, sql, params=()):
                conn.sql.append((" ".join(str(sql).split()), params))

            def fetchall(self):
                return []

        return Cur()


def _offline_store(**attrs):
    store = PgVectorStore.__new__(PgVectorStore)  # skip __init__: no database in offline tests
    store.dimensions, store.table, store.ef_search, store.iterative_scan = 4, "chunks", 100, "off"
    store._has_iterative, store.statement_timeout_ms = False, None
    store.__dict__.update(attrs)
    store._conn = _RecordingConn()
    return store


def test_statement_timeout_is_transaction_local_and_optional():
    store = _offline_store(statement_timeout_ms=250)
    store.search("ns", [0.1, 0.2, 0.3, 0.4], k=3)
    store.hybrid_search("ns", [0.1, 0.2, 0.3, 0.4], "vpn error", k=3)
    timeouts = [p for sql, p in store._conn.sql if "statement_timeout" in sql]
    assert timeouts == [("250",), ("250",)]  # set_config(..., true) inside each read transaction
    assert all("set_config('statement_timeout', %s, true)" in sql for sql, _ in store._conn.sql if "statement_timeout" in sql)
    quiet = _offline_store()
    quiet.search("ns", [0.1, 0.2, 0.3, 0.4], k=3)
    assert not any("statement_timeout" in sql for sql, _ in quiet._conn.sql)
