# path: book/projects/p2-semantic-search/semsearch/adapters/pg_store.py
"""PostgreSQL + pgvector adapter.

One table holds every namespace. The vector column has fixed dimensions (an HNSW index
requires it), authorization fields are real columns with indexes, and free-form metadata
lives in JSONB. Writes for one document happen in one transaction. Searches run in a
transaction too, so that per-query index settings (`SET LOCAL`) never leak to the next
request that reuses the pooled connection.

Requires: `pip install "psycopg[binary]" pgvector` and the `vector` extension (0.5+ for
HNSW; 0.8+ for iterative index scans, which the adapter enables when available).
"""
from __future__ import annotations

import json
import re
from typing import Any, Literal, Sequence

from ..domain.models import DocVersion, SearchFilter, SearchHit, Vector, VectorRecord
from .base import check_dimensions, check_document

_IDENT_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")

SCHEMA_TEMPLATE = """\
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS {table} (
    id              TEXT PRIMARY KEY,                -- doc_id@version#ordinal
    namespace       TEXT NOT NULL,                   -- index name : embedding model : index version
    tenant          TEXT NOT NULL,                   -- 'retail' | 'logistics' | 'shared'
    doc_id          TEXT NOT NULL,
    doc_version     TEXT NOT NULL,
    ordinal         INT  NOT NULL,
    text            TEXT NOT NULL,
    acl_groups      TEXT[] NOT NULL CHECK (cardinality(acl_groups) > 0),
    tags            TEXT[] NOT NULL DEFAULT '{{}}',
    metadata        JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    embedding_model TEXT NOT NULL,
    embedding       vector({dimensions}) NOT NULL,
    text_tsv        tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ANN index. Cosine operator class because vectors are compared by cosine distance (<=>).
CREATE INDEX IF NOT EXISTS {table}_embedding_hnsw
    ON {table} USING hnsw (embedding vector_cosine_ops)
    WITH (m = {m}, ef_construction = {ef_construction});

-- Filter and maintenance indexes.
CREATE INDEX IF NOT EXISTS {table}_ns_tenant ON {table} (namespace, tenant);
CREATE INDEX IF NOT EXISTS {table}_ns_doc    ON {table} (namespace, doc_id);
CREATE INDEX IF NOT EXISTS {table}_acl_gin   ON {table} USING gin (acl_groups);
CREATE INDEX IF NOT EXISTS {table}_tags_gin  ON {table} USING gin (tags);
CREATE INDEX IF NOT EXISTS {table}_meta_gin  ON {table} USING gin (metadata jsonb_path_ops);
CREATE INDEX IF NOT EXISTS {table}_tsv_gin   ON {table} USING gin (text_tsv);
"""


def render_schema(dimensions: int, table: str = "chunks", m: int = 16, ef_construction: int = 64) -> str:
    if not _IDENT_RE.match(table):
        raise ValueError(f"invalid table name {table!r}")
    if dimensions <= 0 or dimensions > 16000:
        raise ValueError("dimensions out of range")
    return SCHEMA_TEMPLATE.format(table=table, dimensions=int(dimensions), m=int(m), ef_construction=int(ef_construction))


_COLUMNS = "id, doc_id, doc_version, ordinal, tenant, text, tags, metadata"


def build_where(namespace: str, flt: SearchFilter | None) -> tuple[str, dict[str, Any]]:
    """Translate a SearchFilter into a parameterized WHERE clause. Values never enter the SQL text."""
    clauses = ["namespace = %(namespace)s"]
    params: dict[str, Any] = {"namespace": namespace}
    if flt is not None:
        if flt.tenants is not None:
            clauses.append("tenant = ANY(%(tenants)s)")
            params["tenants"] = list(flt.tenants)
        if flt.acl_groups is not None:
            clauses.append("acl_groups && %(acl_groups)s::text[]")
            params["acl_groups"] = list(flt.acl_groups)
        if flt.tags_any is not None:
            clauses.append("tags && %(tags_any)s::text[]")
            params["tags_any"] = list(flt.tags_any)
        if flt.doc_ids is not None:
            clauses.append("doc_id = ANY(%(doc_ids)s)")
            params["doc_ids"] = list(flt.doc_ids)
    return " AND ".join(clauses), params


class PgVectorStore:
    def __init__(
        self,
        dsn: str,
        dimensions: int,
        *,
        table: str = "chunks",
        m: int = 16,
        ef_construction: int = 64,
        ef_search: int = 100,
        iterative_scan: Literal["off", "strict_order", "relaxed_order"] = "relaxed_order",
        create_schema: bool = True,
        statement_timeout_ms: int | None = None,
    ) -> None:
        import psycopg
        from pgvector.psycopg import register_vector

        if not _IDENT_RE.match(table):
            raise ValueError(f"invalid table name {table!r}")
        self.dimensions = dimensions
        self.table = table
        self.ef_search = ef_search
        self.iterative_scan = iterative_scan
        # Per-transaction cap on read queries, so a degraded index fails fast instead of holding the
        # caller's thread; the caller degrades (Chapter 12). None leaves the server default.
        self.statement_timeout_ms = statement_timeout_ms
        self._m, self._efc = m, ef_construction
        self._conn = psycopg.connect(dsn, autocommit=True)
        if create_schema:
            self._conn.execute(render_schema(dimensions, table, m, ef_construction))
        register_vector(self._conn)
        self._has_iterative = self._supports_iterative_scan()

    def close(self) -> None:
        self._conn.close()

    def _supports_iterative_scan(self) -> bool:
        row = self._conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
        if not row:
            return False
        major, minor = (int(x) for x in str(row[0]).split(".")[:2])
        return (major, minor) >= (0, 8)

    # ------------------------------------------------------------------ writes
    def _upsert_sql(self) -> str:
        return f"""
            INSERT INTO {self.table}
                (id, namespace, tenant, doc_id, doc_version, ordinal, text, acl_groups, tags,
                 metadata, embedding_model, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                namespace = EXCLUDED.namespace, tenant = EXCLUDED.tenant, text = EXCLUDED.text,
                acl_groups = EXCLUDED.acl_groups, tags = EXCLUDED.tags, metadata = EXCLUDED.metadata,
                embedding_model = EXCLUDED.embedding_model, embedding = EXCLUDED.embedding,
                updated_at = now()
        """

    @staticmethod
    def _row(r: VectorRecord) -> tuple[Any, ...]:
        import numpy as np

        return (
            r.id, r.namespace, r.tenant, r.doc_id, r.doc_version, r.ordinal, r.text,
            list(r.acl_groups), list(r.tags), json.dumps(r.metadata, default=str), r.embedding_model,
            np.asarray(r.vector, dtype=np.float32),
        )

    def upsert(self, records: Sequence[VectorRecord]) -> int:
        check_dimensions(records, self.dimensions)
        if not records:
            return 0
        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.executemany(self._upsert_sql(), [self._row(r) for r in records])
        return len(records)

    def replace_document(self, namespace: str, doc_id: str, doc_version: str, records: Sequence[VectorRecord]) -> int:
        check_dimensions(records, self.dimensions)
        check_document(namespace, doc_id, doc_version, records)
        with self._conn.transaction(), self._conn.cursor() as cur:
            # Serialize concurrent writers of the same document (two ingest workers, one doc).
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"{namespace}/{doc_id}",))
            if records:
                cur.executemany(self._upsert_sql(), [self._row(r) for r in records])
            keep = [r.id for r in records]
            cur.execute(
                f"DELETE FROM {self.table} WHERE namespace = %s AND doc_id = %s AND NOT (id = ANY(%s))",
                (namespace, doc_id, keep),
            )
        return len(records)

    def delete_document(self, namespace: str, doc_id: str) -> int:
        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self.table} WHERE namespace = %s AND doc_id = %s", (namespace, doc_id))
            return cur.rowcount

    # ------------------------------------------------------------------ reads
    def search(
        self,
        namespace: str,
        query: Vector,
        k: int,
        flt: SearchFilter | None = None,
        *,
        exact: bool = False,
    ) -> list[SearchHit]:
        import numpy as np

        if len(query) != self.dimensions:
            raise ValueError(f"query has {len(query)} dimensions, store expects {self.dimensions}")
        if k <= 0:
            return []
        where, params = build_where(namespace, flt)
        params |= {"q": np.asarray(query, dtype=np.float32), "k": int(k)}
        sql = f"""
            SELECT {_COLUMNS}, 1 - (embedding <=> %(q)s) AS score
            FROM {self.table}
            WHERE {where}
            ORDER BY embedding <=> %(q)s
            LIMIT %(k)s
        """
        with self._conn.transaction(), self._conn.cursor() as cur:
            self._apply_timeout(cur)
            if exact:
                # Ground truth: forbid the ANN index so the planner scans and sorts every row.
                cur.execute("SET LOCAL enable_indexscan = off")
                cur.execute("SET LOCAL enable_bitmapscan = off")
            else:
                cur.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(self.ef_search),))
                if self._has_iterative and self.iterative_scan != "off":
                    # Keep scanning the graph until LIMIT rows pass the filter (pgvector 0.8+).
                    cur.execute("SELECT set_config('hnsw.iterative_scan', %s, true)", (self.iterative_scan,))
            cur.execute(sql, params)
            rows = cur.fetchall()
        hits = [self._hit(row) for row in rows]
        # relaxed_order may return rows slightly out of order; restore the contract.
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits

    def hybrid_search(
        self,
        namespace: str,
        query: Vector,
        query_text: str,
        k: int,
        flt: SearchFilter | None = None,
        *,
        candidates: int = 50,
        rrf_k: int = 60,
    ) -> list[SearchHit]:
        """Dense + full-text candidates fused with Reciprocal Rank Fusion, all in one SQL
        statement. Chapter 12 explains RRF and when weighted fusion beats it."""
        import numpy as np

        where, params = build_where(namespace, flt)
        params |= {
            "q": np.asarray(query, dtype=np.float32),
            "qt": query_text,
            "n": int(candidates),
            "k": int(k),
            "rrf": int(rrf_k),
        }
        sql = f"""
            WITH dense AS (
                SELECT id, row_number() OVER (ORDER BY embedding <=> %(q)s) AS rnk
                FROM (SELECT id, embedding FROM {self.table} WHERE {where}
                      ORDER BY embedding <=> %(q)s LIMIT %(n)s) d
            ),
            lexical AS (
                SELECT id, row_number() OVER (ORDER BY ts_rank_cd(text_tsv, query) DESC) AS rnk
                FROM {self.table}, websearch_to_tsquery('english', %(qt)s) AS query
                WHERE {where} AND text_tsv @@ query
                ORDER BY ts_rank_cd(text_tsv, query) DESC
                LIMIT %(n)s
            ),
            fused AS (
                SELECT id, sum(1.0 / (%(rrf)s + rnk)) AS rrf
                FROM (SELECT * FROM dense UNION ALL SELECT * FROM lexical) u
                GROUP BY id
            )
            SELECT {", ".join("c." + c.strip() for c in _COLUMNS.split(","))}, f.rrf AS score
            FROM fused f JOIN {self.table} c ON c.id = f.id
            ORDER BY f.rrf DESC
            LIMIT %(k)s
        """
        with self._conn.transaction(), self._conn.cursor() as cur:
            self._apply_timeout(cur)
            cur.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(max(self.ef_search, candidates)),))
            cur.execute(sql, params)
            return [self._hit(row) for row in cur.fetchall()]

    def _apply_timeout(self, cur: Any) -> None:
        if self.statement_timeout_ms is not None:
            # set_config(..., true) is SET LOCAL: it ends with the transaction, so pooled
            # connections never inherit it (the same reason ef_search is set this way).
            cur.execute("SELECT set_config('statement_timeout', %s, true)", (str(int(self.statement_timeout_ms)),))

    @staticmethod
    def _hit(row: tuple[Any, ...]) -> SearchHit:
        id_, doc_id, doc_version, ordinal, tenant, text, tags, metadata, score = row
        return SearchHit(
            id=id_, doc_id=doc_id, doc_version=doc_version, ordinal=ordinal, tenant=tenant,
            score=float(score), text=text, tags=tuple(tags or ()), metadata=metadata or {},
        )

    def doc_versions(self, namespace: str) -> dict[str, DocVersion]:
        rows = self._conn.execute(
            f"SELECT doc_id, min(doc_version), count(*) FROM {self.table} WHERE namespace = %s GROUP BY doc_id",
            (namespace,),
        ).fetchall()
        return {d: DocVersion(doc_id=d, doc_version=v, chunks=n) for d, v, n in rows}

    def count(self, namespace: str | None = None) -> int:
        if namespace is None:
            row = self._conn.execute(f"SELECT count(*) FROM {self.table}").fetchone()
        else:
            row = self._conn.execute(f"SELECT count(*) FROM {self.table} WHERE namespace = %s", (namespace,)).fetchone()
        return int(row[0]) if row else 0

    def namespaces(self) -> list[str]:
        rows = self._conn.execute(f"SELECT DISTINCT namespace FROM {self.table} ORDER BY 1").fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------ operations
    def reindex(self, concurrently: bool = True) -> None:
        """Rebuild the HNSW index (after bulk loads or heavy deletes). CONCURRENTLY keeps
        reads and writes flowing at the cost of a slower build."""
        kw = "CONCURRENTLY " if concurrently else ""
        self._conn.execute(f"REINDEX INDEX {kw}{self.table}_embedding_hnsw")

    def drop_namespace(self, namespace: str) -> int:
        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self.table} WHERE namespace = %s", (namespace,))
            return cur.rowcount


__all__ = ["PgVectorStore", "render_schema", "build_where", "SCHEMA_TEMPLATE"]
