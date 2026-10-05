-- path: book/projects/p2-semantic-search/sql/schema.sql
-- Rendered by semsearch.adapters.pg_store.render_schema(256). The application renders its own
-- copy for EMBEDDING_DIMENSIONS at startup; a test keeps this file in sync.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id              TEXT PRIMARY KEY,                -- doc_id@version#ordinal
    namespace       TEXT NOT NULL,                   -- index name : embedding model : index version
    tenant          TEXT NOT NULL,                   -- 'retail' | 'logistics' | 'shared'
    doc_id          TEXT NOT NULL,
    doc_version     TEXT NOT NULL,
    ordinal         INT  NOT NULL,
    text            TEXT NOT NULL,
    acl_groups      TEXT[] NOT NULL CHECK (cardinality(acl_groups) > 0),
    tags            TEXT[] NOT NULL DEFAULT '{}',
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding_model TEXT NOT NULL,
    embedding       vector(256) NOT NULL,
    text_tsv        tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ANN index. Cosine operator class because vectors are compared by cosine distance (<=>).
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Filter and maintenance indexes.
CREATE INDEX IF NOT EXISTS chunks_ns_tenant ON chunks (namespace, tenant);
CREATE INDEX IF NOT EXISTS chunks_ns_doc    ON chunks (namespace, doc_id);
CREATE INDEX IF NOT EXISTS chunks_acl_gin   ON chunks USING gin (acl_groups);
CREATE INDEX IF NOT EXISTS chunks_tags_gin  ON chunks USING gin (tags);
CREATE INDEX IF NOT EXISTS chunks_meta_gin  ON chunks USING gin (metadata jsonb_path_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin   ON chunks USING gin (text_tsv);
