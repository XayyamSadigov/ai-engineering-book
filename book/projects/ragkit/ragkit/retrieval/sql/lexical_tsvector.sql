-- path: book/projects/ragkit/ragkit/retrieval/sql/lexical_tsvector.sql
-- Lexical retrieval inside PostgreSQL, on the chunk table from Chapter 9 (semsearch PgVectorStore,
-- default table name "chunks"). Not executed by the offline tests: pgvector/PostgreSQL were not
-- available in the book's environment. Project 3 wires it into its Postgres adapter.
-- The metadata keys used below (title, breadcrumb, context_prefix, identifiers) are written at the
-- top level of each record's metadata by ragkit.retrieval.dense.DenseRetriever.index.

-- 1. A weighted search vector: title (A) > section breadcrumb (B) > body (C).
--    The breadcrumb and title live in metadata; the contextual prefix (if any) joins the body.
ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS search_tsv tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(metadata->>'title', '')), 'A') ||
        setweight(to_tsvector('english', coalesce(metadata->>'breadcrumb', '')), 'B') ||
        setweight(to_tsvector('english', coalesce(metadata->>'context_prefix', '') || ' ' || text), 'C')
    ) STORED;

CREATE INDEX IF NOT EXISTS chunks_search_tsv_gin ON chunks USING gin (search_tsv);

-- 2. Identifiers. The 'english' parser splits "INC-2025-1142" into pieces and may stem them.
--    Index exact identifiers separately so an exact-ID query matches as one token.
ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS ident_tsv tsvector GENERATED ALWAYS AS (
        to_tsvector('simple', coalesce(metadata->>'identifiers', ''))
    ) STORED;

CREATE INDEX IF NOT EXISTS chunks_ident_tsv_gin ON chunks USING gin (ident_tsv);

-- 3. The query. Authorization predicates come first and are not optional; the planner can use
--    the tenant/ACL indexes and the GIN index together. ts_rank_cd is a cover-density rank, not
--    BM25: it has no IDF term and normalizes length differently (flag 32 maps rank to rank/(rank+1)).
--    Parameters: %(namespace)s, %(tenants)s, %(groups)s, %(q)s, %(ident)s, %(n)s
WITH q AS (
    SELECT websearch_to_tsquery('english', %(q)s) AS words,
           plainto_tsquery('simple', %(ident)s)   AS ident
)
SELECT c.id, c.doc_id, c.text,
       ts_rank_cd(c.search_tsv, q.words, 32)
         + CASE WHEN c.ident_tsv @@ q.ident THEN 1.0 ELSE 0.0 END AS lexical_score
FROM chunks c, q
WHERE c.namespace = %(namespace)s
  AND c.tenant = ANY(%(tenants)s)
  AND c.acl_groups && %(groups)s::text[]
  AND (c.search_tsv @@ q.words OR c.ident_tsv @@ q.ident)
ORDER BY lexical_score DESC, c.id
LIMIT %(n)s;
