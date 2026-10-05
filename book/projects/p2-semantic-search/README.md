# Project 2: Semantic search system

Filtered semantic search over the Northwind document set, built in Chapter 9 (Vector Search and
Vector Databases). One `VectorStore` protocol, two adapters:

- `NumpyVectorStore`: exact cosine search, metadata pre-filtering, namespaces, persistence to
  `.npz` + `.json`. No infrastructure; the default and the test backend.
- `PgVectorStore`: PostgreSQL + pgvector, one table, HNSW index, tenant/ACL/tag columns with GIN
  indexes, JSONB metadata, transactional replace-by-document-version, exact-scan mode for
  measuring ANN recall, and a dense + full-text hybrid query fused with RRF.

Around them: an ingestion CLI (front matter, paragraph chunking, incremental by content hash,
pruning), a FastAPI `/search` endpoint whose tenant and ACL filters come from identity headers,
an evaluator (recall@k, MRR, ACL leakage, ingestion gaps), and two offline experiments (IVF
recall vs nprobe, post-filter vs pre-filter under selective filters).

## Layout

```
p2-semantic-search/
  pyproject.toml  .env.example  Dockerfile  docker-compose.yml  README.md
  sql/schema.sql                 rendered pgvector schema (kept in sync by a test)
  semsearch/
    config.py                    SearchSettings, make_embedder, make_store, namespace_for
    service.py                   Principal, SearchService (authorization filter, tracing)
    cli.py                       ingest | search | eval | ann-check | bench-ann | bench-filter | serve
    bench.py                     ANN sweep and filter experiment
    domain/
      models.py                  VectorRecord, SearchFilter, SearchHit, DocVersion, make_namespace
      filters.py                 filter semantics shared by all adapters
      metrics.py                 recall@k, precision@k, reciprocal rank, overlap@k
      ivf.py                     teaching-size IVF index
    adapters/
      base.py                    VectorStore protocol
      numpy_store.py             NumpyVectorStore
      pg_store.py                PgVectorStore, render_schema, build_where
    ingest/
      loader.py                  front matter, paragraph chunking, embedding text
      pipeline.py                incremental ingest with replace_document and prune
    api/app.py                   FastAPI app
    eval/run_eval.py             evaluator; eval/fixtures/retrieval_fixture.jsonl
  tests/                         offline tests; pgvector contract tests marked integration
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `EMBEDDING_PROVIDER` | `fake` | `fake` (hashing, offline) or `openai` (any OpenAI-compatible endpoint) |
| `EMBEDDING_MODEL` | `fake-embedding` | part of the namespace; changing it means a new index |
| `EMBEDDING_DIMENSIONS` | `256` | width of the fake embedder |
| `OPENAI_API_KEY`, `LLM_BASE_URL` | unset | credentials and base URL for real embeddings |
| `VECTOR_BACKEND` | `numpy` | `numpy` or `pgvector` |
| `INDEX_DIR` | `.index` | NumPy store directory |
| `DATABASE_URL` | unset | required for `pgvector` |
| `PG_TABLE` | `chunks` | table name |
| `HNSW_M`, `HNSW_EF_CONSTRUCTION` | `16`, `64` | index build parameters |
| `HNSW_EF_SEARCH` | `100` | query-time candidate list size |
| `HNSW_ITERATIVE_SCAN` | `relaxed_order` | `off`, `strict_order`, `relaxed_order` (pgvector 0.8+) |
| `INDEX_NAME`, `INDEX_VERSION` | `knowledge`, `v1` | namespace = name : model : version |
| `DOCS_DIR` | `../shared-data/docs` | corpus |
| `CHUNK_MAX_CHARS` | `1200` | paragraph-group budget |
| `SEARCH_DEFAULT_K`, `SEARCH_MAX_K` | `8`, `50` | result counts |
| `TRACE_SINK`, `TRACE_PATH` | `none` | `jsonl` writes one span per search/ingest |

## Run

```bash
# from the book root, into the shared virtualenv
uv pip install --python .venv/bin/python -e book/projects/aie_core -e "book/projects/p2-semantic-search[pg,dev]"
# or: pip install -e ../aie_core && pip install -e ".[pg,dev]"

cd book/projects/p2-semantic-search
semsearch ingest                                    # NumPy backend, writes .index/
semsearch search "NorthGate VPN error 412" --tenant shared --groups all
semsearch search "RTO for the database cluster" --tenant logistics --groups all,logistics
semsearch eval                                      # exits 1 on any ACL leak
semsearch bench-ann && semsearch bench-filter
semsearch serve --port 8000
curl -s localhost:8000/search -H 'X-User: u1' -H 'X-Tenant: retail' -H 'X-Groups: all,retail' \
     -H 'Content-Type: application/json' -d '{"query": "return window for rewards members", "k": 5}'
```

With PostgreSQL + pgvector:

```bash
docker compose up -d db
export VECTOR_BACKEND=pgvector DATABASE_URL=postgresql://northwind:northwind@localhost:5432/northwind
semsearch ingest --prune && semsearch eval && semsearch ann-check
docker compose run --rm ingest && docker compose up api     # or fully containerized
```

## Tests

```bash
python -m pytest -q                                  # offline: NumPy store, FakeEmbeddings
DATABASE_URL=postgresql://northwind:northwind@localhost:5432/northwind python -m pytest -q -m integration
```

The store contract (`tests/test_store_contract.py`) runs the same assertions against both
adapters, so a new adapter is done when it passes that file.
