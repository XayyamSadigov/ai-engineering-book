# Chapter 9 — Vector Search and Vector Databases

Chapter 8 covered what embeddings are; this chapter is about storing and searching them: fast enough at your scale, filtered by who is asking, and safe to rebuild when the model or the documents change. You build Project 2, a semantic search system with a `VectorStore` protocol, an exact NumPy adapter, a PostgreSQL + pgvector adapter, an incremental ingestion CLI, a FastAPI `/search` endpoint whose authorization filters come from identity, and an evaluator.

**You will be able to:**
- Decide whether a workload needs a vector database at all, using corpus size, query rate, and latency budget.
- Choose between exact search, HNSW, IVF, and quantized indexes from measured recall, latency, and memory, and tune them against overlap with exact search.
- Design metadata filtering that stays correct under selective filters, and detect filtered-ANN starvation from telemetry.
- Lay out namespaces and tenants so that one customer's vectors can never answer another customer's query.
- Operate an index through re-embedding, blue-green rebuilds, deletions, and replication.

**Prerequisites:** Chapters 3 (the `aie_core` embedding clients, `CachedEmbeddings`, settings, and tracing) and 8 (similarity metrics, normalization, and the embedding space fingerprint). | **Code:** `book/projects/p2-semantic-search/` (run: `cd book/projects/p2-semantic-search && pytest -q`) | **Builds:** Project 2, the `semsearch` package.

**First reading:** Why this matters, Mental model, Core concepts (except the deep dives below), How it works, Architecture, Indexing strategies and rebuilds, Implementation through The pgvector adapter, Failure modes, Tradeoffs, Before you ship. **Deep dives** (skip on a first pass): Distance, normalization, and what a score means; Product quantization; Consistency; Ingestion; The search service and API; Evaluation; The ANN and filter experiments; The store contract test; Code walkthrough; Production considerations.

## Why this matters

Once text is embedded, retrieval is a geometry problem: find the stored vectors closest to the query vector. The dot product takes nanoseconds. The engineering is everything around it.

The first problem is scale. A million 768-dimensional float32 vectors occupy about 3 GB, and one brute-force query over them took about 60 ms on a laptop (illustrative): fine for an internal tool, too slow for hundreds of queries per second. Approximate nearest neighbor (ANN) indexes buy speed by giving up exactness, at the price of a recall loss nobody sees unless they measure it.

The second problem is that a vector is never the whole query. Northwind Assist must answer a logistics employee from logistics documents and shared policies, never retail incident reports, and never the on-call runbooks unless the user is in `it-oncall`. Those metadata filters interact badly with ANN indexes: a filter matching 1% of the corpus can leave an approximate index returning one result or none instead of ten. The bug looks like "the assistant does not know about our product", not like an error.

The third problem is lifecycle. Documents change and get deleted. Vectors from two embedding models are not comparable, so a model upgrade means re-embedding everything and swapping indexes without downtime. A deleted policy that stays retrievable is both a correctness bug and a compliance incident.

Many teams that adopted dedicated vector databases for these problems found that PostgreSQL with pgvector, or a NumPy matrix, would have done with less to operate; others outgrew both. This chapter gives you the criteria to tell which case you are in, and code that starts simple without painting you into a corner.

## Mental model

> **Mental model:** A vector index is a derived, versioned cache of a computation over your corpus. The documents are the source of truth; the index is rebuildable, and every record in it must carry enough identity (document, version, tenant, ACL, embedding model) to be filtered, cited, invalidated, and rebuilt.

**Recall is a property of the whole stack.** The embedding model, the chunker, a filter, or the ANN index can each lose the answer, and each needs its own measurement.

**Filters are part of the query, not decoration on top of it.** "Top 10 neighbors this user may see" is correct by construction. "Top 10 neighbors, then remove the ones this user may not see" silently returns fewer results as filters get more selective, which is exactly the case for small tenants and restricted groups.

**Start exact, measure, and only then approximate.** Exact search is the ground truth for every ANN index; keep an exact path, even offline only, to know what the index costs you.

## Core concepts

### What a vector store actually stores

The unit of storage is a record with a vector in it. Each field of Project 2's `VectorRecord` exists because a later stage fails without it:

| Field | Why it exists | What breaks without it |
|---|---|---|
| `id` | Deterministic `doc_id@version#ordinal` | Re-ingestion duplicates chunks instead of replacing them |
| `namespace` | Index name, embedding model, index version | Vectors from two models get compared; scores become noise |
| `tenant` | Isolation boundary (`retail`, `logistics`, `shared`) | Cross-tenant leakage |
| `acl_groups` | Who may read the chunk | The model quotes a restricted runbook |
| `doc_id`, `doc_version`, `ordinal` | Citation and invalidation | Citations unresolvable; stale chunks unfindable |
| `text` | What the generator and reranker read | A second lookup per hit |
| `tags`, `metadata` | Narrowing filters, titles, sections, dates | No scoped searches; no sources shown |
| `embedding_model` | Audit and migration | Nobody knows which rows need re-embedding |

Two rules follow. First, **the embedding model version is part of the index schema.** Two models produce vectors in unrelated coordinate systems, even at equal dimensions, so a cosine between them means nothing. Project 2 encodes the model in the namespace (`knowledge:fake-embedding:v1`), so a query embedded with a new model goes to a namespace holding only that model's vectors. In production, carry Chapter 8's full embedding space fingerprint, not just the model name.

Second, **a record without an ACL must not mean "public".** The `VectorRecord` validator rejects an empty `acl_groups`, and the loader refuses a file without `acl_groups` in its front matter. Public documents say `["all"]` explicitly.

### Distance, normalization, and what a score means

> **Deep dive.** Operational rules for metrics and scores on top of Chapter 8's math; skip on a first reading.

Use the metric the embedding model was trained for, almost always cosine or dot product (Chapter 8). With vectors L2-normalized at write time, cosine and dot product rank identically and dot product is cheaper, so `NumpyVectorStore` normalizes on write. In pgvector the operator class (`vector_cosine_ops`) must match the `ORDER BY` operator (`<=>`), or the planner ignores the index.

A similarity score is a ranking signal, not a relevance probability: a cosine of 0.82 is not "82% relevant", and its meaning shifts with model, query length, and corpus. Raw thresholds ("drop hits below 0.75") break when the model changes. Put any cut-off after a reranker (Chapter 12), calibrated on judged queries (Chapter 14).

### Exact search, and when it is enough

Exact (brute-force, flat) search scores the query against every candidate and keeps the top k. It has perfect recall, no index to tune, instant inserts and deletes, and trivially correct filters. Its cost is linear in the vectors scored. On a laptop with NumPy (illustrative):

| Vectors (768 dims, float32) | Matrix size | Exact top-10, one query |
|---|---|---|
| 100,000 | 0.3 GB | about 4.5 ms |
| 1,000,000 | 3.1 GB | about 63 ms |

**For corpora up to roughly a million vectors with modest query rates, exact search on one machine is a legitimate production design.** Northwind's production knowledge base, a few thousand documents in tens of thousands of chunks, sits two orders of magnitude below that, as do most internal assistants.

Exact search stops being enough when the corpus outgrows one machine's memory, when query rate times per-query cost exceeds your CPU budget (63 ms is about 16 queries per second per scanning process), or when retrieval's share of a 2-second time-to-first-token budget is tighter than the scan (Chapter 30). The scan is memory-bandwidth bound, so batching queries into one matrix product, or a GPU, helps more than extra cores; try that before adopting an ANN index for a merely bursty workload.

### Approximate nearest neighbor search: the trade-off space

An ANN index routes a query toward its neighborhood and scores only the candidates there. Every method trades among four quantities:

- **Recall**: the fraction of the true top-k the index returns (ANN recall, measured against exact search).
- **Query latency**: vectors scored and memory touched per query.
- **Memory**: the vectors plus the index structure (graph edges, centroids, codebooks).
- **Build and update cost**: build time, insert and delete cost, retraining.

The three families below sit at different points, and production systems often combine them.

### HNSW: a navigable graph

Hierarchical Navigable Small World (HNSW) graphs are the most widely used ANN index; pgvector offers them alongside IVFFlat. The bottom layer links every vector to up to `2 * M` near neighbors. Each higher layer holds a random, exponentially smaller subset of the one below: sparse highways on top, a dense street map at the bottom. Search walks greedily down the layers, then runs a best-first search at the bottom with a candidate list of size `ef_search`. The top k of that list is the answer.

```text
# pseudocode: HNSW query
entry = top_layer_entry_point
for layer in top_layer .. 1:
    entry = greedy_closest(query, entry, layer)            # walk until no neighbor is closer
candidates = best_first(query, entry, layer=0, size=ef_search)
return closest k of candidates
```

Three parameters set the trade-off:

| Parameter | When set | Raises | Costs |
|---|---|---|---|
| `M` (max neighbors per node) | build | recall, robustness on hard data | memory (edges per vector), build time |
| `ef_construction` | build | graph quality, hence recall at a given `ef_search` | build time only |
| `ef_search` | per query | recall | latency, roughly linearly |

pgvector's defaults are `m = 16`, `ef_construction = 64`, and `hnsw.ef_search = 40`: a starting point, not a recommendation. Fix `M` and `ef_construction` at moderate values, build once, then sweep `ef_search` while measuring ANN recall against exact search on real queries; raise `ef_construction` only if no reasonable `ef_search` reaches your target. Without iterative scans, `ef_search` caps the result count: 50 results with `ef_search = 40` is impossible.

HNSW gives high recall at low latency with no training step and handles inserts well. Its costs are memory (full vectors plus edge lists, in RAM), deletes (marked nodes are still traversed until a vacuum or rebuild, so heavy churn degrades recall and latency), and build time (each insert runs a search, so millions of vectors take minutes to hours).

### IVF: partition, then probe

Think of bucketing a phone book by city and searching only the nearest few cities. Inverted file (IVF) indexes use k-means to pick `n_lists` center points (centroids) and store each vector in the list of its nearest centroid. A query is compared with all centroids, and only the `nprobe` closest lists are scanned: with 128 lists and `nprobe = 8`, about 8/128 of the vectors.

Project 2's teaching-size IVF index (`semsearch/domain/ivf.py`) is swept against exact search on synthetic clustered vectors (20,000 vectors, 64 dimensions, 128 lists, 200 queries) by `semsearch bench-ann` (illustrative):

| nprobe | ANN recall@10 | Vectors scanned | ms/query |
|---|---|---|---|
| 1 | 0.541 | 1.0% | 0.033 |
| 2 | 0.758 | 1.7% | 0.038 |
| 4 | 0.908 | 3.3% | 0.048 |
| 8 | 0.983 | 6.5% | 0.065 |
| 16 | 0.999 | 12.9% | 0.101 |
| 32 | 1.000 | 25.5% | 0.181 |

Exact search on the same data took 0.283 ms per query. Recall climbs steeply then flattens while cost grows linearly; the knee, around `nprobe = 8`, gives 98% recall for 6.5% of the work. That shape is typical, so tune against a recall target: the cheapest setting that meets it on your data.

IVF needs representative training data. If the distribution drifts (a new product line, language, or document type), new vectors pile into a few lists and recall and latency degrade until a rebuild retrains it; create pgvector's IVFFlat index only after the table holds representative data. In exchange IVF builds fast, adds little memory, and suits compression and disk storage.

### Product quantization: compress the vectors

> **Deep dive.** How very large corpora fit in memory; skip on a first reading.

At large scale memory, not compute, is the binding constraint. Product quantization (PQ) works like a color palette: it splits each vector into `m` sub-vectors and stores, for each, the index of its nearest entry in a small learned codebook (typically 256 entries, one byte). A 768-dimensional float32 vector (3,072 bytes) as 96 sub-vectors becomes 96 bytes, a 32x reduction (illustrative arithmetic). Scoring uses precomputed query-to-centroid tables, a handful of lookups per vector.

The price is a noisy ranking among close candidates. The remedy is **rescoring**: fetch a larger candidate set (say 100) from the compressed index, re-rank it with full-precision vectors kept on disk or a cheaper tier, and return the top 10. Simpler relatives are scalar quantization (int8 or float16, 2x to 4x smaller, small recall loss) and binary quantization (one bit per dimension, 32x smaller, only with rescoring and only for some models); pgvector offers `halfvec` and `bit` types. Common combinations are IVF-PQ and HNSW over quantized vectors with rescoring; disk-resident graph indexes (DiskANN is the best-known design) keep only compressed vectors in RAM. Northwind needs none of this, but when a sizing exercise says "400 GB of vectors", the answer is "quantize and rescore", not "buy 400 GB of RAM".

### Measuring ANN recall against exact search

Every ANN number in this chapter is relative to exact search with the same vectors and filters, which gives you an automatable test:

1. Take a sample of real queries (or the questions in your retrieval gold set).
2. For each, run the production search and an exact search with identical filters and k.
3. Compute overlap@k: the fraction of the exact top-k that the ANN search also returned.

Project 2 implements this as `semsearch ann-check`, using `VectorStore.search(..., exact=True)`, which `PgVectorStore` implements by disabling index scans for that transaction.

Keep two recall numbers apart. **ANN recall** asks whether the index found what exact search would have found. **Retrieval recall** asks whether the search found what a human judged relevant. If retrieval recall is 0.70 and ANN recall is 0.99, tuning the index cannot help; the problem is the embedding model, the chunking, or the query (Chapters 11 and 12). If retrieval recall drops from 0.85 with exact search to 0.70 with the index, the index costs you 15 points and `ef_search` or `nprobe` is the knob.

### Metadata filtering: pre-filter, post-filter, and in-index

Every Northwind query carries a tenant filter (the user's tenant plus `shared`) and an ACL filter (the user's groups must intersect the chunk's). There are three ways to combine a filter with nearest-neighbor search.

**Pre-filtering** restricts the candidates to allowed rows, then searches among them. With exact search this is simple and always correct; `NumpyVectorStore` uses a mask. With an ANN index, built over all vectors, you either scan the allowed subset exhaustively (fine when small) or traverse the graph skipping disallowed nodes (which can disconnect it under very selective filters).

**Post-filtering** takes the index's top N candidates and drops those that fail the filter. It is the default in many engines and in a naive SQL query against an HNSW index, where the `WHERE` clause is applied to the `ef_search` candidates. The failure is arithmetic. Call the fraction of rows a filter passes its selectivity s (smaller s means more selective). From N candidates you expect about `N * s` survivors, capped at k. `semsearch bench-filter` measures it with N = 40 (playing `ef_search`) and k = 10 (illustrative; small deviations are sampling noise):

| Filter selectivity | Post-filter results (of 10) | Pre-filter results (of 10) |
|---|---|---|
| 50% | 10.00 | 10.00 |
| 10% | 4.21 | 10.00 |
| 2% | 0.82 | 10.00 |
| 0.5% | 0.17 | 10.00 |

At 2% selectivity, typical for "logistics tenant and `it-oncall` group" in a large multi-tenant corpus, post-filtering returns under one result where ten exist. Nothing errors; the assistant abstains or answers from the wrong evidence.

**In-index (filter-aware) search** makes the index respect the filter: count only allowed nodes, keep scanning until k allowed results are found, keep per-value sub-indexes, or switch to an exact scan when the allowed set is small. pgvector 0.8 and later implement "keep scanning" as iterative index scans (`hnsw.iterative_scan`; its `relaxed_order` mode may return rows slightly out of order, so the adapter re-sorts). Dedicated databases differ widely on highly selective filters; test that when you evaluate one.

Rules for filtering:

- **Know your selectivity distribution.** Log how many rows each query's filter allows; the smallest tenants and most restricted groups break first.
- **Route by selectivity.** When the filter allows a few thousand rows, an exact scan through a B-tree or GIN index (PostgreSQL's inverted index for arrays and JSONB) on the filter columns is faster and correct. PostgreSQL's planner chooses this itself when statistics are good.
- **Partition on the dominant filter.** Give large tenants their own partition or index so the tenant filter leaves the ANN problem.
- **Alert on underfilled results.** Project 2's search span records `underfilled` when fewer than k results come back; a rising rate for one tenant is the signature of a filtering problem.

### Namespaces, collections, and tenants

Three isolation models cover most designs.

| Model | How | Isolation | Cost and limits | Fits |
|---|---|---|---|---|
| Shared index, tenant column | Every query filters on `tenant` | Logical only; a missing filter leaks | Cheapest; filtered-ANN problems for small tenants | Many small tenants, internal tools |
| Namespace or partition per tenant | Separate index per tenant in one store | Strong; no tenancy filter needed | Per-namespace overhead | Tens to thousands of mid-size tenants |
| Store or database per tenant | Separate cluster, database, or schema | Physical; separate keys, backups, deletion | Highest operational cost | Regulated or very large tenants |

Hybrids are common: a shared index for the long tail, dedicated namespaces for the largest tenants, dedicated stores where contracts demand it. Whatever you choose, enforce tenancy in one code path, derive the tenant from authenticated identity rather than the request body, and run a tenant-leak test in CI.

Project 2 uses namespaces for a different axis: **index identity**. `knowledge:<embedding model>:<index version>` separates indexes built with different models or chunking, which makes blue-green re-indexing possible (see Indexing strategies and rebuilds). Tenancy is a filtered `tenant` column plus ACL groups, because shared documents must be visible to both tenants, and the `(namespace, tenant)` index lets PostgreSQL serve small tenants with a filtered scan.

Both filters are mandatory. A logistics user sees `tenant IN ('logistics', 'shared')`, so shared documents rely on their ACL alone: the database failover runbook is `tenant: shared` but readable only by `it-oncall`, and a tenant filter alone would leak it to every employee. Chapter 15 owns authorization in retrieval as a whole; this chapter covers the mechanics that keep the filters correct inside a vector search.

## How it works

### The life of a search

A retail support agent asks "how long can a store keep selling while offline". The authenticating gateway sets `X-User`, `X-Tenant: retail`, and `X-Groups: all,retail` and forwards the request to `/search`:

1. **Identity becomes a principal.** `get_principal` builds a `Principal` from the headers, or returns 401 if any is missing or the group list is empty.
2. **The principal becomes a filter.** `SearchService.authorization_filter` produces `tenants = ("retail", "shared")` and `acl_groups = ("all", "retail")`. The request body may add `tags` to narrow the search; nothing in the body can widen it.
3. **The query is embedded** with the same embedding client, and therefore the same model, that built the namespace.
4. **The store searches** the namespace with the vector, k, and the filter: a mask over allowed rows in NumPy, one parameterized SQL statement in pgvector, with `hnsw.ef_search` and iterative scanning set for this transaction only.
5. **The span is recorded** with tenant, groups, latencies, result count, `underfilled`, top score, and document ids: what you need when someone reports "search did not find X".

### The life of a write

Most vector-store bugs live in writes, because derived caches go stale. Project 2 writes per document:

1. The loader parses the front matter and computes a content hash. The document's **index version** is the declared version plus the first eight hex digits of the hash, so an edit that forgets to bump the version still produces a new index version.
2. If the store's `doc_versions` already holds that index version, the document is skipped at no cost, so frequent re-runs are cheap.
3. A new or changed document is chunked, embedded in batches, and passed to `replace_document(namespace, doc_id, index_version, records)`.
4. `replace_document` atomically makes the new chunks the only chunks of that document: in PostgreSQL, one transaction takes an advisory lock on the document (an application-level lock keyed on its id, so two ingestions serialize), upserts the new rows, and deletes the rest. Readers see the old version or the new one. Two separate transactions would open a window where the document is missing or where both versions are retrievable.
5. With `--prune`, documents no longer in the source are deleted. Without it, a revoked policy stays searchable forever; a test pins that behavior so nobody mistakes it for a feature.

### Consistency: what a reader can observe

> **Deep dive.** What a reader can see right after a write, and how to get read-your-writes; skip on a first reading.

Dedicated databases often acknowledge a write before it is searchable, because index updates are applied asynchronously or in background-merged segments; replicas and caches lag further. PostgreSQL with pgvector gives read-after-write on the primary for free, because the HNSW index is updated inside the inserting transaction.

A knowledge base re-indexed every 15 minutes tolerates seconds of lag. A user who uploads a file and asks about it at once needs read-your-writes: route their queries to the primary briefly, wait on the store's visibility token if it has one, or search the fresh document directly. Deletion must never lag silently: measure time-to-unretrievable for legal or permission deletions and alert past the agreed bound.

## Architecture

The first diagram shows Project 2's two paths and where trust changes: documents and queries are untrusted text, and identity comes only from the gateway.

```mermaid
flowchart LR
    subgraph Sources["Untrusted content"]
        D[Markdown docs with front matter]
    end
    subgraph Ingest["Ingestion job"]
        L[load and parse front matter] --> H{index version changed?}
        H -- no --> SK[skip]
        H -- yes --> C[chunk paragraphs] --> E1[embed batch]
        E1 --> R[replace_document in one transaction]
        P[prune missing docs]
    end
    subgraph Store["Vector store: namespace = index:model:version"]
        V[(records: vector, tenant, acl, doc, version, tags)]
    end
    subgraph Edge["Trust boundary: gateway"]
        G[authenticate user] --> HD[X-User, X-Tenant, X-Groups]
    end
    subgraph API["Search service"]
        PR[principal] --> F[authorization filter]
        Q[query text] --> E2[embed query]
        F --> S[filtered top-k]
        E2 --> S
        S --> T[span: underfilled, scores, doc ids]
    end
    D --> L
    R --> V
    P --> V
    HD --> PR
    S <--> V
```

The second diagram shows the lifecycle of an index version. A new model, chunker, or index parameters each create a new namespace, built and verified before any traffic sees it.

```mermaid
stateDiagram-v2
    [*] --> Building: new model, chunker, or index params, dual-write on
    Building --> Verifying: backfill complete
    Verifying --> Building: recall or leak gate fails
    Verifying --> Active: gates pass, alias switched
    Active --> Draining: next version activated
    Draining --> Active: rollback, alias switched back
    Draining --> Retired: rollback window over
    Retired --> [*]: namespace dropped
```

### Indexing strategies and rebuilds

Four situations force a new namespace or index, at different costs.

**The embedding model changes.** Every vector must be recomputed. This is the expensive case, so plan it as a migration:

1. Build the new namespace in the background while the old one serves traffic; the backfill re-embeds the corpus into it.
2. Dual-write: from the moment the backfill starts, ingestion writes every changed document to both namespaces. Backfills take hours while documents keep changing, so this step is required.
3. Verify with the retrieval gold set and the leak gate against the new namespace.
4. Switch the active namespace with one configuration change (in Project 2, `INDEX_VERSION` or the embedding model setting; in a larger system, an alias table the service reads).
5. Keep the old namespace, and keep dual-writing to it, until the rollback window closes.

A new model often has different dimensions, and pgvector's column type is `vector(d)`, so the new namespace needs its own table (a different `PG_TABLE`); plan disk and memory for both indexes during the overlap. Estimate the backfill's cost and wall-clock time first (Chapter 8's `plan_reembed` does this arithmetic).

**The chunker or enrichment changes.** Chunk ids change, so this is also a full rebuild, cheaper when embeddings are cached by text hash (`CachedEmbeddings` in `aie_core`).

**Index parameters change.** A new `M` or `ef_construction`, IVFFlat to HNSW, or quantization rebuilds only the index. In PostgreSQL, `CREATE INDEX CONCURRENTLY` on the new definition, then dropping the old one, avoids blocking writes; `PgVectorStore.reindex()` wraps `REINDEX INDEX CONCURRENTLY` for the same definition.

**The index has degraded.** Heavy churn leaves HNSW graphs full of deleted nodes and IVF centroids stale: latency rises at a fixed `ef_search` and `ann-check` recall falls. Rebuild on a schedule or a metric threshold, not when users complain.

For initial loads and full rebuilds, load the table first and create the HNSW index afterwards with a generous `maintenance_work_mem`; inserting row by row into an existing index is many times slower. Small incremental runs can insert into the live index.

## Implementation

Project 2 lives in `book/projects/p2-semantic-search/` and uses `aie_core` for embeddings, settings, and tracing.

```text
p2-semantic-search/
  pyproject.toml  .env.example  Dockerfile  docker-compose.yml  README.md
  sql/schema.sql                 rendered pgvector schema, kept in sync by a test
  semsearch/
    config.py                    SearchSettings, make_embedder, make_store, namespace_for
    service.py                   Principal, SearchService
    cli.py                       ingest | search | eval | ann-check | bench-ann | bench-filter | serve
    bench.py                     IVF sweep and filter experiment
    domain/                      models.py, filters.py, metrics.py, ivf.py   (pure, no I/O)
    adapters/                    base.py (protocol), numpy_store.py, pg_store.py
    ingest/                      loader.py, pipeline.py
    api/app.py                   FastAPI
    eval/run_eval.py             evaluator + fixtures/retrieval_fixture.jsonl
  tests/                         offline tests; pgvector contract tests marked integration
```

Configuration is environment-driven; the README lists every variable. The ones that matter here:

| Variable | Default | Meaning |
|---|---|---|
| `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL` | `fake`, `fake-embedding` | embedder; the model name becomes part of the namespace |
| `VECTOR_BACKEND` | `numpy` | `numpy` or `pgvector` |
| `DATABASE_URL`, `PG_TABLE` | unset, `chunks` | pgvector connection and table |
| `HNSW_M`, `HNSW_EF_CONSTRUCTION`, `HNSW_EF_SEARCH` | `16`, `64`, `100` | index build and query parameters |
| `HNSW_ITERATIVE_SCAN` | `relaxed_order` | filtered-scan behavior on pgvector 0.8+ |
| `PG_STATEMENT_TIMEOUT_MS` | unset | per-transaction cap on pgvector reads |
| `INDEX_NAME`, `INDEX_VERSION` | `knowledge`, `v1` | namespace identity |
| `DOCS_DIR`, `CHUNK_MAX_CHARS` | shared-data docs, `1200` | corpus and chunk budget |

Run it:

```bash
uv pip install --python .venv/bin/python -e book/projects/aie_core -e "book/projects/p2-semantic-search[pg,dev]"
cd book/projects/p2-semantic-search
semsearch ingest
semsearch search "NorthGate VPN error 412" --tenant shared --groups all
semsearch eval
semsearch bench-ann && semsearch bench-filter
semsearch serve --port 8000
# with pgvector
docker compose up -d db
VECTOR_BACKEND=pgvector DATABASE_URL=postgresql://northwind:northwind@localhost:5432/northwind semsearch ingest --prune
python -m pytest -q
```

### The record and the filter

The domain model has no I/O: adapters store `VectorRecord`s, and queries carry a `SearchFilter`.

```python
# path: book/projects/p2-semantic-search/semsearch/domain/models.py (excerpt; full file on disk)
_NAMESPACE_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")


def make_namespace(index_name: str, embedding_model: str, index_version: str) -> str:
    """The embedding model is part of the index identity: vectors from two models never mix."""
    model = re.sub(r"[^a-z0-9_.-]+", "-", embedding_model.lower()).strip("-")
    return f"{index_name}:{model}:{index_version}"


def chunk_id(doc_id: str, doc_version: str, ordinal: int) -> str:
    """Deterministic, so re-ingesting the same content upserts instead of duplicating."""
    return f"{doc_id}@{doc_version}#{ordinal:04d}"


class VectorRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    namespace: str
    tenant: str
    doc_id: str
    doc_version: str
    ordinal: int = Field(ge=0)
    text: str
    vector: Vector
    acl_groups: tuple[str, ...]
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding_model: str

    @field_validator("namespace")
    @classmethod
    def _valid_namespace(cls, v: str) -> str:
        if not _NAMESPACE_RE.match(v):
            raise ValueError(f"invalid namespace {v!r}")
        return v

    @field_validator("acl_groups")
    @classmethod
    def _acl_not_empty(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        # A record nobody may read is a bug; a record with no ACL must not mean "everyone".
        if not v:
            raise ValueError("acl_groups must not be empty; use ('all',) for public documents")
        return v


class SearchFilter(BaseModel):
    """Conjunction of constraints. None means 'no constraint on this field'.

    tenants: record.tenant must be one of these.
    acl_groups: record.acl_groups must intersect these (the caller's groups).
    tags_any: record.tags must intersect these.
    doc_ids: record.doc_id must be one of these.
    """

    model_config = ConfigDict(frozen=True)

    tenants: tuple[str, ...] | None = None
    acl_groups: tuple[str, ...] | None = None
    tags_any: tuple[str, ...] | None = None
    doc_ids: tuple[str, ...] | None = None

    def is_empty(self) -> bool:
        return all(v is None for v in (self.tenants, self.acl_groups, self.tags_any, self.doc_ids))
```

`SearchHit` (a search result with its cosine score) and `DocVersion` (what the store believes is indexed for one document) complete the module on disk. Filter semantics are defined once, in pure Python, so the NumPy adapter uses them directly and the pgvector adapter's SQL can be tested against them.

```python
# path: book/projects/p2-semantic-search/semsearch/domain/filters.py
"""Filter semantics in one place so every adapter and every test agrees on them."""
from __future__ import annotations

from .models import SearchFilter, VectorRecord


def matches(record: VectorRecord, f: SearchFilter | None) -> bool:
    if f is None:
        return True
    if f.tenants is not None and record.tenant not in f.tenants:
        return False
    if f.acl_groups is not None and not set(record.acl_groups) & set(f.acl_groups):
        return False
    if f.tags_any is not None and not set(record.tags) & set(f.tags_any):
        return False
    if f.doc_ids is not None and record.doc_id not in f.doc_ids:
        return False
    return True


def visible_tenants(user_tenant: str, shared_tenant: str = "shared") -> tuple[str, ...]:
    """A tenant user sees their own tenant plus the cross-tenant 'shared' corpus."""
    if user_tenant == shared_tenant:
        return (shared_tenant,)
    return (user_tenant, shared_tenant)


__all__ = ["matches", "visible_tenants"]
```

### The VectorStore protocol

The protocol names what the application needs. `replace_document` is ingestion's only write path; `exact=True` lets an approximate store be measured against itself.

```python
# path: book/projects/p2-semantic-search/semsearch/adapters/base.py (excerpt)
@runtime_checkable
class VectorStore(Protocol):
    dimensions: int

    def upsert(self, records: Sequence[VectorRecord]) -> int:
        """Insert or replace records by id. Returns the number written."""
        ...

    def replace_document(self, namespace: str, doc_id: str, doc_version: str, records: Sequence[VectorRecord]) -> int:
        """Atomically make `records` the only chunks of `doc_id` in `namespace`.

        Chunks of any other version of the document are removed in the same operation, so a
        reader sees either the old version or the new one, never neither and never both.
        """
        ...

    def delete_document(self, namespace: str, doc_id: str) -> int:
        """Remove every chunk of a document. Returns the number removed."""
        ...

    def search(
        self,
        namespace: str,
        query: Vector,
        k: int,
        flt: SearchFilter | None = None,
        *,
        exact: bool = False,
    ) -> list[SearchHit]:
        """Top-k by cosine similarity among records that satisfy `flt`.

        `exact=True` forces a brute-force scan; it exists so that an approximate index can
        be measured against ground truth. Adapters that are always exact ignore it.
        """
        ...

    def doc_versions(self, namespace: str) -> dict[str, DocVersion]:
        """doc_id -> what is currently indexed. Drives incremental ingestion and pruning."""
        ...

    def count(self, namespace: str | None = None) -> int: ...

    def namespaces(self) -> list[str]: ...
```

### The exact NumPy adapter

The excerpts show a lazily materialized matrix, atomic replacement under a lock, and pre-filtered exact search.

```python
# path: book/projects/p2-semantic-search/semsearch/adapters/numpy_store.py (excerpt)
@dataclass
class _Namespace:
    records: dict[str, VectorRecord] = field(default_factory=dict)
    # materialized view, rebuilt when dirty
    ids: list[str] = field(default_factory=list)
    matrix: np.ndarray | None = None
    dirty: bool = True

    def materialize(self, dimensions: int) -> None:
        if not self.dirty:
            return
        self.ids = list(self.records.keys())
        if self.ids:
            self.matrix = np.vstack([_unit(self.records[i].vector) for i in self.ids]).astype(np.float32)
        else:
            self.matrix = np.zeros((0, dimensions), dtype=np.float32)
        self.dirty = False


class NumpyVectorStore:
    """Exact cosine search. Good to roughly a million vectors per namespace on one machine,
    as long as the matrix fits in RAM (n * d * 4 bytes) and the query rate is modest."""

    def __init__(self, dimensions: int, persist_dir: str | Path | None = None) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self._ns: dict[str, _Namespace] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ writes
    def upsert(self, records: Sequence[VectorRecord]) -> int:
        check_dimensions(records, self.dimensions)
        with self._lock:
            for r in records:
                ns = self._ns.setdefault(r.namespace, _Namespace())
                ns.records[r.id] = r
                ns.dirty = True
        return len(records)

    def replace_document(self, namespace: str, doc_id: str, doc_version: str, records: Sequence[VectorRecord]) -> int:
        check_dimensions(records, self.dimensions)
        check_document(namespace, doc_id, doc_version, records)
        with self._lock:  # the lock makes delete+insert one step for readers
            ns = self._ns.setdefault(namespace, _Namespace())
            stale = [rid for rid, r in ns.records.items() if r.doc_id == doc_id]
            for rid in stale:
                del ns.records[rid]
            for r in records:
                ns.records[r.id] = r
            ns.dirty = True
        return len(records)
```

```python
# path: book/projects/p2-semantic-search/semsearch/adapters/numpy_store.py (excerpt)
    # ------------------------------------------------------------------ reads
    def search(
        self,
        namespace: str,
        query: Vector,
        k: int,
        flt: SearchFilter | None = None,
        *,
        exact: bool = False,  # always exact; accepted for protocol compatibility
    ) -> list[SearchHit]:
        if len(query) != self.dimensions:
            raise ValueError(f"query has {len(query)} dimensions, store expects {self.dimensions}")
        if k <= 0:
            return []
        with self._lock:
            ns = self._ns.get(namespace)
            if ns is None or not ns.records:
                return []
            ns.materialize(self.dimensions)
            ids, matrix, records = ns.ids, ns.matrix, ns.records
            assert matrix is not None
            if flt is None or flt.is_empty():
                rows = np.arange(len(ids))
            else:
                rows = np.fromiter((i for i, rid in enumerate(ids) if matches(records[rid], flt)), dtype=np.int64)
            if rows.size == 0:
                return []
            scores = matrix[rows] @ _unit(query)
            kk = min(k, rows.size)
            top = np.argpartition(-scores, kk - 1)[:kk]
            top = top[np.argsort(-scores[top], kind="stable")]
            return [self._hit(records[ids[rows[t]]], float(scores[t])) for t in top]
```

The mask is a Python loop over `matches`, fine for tens of thousands of rows; vectorize it with per-field arrays if you go much larger. Persistence (on disk) saves through temporary files and an atomic rename, and loading never unpickles (`allow_pickle=False`), because an index directory is an input file like any other.

### The pgvector adapter

The schema is rendered from a template so the vector dimensions and HNSW parameters come from configuration. Authorization fields are indexed columns, not JSONB keys, because every query uses them.

```python
# path: book/projects/p2-semantic-search/semsearch/adapters/pg_store.py (excerpt)
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
```

Filters become a parameterized `WHERE` clause. Values never enter the SQL text, and a test asserts it.

```python
# path: book/projects/p2-semantic-search/semsearch/adapters/pg_store.py (excerpt)
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
```

A document write is one transaction. The read path sets index parameters with `set_config(..., true)`, the function form of `SET LOCAL`, so they cannot leak to the next request on a pooled connection.

```python
# path: book/projects/p2-semantic-search/semsearch/adapters/pg_store.py (excerpt)
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
```

```python
# path: book/projects/p2-semantic-search/semsearch/adapters/pg_store.py (excerpt)
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
```

`_apply_timeout` sets `statement_timeout` the same way when `PG_STATEMENT_TIMEOUT_MS` is configured, so a slow search raises and the caller degrades (Chapter 12 skips a retriever that misses its deadline). The adapter holds one connection, which serializes searches; a concurrent service needs a store per worker or a pool behind it.

The adapter also has `hybrid_search`, which fuses dense candidates with full-text candidates from the generated `text_tsv` column using Reciprocal Rank Fusion in one SQL statement. Chapter 12 owns hybrid retrieval; the method is here because "pgvector plus `tsvector` in one database" is a large part of the case for PostgreSQL.

### Ingestion

> **Deep dive.** The incremental ingestion loop in code; skip on a first reading.

The loader parses the front matter subset the corpus uses, fails closed on a missing ACL, and groups paragraphs under their nearest heading up to a size budget. Chapter 11 replaces this chunker; the store does not care which chunker produced its records. The pipeline holds incrementality and pruning:

```python
# path: book/projects/p2-semantic-search/semsearch/ingest/pipeline.py (excerpt)
def ingest(
    docs: Sequence[SourceDocument],
    store: VectorStore,
    embedder: EmbeddingClient,
    namespace: str,
    *,
    max_chars: int = 1200,
    prune: bool = False,
    batch_size: int = 64,
    tracer: Tracer | None = None,
) -> IngestReport:
    tracer = tracer or NoopTracer()
    report = IngestReport(namespace=namespace)
    start = time.perf_counter()
    with tracer.span("ingest", namespace=namespace, documents=len(docs)) as span:
        indexed = store.doc_versions(namespace)
        for doc in docs:
            current = indexed.get(doc.doc_id)
            if current is not None and current.doc_version == doc.index_version:
                report.unchanged.append(doc.doc_id)
                continue
            chunks = chunk_paragraphs(doc.body, max_chars=max_chars)
            texts = [embedding_text(doc, c) for c in chunks]
            vectors: list[list[float]] = []
            for i in range(0, len(texts), batch_size):
                vectors.extend(embedder.embed(texts[i : i + batch_size]))
            report.texts_embedded += len(texts)
            records = build_records(doc, list(zip(texts, vectors)), namespace, embedder.model, [c.section for c in chunks])
            report.chunks_written += store.replace_document(namespace, doc.doc_id, doc.index_version, records)
            (report.updated if current is not None else report.added).append(doc.doc_id)
        if prune:
            source_ids = {d.doc_id for d in docs}
            for doc_id in sorted(set(indexed) - source_ids):
                store.delete_document(namespace, doc_id)
                report.deleted.append(doc_id)
        report.seconds = time.perf_counter() - start
        span.set_attribute("added", len(report.added))
        span.set_attribute("updated", len(report.updated))
        span.set_attribute("deleted", len(report.deleted))
        span.set_attribute("texts_embedded", report.texts_embedded)
    save = getattr(store, "save", None)
    if callable(save) and getattr(store, "persist_dir", None) is not None:
        save()
    return report
```

### The search service and API

> **Deep dive.** The code behind steps 1, 2, and 5 of the life of a search; skip on a first reading.

Authorization is derived from the principal in one method and applied on every query. The API only turns trusted headers into a principal and refuses requests without one.

```python
# path: book/projects/p2-semantic-search/semsearch/service.py (excerpt)
class SearchService:
    def __init__(
        self,
        store: VectorStore,
        embedder: EmbeddingClient,
        namespace: str,
        *,
        default_k: int = 8,
        max_k: int = 50,
        tracer: Tracer | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.namespace = namespace
        self.default_k = default_k
        self.max_k = max_k
        self.tracer = tracer or NoopTracer()

    def authorization_filter(self, principal: Principal, tags: list[str] | None = None) -> SearchFilter:
        return SearchFilter(
            tenants=visible_tenants(principal.tenant),
            acl_groups=tuple(principal.groups),
            tags_any=tuple(tags) if tags else None,
        )

    def search(self, query: str, principal: Principal, *, k: int | None = None, tags: list[str] | None = None) -> SearchResult:
        k = min(k or self.default_k, self.max_k)
        flt = self.authorization_filter(principal, tags)
        start = time.perf_counter()
        with self.tracer.span(
            "search",
            namespace=self.namespace,
            k=k,
            tenant=principal.tenant,
            groups=list(principal.groups),
            tags=tags or [],
            embedding_model=self.embedder.model,
        ) as span:
            t0 = time.perf_counter()
            qvec = self.embedder.embed_query(query)
            span.set_attribute("embed_ms", round((time.perf_counter() - t0) * 1000, 2))
            t1 = time.perf_counter()
            hits = self.store.search(self.namespace, qvec, k, flt)
            span.set_attribute("store_ms", round((time.perf_counter() - t1) * 1000, 2))
            span.set_attribute("returned", len(hits))
            # Fewer hits than k under a filter is the signature of an over-selective filter
            # or an ANN index that ran out of candidates; alert on its rate.
            span.set_attribute("underfilled", len(hits) < k)
            span.set_attribute("top_score", hits[0].score if hits else None)
            span.set_attribute("doc_ids", [h.doc_id for h in hits])
        return SearchResult(
            namespace=self.namespace, query=query, k=k, took_ms=round((time.perf_counter() - start) * 1000, 2), hits=hits
        )
```

```python
# path: book/projects/p2-semantic-search/semsearch/api/app.py (excerpt)
def get_principal(
    x_user: Annotated[str | None, Header()] = None,
    x_tenant: Annotated[str | None, Header()] = None,
    x_groups: Annotated[str | None, Header()] = None,
) -> Principal:
    if not x_user or not x_tenant or not x_groups:
        # Fail closed: no identity, no search. Never default to "all".
        raise HTTPException(status_code=401, detail="missing identity headers")
    groups = tuple(g.strip() for g in x_groups.split(",") if g.strip())
    if not groups:
        raise HTTPException(status_code=401, detail="empty group list")
    return Principal(user_id=x_user, tenant=x_tenant.strip(), groups=groups)
```

### Evaluation

> **Deep dive.** The retrieval evaluator and leak gate in code; skip on a first reading.

The evaluator reports recall@k, hit rate, and MRR (Chapter 10 defines them) by running every gold question through the API's `SearchService` as that row's principal. It requests four times the largest k in chunks so enough distinct documents survive the collapse to documents. `forbidden-doc` rows are leak checks, and the CLI exits non-zero on any leak, which makes it a CI gate.

```python
# path: book/projects/p2-semantic-search/semsearch/eval/run_eval.py (excerpt)
def evaluate(service: SearchService, gold: Sequence[GoldRow], ks: tuple[int, ...] = (1, 3, 5, 10), gold_path: str = "") -> EvalReport:
    report = EvalReport(gold_path=gold_path, ks=ks)
    kmax = max(ks)
    indexed = set(service.store.doc_versions(service.namespace))
    recalls: dict[int, list[float]] = defaultdict(list)
    hits: dict[int, list[float]] = defaultdict(list)
    rrs: list[float] = []
    by_tag: dict[str, list[float]] = defaultdict(list)
    for row in gold:
        principal = Principal(user_id=f"eval:{row.id}", tenant=row.tenant, groups=tuple(row.user_groups))
        # Ask for more chunks than kmax so that kmax *documents* survive the collapse.
        result = service.search(row.question, principal, k=min(service.max_k, kmax * 4))
        ranked = unique_in_order(h.doc_id for h in result.hits)
        if row.forbidden:
            report.forbidden_rows += 1
            if set(row.required_doc_ids) & set(ranked):
                report.leaks.append(row.id)
            continue
        missing = [d for d in row.required_doc_ids if d not in indexed]
        if missing:
            report.ingestion_gaps.append(f"{row.id}:{','.join(missing)}")
            continue
        report.rows += 1
        for k in ks:
            r = recall_at_k(ranked, row.required_doc_ids, k)
            recalls[k].append(r)
            hits[k].append(1.0 if r > 0 else 0.0)
        rrs.append(reciprocal_rank(ranked, row.required_doc_ids))
        r_max = recalls[kmax][-1]
        for tag in row.tags:
            by_tag[tag].append(r_max)
        if r_max < 1.0:
            report.misses.append({"id": row.id, "required": row.required_doc_ids, "got": ranked[:kmax]})
    if report.rows:
        report.recall = {k: mean(recalls[k]) for k in ks}
        report.hit_rate = {k: mean(hits[k]) for k in ks}
        report.mrr = mean(rrs)
        report.recall_by_tag = {t: mean(v) for t, v in by_tag.items()}
    else:
        report.recall = {k: 0.0 for k in ks}
        report.hit_rate = {k: 0.0 for k in ks}
    return report
```

### The ANN and filter experiments

> **Deep dive.** The code behind the IVF and post-filter tables; skip on a first reading.

The IVF index is about 80 lines of NumPy: spherical k-means for training, then centroid routing and an exhaustive scan of the probed lists.

```python
# path: book/projects/p2-semantic-search/semsearch/domain/ivf.py (excerpt)
    def search(self, query: np.ndarray, k: int, nprobe: int) -> tuple[np.ndarray, int]:
        """Returns (row ids of the approximate top-k, number of vectors scored)."""
        if self.centroids is None or self.vectors is None:
            raise RuntimeError("index is not trained")
        q = np.asarray(query, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        nprobe = max(1, min(nprobe, self.n_lists))
        probe = np.argsort(-(self.centroids @ q))[:nprobe]
        candidates = np.concatenate([self.lists[c] for c in probe])
        if candidates.size == 0:
            return np.array([], dtype=np.int64), 0
        local = exact_top_k(self.vectors[candidates], q, k)
        return candidates[local], int(candidates.size)
```

`semsearch/bench.py` generates clustered synthetic vectors (real embeddings are clustered, which matters for IVF), sweeps `nprobe`, and runs the post-filter experiment.

### The store contract test

> **Deep dive.** How one test file pins every adapter's behavior; skip on a first reading.

One test file defines a correct vector store and runs against every adapter (pgvector when `DATABASE_URL` is set and the `integration` marker is selected).

```python
# path: book/projects/p2-semantic-search/tests/test_store_contract.py (excerpt)
def test_tenant_filter_excludes_other_tenants(store, vocab_embedder):
    q = vocab_embedder.embed_query("api error")
    hits = store.search(NS, q, k=10, flt=SearchFilter(tenants=("retail", "shared")))
    assert "retail-returns" in ids(hits)
    assert "logistics-tracking" not in ids(hits)


def test_acl_filter_hides_restricted_documents(store, vocab_embedder):
    q = vocab_embedder.embed_query("incident sev1")
    visible = ids(store.search(NS, q, k=10, flt=SearchFilter(acl_groups=("all",))))
    assert "sev1-runbook" not in visible
    oncall = ids(store.search(NS, q, k=10, flt=SearchFilter(acl_groups=("all", "it-oncall"))))
    assert oncall[0] == "sev1-runbook"


def test_selective_filter_still_returns_matches(store, vocab_embedder):
    # The only allowed record is far from the query; a post-filtered ANN list would miss it.
    q = vocab_embedder.embed_query("refund policy deadline")
    hits = store.search(NS, q, k=3, flt=SearchFilter(tags_any=("api",), tenants=("logistics",)))
    assert ids(hits) == ["logistics-tracking"]
```

## Code walkthrough

> **Deep dive.** What the code does on a real run, and the numbers to watch; skip on a first reading.

How it works traced one query and one document; two details remain.

**Ingestion is checked by its second run.** `load_corpus` rejects duplicate document ids, because two files claiming one id would overwrite each other's chunks. On the Northwind corpus the first run adds 24 documents and 246 chunks (246 because this chapter's paragraph chunker runs at `CHUNK_MAX_CHARS=1200`; later chapters chunk the same 24 documents differently and report other counts). The second run embeds nothing. If a no-op run re-embeds anything, your hashes are unstable (a timestamp in the text, nondeterministic parsing) and you are paying for embeddings you do not need.

**The planner chooses the path.** In pgvector, `ORDER BY embedding <=> query` uses the HNSW index when the filter is broad and a filtered scan through the `(namespace, tenant)` or GIN indexes when it is selective.

## Production considerations

> **Deep dive.** Latency, sizing, security, replication, and deletion details behind the Before you ship checklist; skip on a first reading.

**Latency.** The span separates query embedding (often the larger part with a remote embedder, tens of milliseconds, illustrative) from the store call, because the fixes differ: cache query embeddings for the first, tune index parameters and filters for the second. A p95 store latency creeping up over weeks usually means index degradation or a filter that has tipped the planner off the index.

**Memory.** Size the index before choosing it. Raw float32 vectors cost `n * d * 4` bytes: 10 million 1,024-dimensional vectors are 41 GB before index structure (illustrative). Half precision halves that; quantization with rescoring cuts it by an order of magnitude or more.

**Security.** Treat the store as a copy of the documents: embeddings can leak source text through inversion attacks, and the `text` column is the source text. Apply the source system's access controls, encryption, and retention, and keep the store off the public network. Parameterize all SQL and validate identifiers such as table names, as `render_schema` in `pg_store.py` does.

**Replication and recovery.** HNSW indexes are large, so a new replica or a restore takes longer than the row count suggests. The index is derived data, so rebuild-from-source time is your worst-case recovery time.

**Deletion.** A deleted row is not unretrievable at once. PostgreSQL keeps its index entries until vacuum, where they consume HNSW candidate slots and underfill results; many dedicated stores apply tombstones (delete markers) at the next segment merge. For compliance deletions, verify with a query, purge every derived copy (caches, evaluation snapshots, logs with retrieved text), and record the deletion.

## Common mistakes

Mixing models, post-filtering, and skipping pruning appear in Failure modes; thresholds on raw scores, `ef_search` below k, and index build order are covered in Core concepts and Indexing strategies and rebuilds.

- **Adopting a dedicated vector database for 50,000 chunks.** The team gains a stateful system to secure, back up, and keep in sync, for latency it did not need.
- **Tuning ANN parameters before measuring exact-search recall.** Without the exact baseline, nobody knows whether a recall problem is the index or the embeddings.
- **Chunk ids that are not deterministic.** Random UUIDs per ingestion run turn every re-ingest into duplicates that crowd out other documents in the top k.

## Failure modes

| Failure | How it shows up | Telemetry that reveals it | Test |
|---|---|---|---|
| Filtered-ANN starvation | Small tenant or restricted group gets few or no results; assistant abstains on answerable questions | `underfilled = true` rate by tenant; `returned` well below k | Selective filter whose only match is far from the query |
| Model mismatch | Relevance collapses after a deploy; scores cluster in a narrow band | Span `embedding_model` differs from namespace model; top-score distribution shifts | Startup check that query model equals namespace model |
| Stale index | Answers cite a superseded policy version | Ingestion lag; store versions differ from source | Edit content without bumping version, assert a new `doc_version` |
| Zombie documents | Deleted or reclassified documents still retrieved | Result doc ids absent from the source inventory | Delete source, ingest with prune, assert gone |
| Cross-tenant or ACL leak | A user sees a document outside their tenant or groups | Evaluator leak count; returned doc ids audited against principal | Forbidden-doc gold rows; API tests per tenant |
| Duplicate chunks | Same text several times in top k | Distinct doc ids per result falls; chunk count grows on no-op ingestion | Second ingest re-embeds nothing |
| ANN recall regression | Retrieval recall drops after an index change, embeddings unchanged | Nightly overlap@k against exact | `ann-check` threshold in CI on index changes |
| Index degradation from churn | p95 latency rises at fixed `ef_search`; ANN recall drifts down | Index size per live row; dead tuples | Scheduled rebuild with before/after `ann-check` |
| Empty active namespace | Every query returns nothing, no errors | `/healthz` 503; chunks per namespace | Health check on an empty store |

## Tradeoffs

### PostgreSQL with pgvector versus a dedicated vector database

The choice turns on operational surface, consistency with your other data, filter behavior, and the scale at which one machine stops being enough, rarely on benchmark speed. Dedicated systems include, for example, Qdrant, Weaviate, Milvus, and Pinecone; OpenSearch, Elasticsearch, and Redis also offer vector indexes. Capabilities change quickly, so treat the right-hand column as questions to verify.

| Dimension | PostgreSQL + pgvector | Dedicated vector database |
|---|---|---|
| Operational surface | One database you probably already run and back up | A new stateful system: deployment, upgrades, backups, access control, on-call |
| Consistency with source data | Vectors, ACLs, and documents in one transaction; read-after-write on the primary | Kept in sync by a pipeline; usually eventual consistency |
| Filtering | Full SQL, row-level security; planner picks index or scan; iterative scans | Purpose-built filtered ANN, often stronger under highly selective filters; engine-specific filter language |
| Hybrid retrieval | `tsvector` full-text and reciprocal rank fusion (RRF, Chapter 12) in SQL | Built-in sparse or BM25 in some engines, external in others |
| Scale ceiling | One node's memory for the index; replicas for reads; sharding needs extra tooling | Horizontal sharding and replication built in; designed for billions of vectors |
| Index options | HNSW, IVFFlat; half-precision and binary types | Often more: PQ, disk-resident, GPU, tiered storage |
| Multi-tenancy | Columns, partitions, schemas, or databases | Collections, namespaces, or partitions; per-tenant limits vary |
| Cost and skills | Existing capacity, shared with transactional load; SQL skills | Separate infrastructure or usage pricing; engine-specific skills |

### Decision matrix

Pick the first row whose conditions hold.

| Situation | Choose | Why |
|---|---|---|
| Under about a million vectors, modest QPS | NumPy or an in-process exact index | Perfect recall, trivial filters, nothing to operate |
| You already run PostgreSQL; up to tens of millions of vectors; filters, joins, and transactional consistency with ACLs matter | PostgreSQL + pgvector | One system, SQL filters, consistency, hybrid with `tsvector` |
| Hundreds of millions of vectors, index memory beyond one large node, or very high QPS with tight p99 | Dedicated vector database | Sharding, quantization, and filtered ANN built in |
| Tenants with strict physical isolation requirements | Per-tenant databases or stores | No query filter provides physical separation |
| Mostly exact lookups by id, code, or field | No vector index; SQL, full-text, or key-value | More accurate and cheaper |
| Lexical and dense search with heavy text analytics | A search engine with vector support | One engine for BM25, aggregations, and vectors |

Whatever you choose, keep the store behind a protocol like `VectorStore`, the documents as the source of truth, and an exact path for measurement; migrating engines is then a re-ingestion plus a contract test run.

### When a vector database is unnecessary

The retrieval in RAG does not have to be vectors. Four common situations need no vector index:

- **The corpus is small.** A few hundred documents fit in a NumPy matrix or often in the context window (Chapter 10 compares RAG with long context).
- **The queries are exact lookups.** "Status of ticket TCK-1042" or "error code RET-002" are key or keyword queries; embedding models are weak at rare identifiers, and SQL or full-text search answers them exactly. The Northwind gold set tags such questions `exact-id` so you can measure dense retrieval on them separately.
- **The data is structured.** Metrics, orders, and inventory belong in SQL, possibly model-generated over a semantic layer (the `query_metrics` tool; Chapter 36). Similarity search over rows loses exact filtering, aggregation, and joins.
- **The content is navigable.** Manuals, legal codes, and API references can be retrieved through their hierarchy, metadata filters, or a model choosing sections from an outline; Chapter 37 implements this.

The sensible default for a new project: full-text search and an exact vector scan in the database you already have, measured on real questions, with an ANN index or a dedicated store added when measurements show the need.

## Evaluation and testing

**Contract tests** run the same behavior against every adapter. The most important is a selective filter whose only match is far from the query, which catches post-filtering. Offline tests also check the adapter's SQL: parameterized `WHERE` clauses, the expected indexes, and a checked-in `sql/schema.sql` that matches the renderer.

**Component tests** cover the loader (fail-closed ACL, content-hash versioning), ingestion (zero embedding calls on an unchanged corpus, pruning), the API (401 without identity, filters from headers only), the metrics, and the ANN experiments. The metrics test pins a worked example: two relevant documents, one found at rank 2, gives recall@5 of 0.5, precision@5 of 0.2, and reciprocal rank 0.5.

**The retrieval evaluation** runs the shared Northwind gold set (`shared-data/eval/retrieval_gold.jsonl`, 40 questions) through the real service. With the offline hashing embedder, which behaves like a bag-of-words model, the run produced (illustrative, abridged):

```text
scored rows: 37  forbidden rows: 3
recall@1: 0.622   recall@3: 0.919   recall@10: 1.000
MRR: 0.776
ACL leaks: 0
ingestion gaps: []
```

Read these skeptically. Each user sees at most about 20 of the 24 documents, so recall@10 of 1.0 means little; recall@1 and MRR are informative at this size. On a real corpus the per-tag breakdown (`exact-id`, `paraphrase`, `multi-hop`) shows where dense retrieval needs Chapter 12's hybrid retrieval. The leak count is meaningful at any size: the three forbidden rows ask about documents the user may not read. (One, RQ-037, is also answerable from a public FAQ; Chapter 14 treats it as a gold-label defect, but its leak check is still valid.) Chapter 14 builds the full retrieval evaluation methodology on this harness.

For ANN, the test is the overlap check from Measuring ANN recall: require overlap@10 above a threshold chosen from the cost curve (for example 0.95) before activating a new index version.

## Before you ship

- [ ] The namespace encodes the embedding model (ideally the full space fingerprint) and the index version, and the service refuses to start when the query embedder does not match the active namespace.
- [ ] Every record has a non-empty `acl_groups` and a tenant; ingestion rejects documents without them, and the database enforces it with a `CHECK` constraint.
- [ ] Tenant and group filters are built from verified identity in one function, and an API test proves a request body cannot widen them.
- [ ] A contract test with a selective filter whose only match is far from the query passes against the production adapter, not only the NumPy one.
- [ ] On pgvector, iterative index scans are enabled (0.8 or later) or `ef_search` is at least k divided by the smallest filter selectivity you expect; `ef_search`, iterative scan, and `statement_timeout` are set per transaction.
- [ ] `ann-check` overlap@10 against exact search meets a written threshold (for example 0.95) on real queries, and runs nightly and on every index change.
- [ ] The retrieval gold set, including forbidden-doc leak rows, runs in CI and blocks a deploy on any leak or on a recall drop beyond the agreed margin.
- [ ] A second ingestion run on an unchanged corpus embeds zero texts and writes zero chunks.
- [ ] Ingestion runs with pruning, and a test shows a deleted source document stops being retrievable; time-to-unretrievable for compliance deletions is measured and alerted on.
- [ ] Dashboards show underfilled-result rate by tenant, store p95 latency, chunks per active namespace, and ingestion lag, with alerts.
- [ ] `/healthz` fails when the active namespace is empty, and the full rebuild-from-source time is measured and recorded as the worst-case recovery time.
- [ ] A rebuild or migration runbook exists: build the new namespace, dual-write, verify, switch the alias, keep the old namespace for the rollback window.

## Exercises

**Start here:** K3, K4, E2, P3, D1 (about 4 hours). The rest go deeper.

### Knowledge questions

**K1.** Explain why vectors produced by two different embedding models cannot share an index, even when the two models output vectors of the same dimension. What does Project 2 do to make mixing impossible?

**K2.** Describe what `M`, `ef_construction`, and `ef_search` control in an HNSW index. Which of them can be changed without rebuilding the index, and what does raising each one cost?

**K3.** Distinguish ANN recall from retrieval recall. For each, name what it is measured against and one change that would improve it.

**K4.** A filter passes 3% of the rows. The index returns 60 candidates before filtering, and the query asks for k = 10. Estimate how many results post-filtering returns on average, and describe two ways to get all ten.

**K5.** Why does product quantization usually come with a rescoring step? What is kept where, and what does rescoring cost?

**K6.** Name three situations where a retrieval-augmented system should not use a vector index at all, and the retrieval mechanism you would use instead.

### Engineering questions

**E1.** Northwind is acquiring a third business unit, `wholesale`, whose contract requires that its documents be stored separately and deletable on request within 24 hours, with an audit trail. Design the tenancy model for the vector store across the three units and the shared corpus. Address query routing, the shared documents, deletion, and how you would test isolation.

**E2.** You must migrate the knowledge index from the current embedding model to a new one with different dimensions, without downtime, while documents keep changing during a backfill that takes six hours. Write the migration plan: states, dual-write, verification gates, switch-over, rollback, and what you monitor.

**E3.** A product manager asks for "a vector database" for a new feature that searches 300,000 support tickets by meaning, filtered by customer account (12,000 accounts, very uneven sizes) and date range, at about 20 queries per second. Recommend a storage design using the decision matrix, justify it with numbers, and state what measurement would make you change your mind.

**E4.** The pgvector adapter sets `hnsw.ef_search` with `set_config(..., true)` inside a transaction instead of running `SET hnsw.ef_search` once at connection start. Explain the failure this prevents in a service using a connection pool, and describe a scenario in which the global setting would cause a correctness or performance incident.

### Practical exercises

**P1.** (about 90 min) Add a `SearchFilter.updated_after: date | None` field that filters on the `updated_at` value from the front matter. Implement it in `matches`, in `build_where` (using the JSONB metadata or a new column, your choice, justified), and add contract tests, including one where the filter is highly selective.

**P2.** (about 3 hours) Implement a minimal single-layer navigable graph index in `semsearch/domain/` (each vector linked to its M nearest neighbors, greedy best-first search with a candidate list of size `ef`). Extend `bench.py` with a sweep over `ef` that reports recall@10 against exact search and vectors scored per query, and compare the curve with the IVF sweep at similar work.

**P3.** (about 2 hours) Add a selectivity-aware search path to `NumpyVectorStore`: when an approximate index (your P2 graph or the IVF index) is attached, use it for unfiltered or weakly filtered queries, and fall back to exact pre-filtered search when the filter allows fewer than a configurable number of rows. Prove with tests that results never underfill when matches exist.

**P4.** (about 3 hours) Add a `semsearch reindex --to-version v2` command that builds a new namespace from the source documents, runs the evaluator and `ann-check` against it, refuses to proceed if any leak occurs or recall@5 drops by more than a configured margin against the active namespace, and otherwise writes the new active version to a small state file that `make_store` and the API read.

### Debugging exercises

**D1.** After a rollout, retail users report that Northwind Assist "does not know anything about Lumen POS anymore", while logistics users see no change. The search spans for retail queries show `returned` values of 0 to 3 with k = 8 and `underfilled = true` on most requests; logistics spans look normal. The rollout moved the knowledge index from the NumPy store to pgvector 0.7 with `hnsw.ef_search` left at its default of 40. The retail tenant holds about 2% of all chunks. Diagnose the cause and propose two fixes.

**D2.** Retrieval quality collapses on Monday morning. Top scores in search spans, previously spread between 0.3 and 0.8, now sit between 0.02 and 0.11 for every query. Ingestion logs show nothing unusual and chunk counts are unchanged. The deploy log shows that on Friday the embedding service's configuration was updated to "the latest model" for a different team's feature. What happened, which span field confirms it, and what structural change prevents it?

**D3.** A user in the `all` group asks about database failover and receives an answer quoting the PostgreSQL failover runbook, which is restricted to `it-oncall`. The trace shows `tenant = shared`, `groups = ["all"]`, and the runbook's chunk id in `doc_ids`. Last week the runbook's front matter was corrected: its `acl_groups` had been `["all"]` by mistake and is now `["it-oncall"]`; nobody changed its `version` field. Production is indexed by an older ingestion job that skips a document when its declared version matches the indexed one. The evaluator, which reports zero leaks, runs in CI against an index built from scratch from the repository. Find the root cause and explain why the evaluator did not catch it.

## Key takeaways

- A vector index is a derived, rebuildable cache. Every record must carry tenant, ACL, document identity and version, and the embedding model, or later stages cannot filter, cite, invalidate, or rebuild it.
- Exact search is perfectly accurate, trivially filterable, and fast enough up to roughly a million vectors at modest query rates. It is also the ground truth for measuring any ANN index.
- HNSW, IVF, and product quantization trade recall, latency, memory, and build cost. Tune them against a recall target measured as overlap with exact search on your own queries.
- Keep ANN recall and retrieval recall separate. Index tuning cannot fix a problem that lives in the embeddings, the chunking, or the query.
- Post-filtering an ANN candidate list starves selective filters silently. Use pre-filtering, filter-aware or iterative index scans, routing by selectivity, or partitioning on the dominant filter, and alert on underfilled results.
- Tenancy and authorization are two mandatory filters derived from verified identity in one code path; the request body may narrow a search but never widen it.
- The embedding model is part of the index identity. Model or chunker changes are migrations: build a new namespace, dual-write, verify recall and leaks, switch, and keep a rollback window.
- PostgreSQL with pgvector is the strong default when you already run PostgreSQL and need SQL filters, transactional consistency, and hybrid search; dedicated vector databases earn their operational cost at very large scale or with demanding filtered-ANN workloads.
- Many retrieval problems need no vector index at all: small corpora, exact lookups, structured data, and navigable documents are better served by SQL, full-text search, or structure.

## Further reading

- *Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs* (Malkov and Yashunin, 2018): the HNSW paper; read it for why the layered graph works and what `M` and `ef` actually control.
- *Product Quantization for Nearest Neighbor Search* (Jégou, Douze, and Schmid, 2011): the original PQ paper, including the asymmetric distance tables behind compressed scoring.
- *Billion-scale Similarity Search with GPUs* (Johnson, Douze, and Jégou, 2017): the FAISS paper; IVF and PQ at a scale where memory is the binding constraint.
- pgvector documentation (github.com/pgvector/pgvector): index options, operator classes, `ef_search`, and iterative index scans; check it for the behavior of the version you run.
- *Designing Data-Intensive Applications* (Kleppmann, 2017): replication lag, derived data, and read-your-writes, the database ideas behind treating an index as a rebuildable cache.
