# Chapter 9 — Vector Search and Vector Databases

Chapter 8 covered what embeddings are; this chapter is about storing and searching them: fast enough at your scale, filtered by who is asking, and safe to rebuild when the model or the documents change. You build Project 2, a semantic search system with a `VectorStore` protocol, an exact NumPy adapter, a PostgreSQL + pgvector adapter, an incremental ingestion CLI, a FastAPI `/search` endpoint whose authorization filters come from identity, and an evaluator.

**You will be able to:**
- Decide whether a workload needs a vector database at all, using corpus size, query rate, and latency budget.
- Choose between exact search, HNSW, IVF, and quantized indexes from measured recall, latency, and memory, and tune them against overlap with exact search.
- Design metadata filtering that stays correct under selective filters, and detect filtered-ANN starvation from telemetry.
- Lay out namespaces and tenants so that one customer's vectors can never answer another customer's query.
- Operate an index through re-embedding, blue-green rebuilds, deletions, and replication.

**Prerequisites:** Chapters 3 (the `aie_core` embedding clients, `CachedEmbeddings`, settings, and tracing) and 8 (similarity metrics, normalization, and the embedding space fingerprint). | **Code:** `book/projects/p2-semantic-search/` (run: `cd book/projects/p2-semantic-search && pytest -q`) | **Builds:** Project 2, the `semsearch` package.

## Why this matters

Once text is embedded, retrieval becomes a geometry problem: find the stored vectors closest to the query vector. The arithmetic is trivial. A dot product over a few hundred floats takes nanoseconds. What makes vector search an engineering topic is everything around that arithmetic.

The first problem is scale. Comparing a query with every stored vector is linear in the corpus size, and the constant is not small: a million 768-dimensional float32 vectors occupy about 3 GB, and on a laptop a single brute-force query over them took about 60 ms in our measurements (illustrative). That is fine for an internal tool and unacceptable for a service doing hundreds of queries per second. Approximate nearest neighbor (ANN) indexes fix the speed by giving up exactness, and the price of that bargain is a recall loss that nobody sees unless they measure it.

The second problem is that a vector is never the whole query. Northwind Assist must answer a logistics employee using logistics documents and shared policies, never retail incident reports, and never the on-call runbooks unless the user is in the `it-oncall` group. Those constraints are metadata filters, and they interact badly with ANN indexes: a filter that matches 1% of the corpus can leave an approximate index returning one result or none instead of ten. The bug looks like "the assistant does not know about our product", not like an error.

The third problem is lifecycle. Documents change, get deleted, get reclassified. Embedding models get upgraded, and vectors from two models are not comparable, so a model change means re-embedding everything and swapping indexes without downtime. Deleted documents must stop being retrievable promptly, because "the vector store still returns the revoked policy" is both a correctness bug and a compliance incident.

The industry answer to these problems was a wave of dedicated vector databases. Many teams then discovered that PostgreSQL with the pgvector extension, or even a NumPy matrix in memory, covered their workload with less operational surface. Others outgrew those and needed the dedicated systems. This chapter gives you the criteria to tell which situation you are in, and the code to start on the simplest option without painting yourself into a corner.

## Mental model

> **Mental model:** A vector index is a derived, versioned cache of a computation over your corpus. The documents are the source of truth; the index is rebuildable, and every record in it must carry enough identity (document, version, tenant, ACL, embedding model) to be filtered, cited, invalidated, and rebuilt.

Three corollaries guide the rest of the chapter.

**Recall is a property of the whole stack, and each layer can lose it.** End-to-end retrieval recall can drop because the embedding model does not represent a term, because chunking split the answer, because a filter excluded the right document, or because the ANN index skipped it. Each of these needs its own measurement. An ANN index that misses the answer looks fast and costs recall you cannot see without an exact baseline.

**Filters are part of the query, not decoration on top of it.** A search for "top 10 neighbors that this user may see" is a different query from "top 10 neighbors, then remove the ones this user may not see". The first is correct by construction. The second silently returns fewer results, and the shortfall grows as filters get more selective, which is exactly the case for small tenants and restricted groups.

**Start exact, measure, and only then approximate.** Exact search is the ground truth for every ANN index you will ever tune. Keeping an exact path available, even if only for offline measurement, is what lets you know what your index is costing you.

## Core concepts

### What a vector store actually stores

The unit of storage is not a vector. It is a record with a vector in it. Project 2's `VectorRecord` has these fields, and each one exists because some later stage fails without it:

| Field | Why it exists | What breaks without it |
|---|---|---|
| `id` | Deterministic `doc_id@version#ordinal` | Re-ingestion duplicates chunks instead of replacing them |
| `namespace` | Index name, embedding model, index version | Vectors from two models get compared; scores become noise |
| `tenant` | Isolation boundary (`retail`, `logistics`, `shared`) | Cross-tenant leakage through retrieval |
| `acl_groups` | Who may read the chunk | Permission leaks; the model quotes a restricted runbook |
| `doc_id`, `doc_version`, `ordinal` | Citation and invalidation | Citations cannot be resolved; stale chunks cannot be found |
| `text` | What the generator and reranker read | A second lookup per hit, or no evidence to show |
| `tags`, `metadata` | Narrowing filters, titles, sections, dates | Users cannot scope searches; answers cannot show sources |
| `embedding_model` | Audit and migration | Nobody can tell which rows still need re-embedding |

Two rules follow. First, **the embedding model version is part of the index schema.** Two embedding models produce vectors in unrelated coordinate systems, even when the dimensions match. A cosine similarity between a vector from model A and one from model B is a number with no meaning. Project 2 encodes the model in the namespace (`knowledge:fake-embedding:v1`), so mixing is structurally impossible: a query embedded with a new model is sent to a namespace that only holds vectors from that model. The model name is the minimum. Chapter 8 defines the full embedding space fingerprint (dimensions, prefixes, normalization, text preparation), and a production namespace should carry that fingerprint so that a prefix or dimension change also lands in a new namespace.

Second, **a record without an ACL must not mean "public".** The `VectorRecord` validator rejects an empty `acl_groups`, and the document loader refuses a file without `acl_groups` in its front matter. Public documents say so explicitly with `["all"]`. Failing closed at ingestion is cheaper than discovering at query time that a missing field was interpreted as "no restriction".

### Distance, normalization, and what a score means

Vector stores offer three distance functions: Euclidean (L2) distance, inner (dot) product, and cosine distance. Chapter 8 covers the math. The operational points are short. Use the metric the embedding model was trained for, which is almost always cosine or dot product. If you L2-normalize every vector at write time, cosine similarity and dot product produce the same ranking, and dot product is cheaper, which is why `NumpyVectorStore` normalizes on write and scores with a single matrix-vector product. Configure the index with the same metric you query with; in pgvector that means the operator class (`vector_cosine_ops`) must match the operator in `ORDER BY` (`<=>`), or the planner will not use the index at all.

A similarity score is a ranking signal, not a calibrated relevance probability. A cosine of 0.82 does not mean "82% relevant", and the same score means different things for different models, different query lengths, and different corpora. Thresholds on raw similarity ("drop hits below 0.75") are brittle: they need recalibration whenever the model changes and they behave differently for short keyword queries than for long questions. If you need a relevance cut-off, put it after a reranker (Chapter 12) and calibrate it on judged queries (Chapter 14).

### Exact search, and when it is enough

Exact (brute-force, flat) search computes the similarity between the query and every candidate vector and keeps the top k. Its properties are attractive: perfect recall by definition, no index to build or tune, instant inserts and deletes, and filters that are trivially correct because you can restrict the candidate set before scoring.

Its cost is linear in the number of vectors scored. Measured on a laptop with NumPy (illustrative; your hardware will differ):

| Vectors (768 dims, float32) | Matrix size | Exact top-10, one query |
|---|---|---|
| 100,000 | 0.3 GB | about 4.5 ms |
| 1,000,000 | 3.1 GB | about 63 ms |

That table carries the main conclusion: **for corpora up to roughly a million vectors with modest query rates, exact search on one machine is a legitimate production design.** Northwind's knowledge base, a few thousand documents chunked into tens of thousands of chunks, is two orders of magnitude below the point where exact search becomes slow. Most internal assistants are in the same position.

Exact search stops being enough in three situations. The corpus no longer fits in memory on one machine. The query rate times the per-query cost exceeds your CPU budget: 63 ms per query is about 16 queries per second per scanning process, and because the scan is bound by memory bandwidth, extra cores help less than batching does. Or the latency budget for retrieval is tighter than the scan allows, which happens when retrieval is one of several stages inside a 2-second time-to-first-token budget (Chapter 30). Batching queries into a matrix-matrix product and using a GPU push all three limits out considerably, which is worth remembering before adopting an ANN index for a workload that is merely bursty.

### Approximate nearest neighbor search: the trade-off space

An ANN index avoids scoring most vectors. It organizes them so that a query can be routed toward its neighborhood and only score candidates there. Every ANN method trades among four quantities:

- **Recall**: the fraction of the true top-k that the index returns. This is ANN recall, measured against exact search, not retrieval recall against human judgments.
- **Query latency**: how many vectors are scored and how much memory is touched per query.
- **Memory**: the vectors themselves plus the index structure (graph edges, centroids, codebooks).
- **Build and update cost**: time to build, cost of inserts and deletes, need for retraining.

You cannot have all four. The three families below sit at different points, and production systems often combine them.

### HNSW: a navigable graph

Hierarchical Navigable Small World (HNSW) graphs are the most widely used ANN index; pgvector offers them alongside IVFFlat. "Small world" refers to graphs in which most nodes are reachable from any other in a few hops. The structure is a stack of proximity graphs. The bottom layer contains every vector, each connected to up to `2 * M` near neighbors. Each higher layer contains a random, exponentially smaller subset of the layer below, so the top layers are sparse "highways" across the space and the bottom layer is a dense local street map.

Search starts at an entry point in the top layer, walks greedily to the neighbor closest to the query, and descends a layer when no neighbor is closer. At the bottom layer it switches from greedy walking to a best-first search that keeps a candidate list of size `ef_search` and expands the closest unexplored candidates until no improvement is possible. The top k of that list is the answer.

```text
# pseudocode: HNSW query
entry = top_layer_entry_point
for layer in top_layer .. 1:
    entry = greedy_closest(query, entry, layer)            # walk until no neighbor is closer
candidates = best_first(query, entry, layer=0, size=ef_search)
return closest k of candidates
```

The parameters map onto the trade-off space directly:

| Parameter | When set | Raises | Costs |
|---|---|---|---|
| `M` (max neighbors per node) | build | recall, robustness on hard data | memory (edges per vector), build time |
| `ef_construction` | build | graph quality, hence recall at a given `ef_search` | build time only |
| `ef_search` | per query | recall | latency, roughly linearly |

In pgvector the defaults are `m = 16`, `ef_construction = 64`, and `hnsw.ef_search = 40`. Treat these as a starting point, not a recommendation for your data. A common tuning path is to fix `M` and `ef_construction` at moderate values, build once, and then sweep `ef_search` at query time while measuring ANN recall against exact search on a sample of real queries. Raise `ef_construction` only if no reasonable `ef_search` reaches your recall target. In pgvector without iterative scans, `ef_search` caps the result count: asking for 50 results with `ef_search = 40` cannot return 50.

HNSW's strengths are high recall at low latency without a training step, and good behavior under incremental inserts. Its weaknesses are memory and deletes. Every vector carries its full-precision copy plus its edge lists, so the index usually lives in RAM at a size somewhat above the raw vectors. Deletes are handled by marking nodes as deleted; the search still traverses them but excludes them from results until a vacuum or rebuild repairs the graph, and heavy delete churn degrades both recall and latency over time.

Builds are also expensive: inserting each vector runs a search to find its neighbors, so building over millions of vectors takes minutes to hours and benefits from a large memory budget for the build process.

### IVF: partition, then probe

Think of bucketing a phone book by city and searching only the few nearest cities. Inverted file (IVF) indexes partition the vector space into `n_lists` clusters with k-means, an algorithm that picks `n_lists` center points so that each vector is close to one of them. Each vector is stored in the list of its nearest centroid. At query time the index compares the query with all centroids, picks the `nprobe` closest lists, and scans only those exhaustively. With 128 lists and `nprobe = 8`, a query scores the centroids plus roughly 8/128 of the vectors.

Project 2 includes a teaching-size IVF index (`semsearch/domain/ivf.py`) and a sweep that measures its recall against exact search on synthetic clustered vectors (20,000 vectors, 64 dimensions, 128 lists, 200 queries). The output, from `semsearch bench-ann` on a laptop (illustrative):

| nprobe | ANN recall@10 | Vectors scanned | ms/query |
|---|---|---|---|
| 1 | 0.541 | 1.0% | 0.033 |
| 2 | 0.758 | 1.7% | 0.038 |
| 4 | 0.908 | 3.3% | 0.048 |
| 8 | 0.983 | 6.5% | 0.065 |
| 16 | 0.999 | 12.9% | 0.101 |
| 32 | 1.000 | 25.5% | 0.181 |

Exact search on the same data took 0.283 ms per query. Read the curve: recall climbs steeply and then flattens, while cost grows linearly with `nprobe`. The knee, here around `nprobe = 8`, delivers 98% recall for about 6.5% of the work. That shape is typical, and it is why "tune against a recall target" is the right framing: choose the cheapest setting that meets the target on your data.

IVF needs training data. The centroids are learned from a sample, and if the data distribution drifts (a new product line, a new language, a new document type), new vectors pile into a few lists and both recall and latency degrade. The fix is periodic retraining, which means a rebuild. pgvector's IVFFlat index has the same property and should be created after the table has representative data, not on an empty table. IVF's advantages are fast builds, low memory overhead (just centroids and list assignments), and natural fit with compression and with disk-resident storage, because each probed list is a contiguous scan.

### Product quantization: compress the vectors

Both HNSW and IVF still store full-precision vectors, and at large scale memory, not compute, is the binding constraint. Product quantization (PQ) works like a color palette: instead of storing each slice of a vector exactly, it stores which of a small set of prototype slices it is closest to. Concretely, PQ compresses vectors by splitting each one into `m` sub-vectors and replacing each sub-vector with the index of its nearest centroid in a small learned codebook (typically 256 entries, so one byte). A 768-dimensional float32 vector is 3,072 bytes; with 96 sub-vectors of 8 dimensions each it becomes 96 bytes, a 32x reduction (illustrative arithmetic). Distances are approximated from precomputed tables of query-to-centroid distances, so scoring a compressed vector costs a handful of table lookups.

The price is precision: PQ distances are approximate, so the ranking among close candidates gets noisy. The standard remedy is **rescoring**: use the compressed index to fetch a larger candidate set (say 100), then re-rank those candidates with full-precision vectors kept on disk or in a cheaper tier, and return the top 10. Simpler relatives are scalar quantization (float32 to int8 or float16, 2x to 4x smaller, small recall loss) and binary quantization (one bit per dimension, 32x smaller, usable only with rescoring and only for some embedding models). pgvector offers half-precision (`halfvec`) and binary (`bit`) types for the same purpose.

Typical combinations are IVF with PQ codes (IVF-PQ) for very large, memory-constrained corpora, and HNSW over quantized vectors with full-precision rescoring. A third family, disk-resident graph indexes (DiskANN is the best-known design), keeps only compressed vectors in RAM and stores the graph and full-precision vectors on SSD, trading a few disk reads per query for a much smaller memory footprint. You do not need any of this for the Northwind corpus. You need to know it exists so that when a sizing exercise says "400 GB of vectors", the answer is "quantize and rescore", not "buy 400 GB of RAM".

### Measuring ANN recall against exact search

Every ANN number in this chapter is defined relative to exact search on the same vectors with the same filters. That definition gives you a test you can automate:

1. Take a sample of real queries (or the questions in your retrieval gold set).
2. For each, run the production search and an exact search with identical filters and k.
3. Compute overlap@k: the fraction of the exact top-k that the ANN search also returned.

Project 2 implements this as `semsearch ann-check`. The `VectorStore.search` method takes `exact=True`, which `PgVectorStore` implements by disabling index scans inside the query's transaction, forcing the planner to scan and sort every matching row. The NumPy store is always exact, so its overlap is 1.0 by construction, which makes it the reference when you migrate.

Keep two recall numbers apart. **ANN recall** asks whether the index found what exact search would have found. **Retrieval recall** asks whether the search found what a human said was relevant. If retrieval recall is 0.70 and ANN recall is 0.99, tuning the index cannot help; the problem lives in the embedding model, the chunking, or the query (Chapters 11 and 12). If retrieval recall drops from 0.85 with exact search to 0.70 with the index, the index is costing you 15 points and `ef_search` or `nprobe` is the knob. Measuring both is the only way to know which situation you are in.

### Metadata filtering: pre-filter, post-filter, and in-index

Every Northwind query carries at least two filters: tenant (the user's tenant plus `shared`) and ACL (the user's groups must intersect the chunk's groups). Users may add more, such as tags. There are three ways to combine a filter with nearest-neighbor search.

**Pre-filtering** restricts the candidate set to rows that pass the filter, then searches among them. With exact search this is simple and always correct: build a mask, score only the allowed rows. `NumpyVectorStore` does this. With an ANN index it is harder, because the index structure was built over all vectors; restricting to a subset means either scanning the subset exhaustively (fine when the subset is small) or traversing the graph while skipping disallowed nodes (which can disconnect the graph for very selective filters).

**Post-filtering** asks the index for the top N candidates, then removes the ones that fail the filter. It is easy to implement and it is what you get by default from many engines and from a naive SQL query against an HNSW index: pgvector's HNSW scan produces `ef_search` candidates, and the `WHERE` clause is applied to those. The failure is arithmetic. Call the fraction of rows a filter passes its selectivity s; a smaller s means a more selective filter. If the index returns N candidates, you expect about `N * s` survivors, capped at k (small deviations in the table are sampling noise). `semsearch bench-filter` measures it with N = 40 (playing the role of `ef_search`) and k = 10 (illustrative):

| Filter selectivity | Post-filter results (of 10) | Pre-filter results (of 10) |
|---|---|---|
| 50% | 10.00 | 10.00 |
| 10% | 4.21 | 10.00 |
| 2% | 0.82 | 10.00 |
| 0.5% | 0.17 | 10.00 |

At 2% selectivity, typical for "logistics tenant and `it-oncall` group" in a large multi-tenant corpus, post-filtering returns less than one result on average where ten exist. Nothing errors. The assistant just abstains or answers from the wrong evidence. This is a common silent failure in production retrieval.

**In-index (filter-aware) search** makes the index itself respect the filter. Approaches include traversing the HNSW graph while only counting allowed nodes toward the result, continuing the scan until k allowed results are found, maintaining per-value sub-indexes, or switching automatically to exact scanning when the filter is selective enough that the allowed set is small. pgvector 0.8 and later implement the "keep scanning" approach as iterative index scans (`hnsw.iterative_scan`, with a `relaxed_order` mode that may return results slightly out of order, so the adapter re-sorts them). Dedicated vector databases implement various combinations and differ substantially in how well they handle highly selective filters (small s); test this specifically when you evaluate one.

Practical rules for filtering:

- **Know your selectivity distribution.** Log, per query, how many rows the filter allows (or estimate it from tenant sizes and group membership). The smallest tenants and the most restricted groups are where filtered ANN breaks first.
- **Route by selectivity.** When the filter allows a few thousand rows, an exact scan of those rows using a B-tree or GIN index (PostgreSQL's inverted index for arrays and JSONB) on the filter columns is both faster and correct. PostgreSQL's planner makes this choice itself when statistics are good; dedicated systems often have a threshold setting.
- **Partition on the dominant filter.** If every query filters by tenant, give large tenants their own partition or index (see the next section) so that the tenant filter disappears from the ANN problem.
- **Alert on underfilled results.** Project 2's search span records `underfilled` whenever fewer than k results come back. A rising underfill rate for one tenant is the signature of a filtering problem.

### Namespaces, collections, and tenants

Vector stores provide some way to group vectors: collections, indexes, namespaces, partitions, or simply tables. Three isolation models cover most designs.

| Model | How | Isolation | Cost and limits | Fits |
|---|---|---|---|---|
| Shared index, tenant column | One index; every row has `tenant`; every query filters on it | Logical only; a missing filter leaks | Cheapest; filtered-ANN problems for small tenants | Many small tenants, internal tools |
| Namespace or partition per tenant | Separate index per tenant inside one store | Strong at query time; no filter needed for tenancy | Per-namespace overhead; many tiny indexes are wasteful | Tens to thousands of mid-size tenants |
| Store or database per tenant | Separate cluster, database, or schema | Physical; separate keys, backups, deletion | Highest operational cost | Regulated tenants, contractual isolation, very large tenants |

Hybrids are common: a shared index for the long tail of small tenants, dedicated namespaces for the largest ones, and dedicated stores for tenants whose contracts demand it. Whatever you choose, enforce tenancy in one code path that every query goes through, derive the tenant from the authenticated identity rather than from the request body, and test it with a tenant-leak test that runs in CI.

Project 2 uses namespaces for a different axis: **index identity**. The namespace `knowledge:<embedding model>:<index version>` separates indexes built with different embedding models or chunking configurations, which is what makes blue-green re-indexing possible: build the new index beside the live one, then switch traffic to it (see Indexing strategies and rebuilds). Tenancy is a filtered column (`tenant`), plus ACL groups, because Northwind has two tenants and a shared corpus, and the shared documents must be visible to both. The schema indexes `(namespace, tenant)` so that PostgreSQL can serve small-tenant queries with a filtered scan instead of the ANN index.

Northwind's `shared` tenant illustrates a subtlety. A logistics user sees `tenant IN ('logistics', 'shared')`. Shared documents appear in every tenant's results, so their ACL groups do all the work of restricting them; the database failover runbook is `tenant: shared` but readable only by `it-oncall`. A tenant filter alone would leak it to every employee. Tenancy and authorization are two separate filters, and both are mandatory.

This chapter covers the mechanics that make those filters correct inside a vector search. Chapter 15 owns authorization in retrieval as a whole: permission filters are applied before scoring, never after generation, and ACL changes, caches, and audit must keep up with them.

## How it works

With the pieces defined, here is how a query and a write move through Project 2.

### The life of a search

A Northwind support agent in the retail business unit types "how long can a store keep selling while offline". The request reaches Project 2's `/search` endpoint through the authenticating gateway, which has set `X-User`, `X-Tenant: retail`, and `X-Groups: all,retail`. From there:

1. **Identity becomes a principal.** `get_principal` builds a `Principal` from the headers and rejects the request with 401 if any of them is missing or the group list is empty. There is no default identity.
2. **The principal becomes a filter.** `SearchService.authorization_filter` produces `tenants = ("retail", "shared")` and `acl_groups = ("all", "retail")`. The request body may add `tags` to narrow the search; nothing in the body can widen it.
3. **The query is embedded** with the same embedding client, and therefore the same model, that built the namespace.
4. **The store searches** the namespace with the vector, k, and the filter. The NumPy store builds a boolean mask from the filter and scores only allowed rows. The pgvector store sends one parameterized SQL statement with the filter in the `WHERE` clause, with `hnsw.ef_search` and, where supported, iterative scanning set for this transaction only.
5. **The span is recorded**: namespace, k, tenant, groups, embedding latency, store latency, number of results, whether the result was underfilled, the top score, and the returned document ids. These are the fields you need when someone reports "search did not find X".

### The life of a write

Writes are where most vector-store bugs live, because the index is a derived cache and caches go stale. Project 2 writes per document, not per chunk:

1. The loader reads the document, parses the front matter, and computes a content hash. The document's **index version** is the declared version plus the first eight hex digits of the hash, so an edit that forgets to bump the version still produces a new index version.
2. The pipeline asks the store what it currently holds for each document (`doc_versions`). If the index version matches, the document is skipped and costs nothing. This is what makes re-running ingestion every few minutes cheap.
3. For a new or changed document, the pipeline chunks it, embeds the chunks in batches, and calls `replace_document(namespace, doc_id, index_version, records)`.
4. `replace_document` makes the new chunks the only chunks of that document, atomically. In PostgreSQL it runs in one transaction that takes an advisory lock on the document (an application-level lock keyed on the document id, so two ingestions of the same document serialize), upserts the new rows, and deletes the document's rows whose ids are not in the new set. A concurrent reader sees either the old version or the new one. The naive alternative, "delete old chunks, then insert new ones" in two transactions, opens a window where the document does not exist; the inverse order opens a window where both versions are retrievable and a stale fact can be cited next to the current one.
5. With `--prune`, documents that exist in the store but no longer exist in the source are deleted. Without pruning, a revoked policy stays searchable forever. The test `test_without_prune_deleted_sources_stay_searchable` pins this behavior down so that nobody mistakes it for a feature.

### Consistency: what a reader can observe

Most vector stores are eventually consistent in at least one place. Dedicated databases often acknowledge a write before it is visible to search, because index updates are applied asynchronously or in segments that are merged in the background. Replicas lag the primary. Caches in front of the store lag both. PostgreSQL with pgvector gives you read-after-write consistency on the primary for free, because the HNSW index is updated inside the inserting transaction, an advantage of keeping vectors in your transactional database that is easy to overlook.

The consistency you need depends on the use case. A knowledge base that is re-indexed every 15 minutes tolerates seconds of lag. A user who uploads a file and immediately asks about it does not; that flow needs read-your-writes, which you can provide by routing that user's queries to the primary for a short window, by waiting on the write's visibility token where the store offers one, or by searching the freshly uploaded document directly before it reaches the shared index.

Deletion is the case that must not lag silently: if a document is deleted for legal or permission reasons, measure the time until it stops appearing in search results and alert when it exceeds the agreed bound.

## Architecture

The first diagram shows the two paths through Project 2 and where trust changes. Documents and queries are untrusted text; identity comes from the gateway; the store holds only what ingestion put there.

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

The second diagram shows the lifecycle of an index version. Re-embedding with a new model, changing the chunker, or changing index parameters all create a new namespace that is built and verified offline before any traffic sees it.

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

The lifecycle diagram shows when a new namespace is built; this section covers what forces one. Four situations force a rebuild, and they differ in cost.

**The embedding model changes.** Every vector must be recomputed, because old and new vectors are incomparable. This is the expensive one: the embedding cost of the whole corpus, plus build time. Plan it as a migration:

1. Build the new namespace in the background while the old one serves traffic. The backfill re-embeds the existing corpus into it.
2. Dual-write: from the moment the backfill starts, ingestion writes every changed document to both namespaces, so the new namespace does not fall behind during a backfill that may take hours.
3. Verify with the retrieval gold set and the leak gate against the new namespace.
4. Switch the active namespace with one configuration change (in Project 2, `INDEX_VERSION` or the embedding model setting; in a larger system, an alias table the service reads).
5. Keep the old namespace, and keep dual-writing to it, until the rollback window closes.

Two details decide whether this goes smoothly. First, a new model often has different dimensions, and in pgvector the column type is `vector(d)`, so the new namespace needs its own table (in Project 2, a different `PG_TABLE`) rather than new rows in the old one; plan the disk and memory for both indexes side by side during the overlap. Second, estimate the backfill before starting: total tokens times the embedding price, and total tokens divided by your rate limit for wall-clock time (Chapter 8's `plan_reembed` does this arithmetic). Because a backfill can take hours while documents keep changing, dual-write is a required step, not an option.

**The chunker or enrichment changes.** Chunk boundaries change, so chunk ids change, so this is also a full rebuild, but often cheaper if embeddings are cached by text hash (`CachedEmbeddings` in `aie_core` does exactly that): unchanged chunk texts reuse their vectors.

**Index parameters change.** A new `M` or `ef_construction`, a switch from IVFFlat to HNSW, or quantization. The vectors stay; only the index structure is rebuilt. In PostgreSQL, `CREATE INDEX CONCURRENTLY` on the new definition followed by dropping the old index does this without blocking writes. `PgVectorStore.reindex()` wraps `REINDEX INDEX CONCURRENTLY` for the same-definition case.

**The index has degraded.** Heavy delete and update churn leaves HNSW graphs with many deleted nodes and IVF lists with stale centroids. Symptoms are rising latency at a fixed `ef_search` and falling ANN recall on the `ann-check` sample. Rebuild on a schedule or when those metrics cross a threshold, not when users complain.

Bulk loads deserve a note. Building an HNSW index after loading the data is much faster than inserting into an existing index row by row, so initial loads and full rebuilds should load the table first and create the index afterwards, with a generous memory budget for the build (`maintenance_work_mem` in PostgreSQL). Incremental ingestion of a few changed documents per run can insert into the live index.

## Implementation

Project 2 lives in `book/projects/p2-semantic-search/`. It depends on `aie_core` for embeddings (`FakeEmbeddings`, `OpenAICompatibleEmbeddings`, `CachedEmbeddings`), settings, and tracing, and adds nothing that `aie_core` already provides.

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

Configuration is environment-driven. The variables that matter for this chapter:

| Variable | Default | Meaning |
|---|---|---|
| `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL` | `fake`, `fake-embedding` | embedder; the model name becomes part of the namespace |
| `EMBEDDING_DIMENSIONS` | `256` | width of the offline hashing embedder |
| `VECTOR_BACKEND` | `numpy` | `numpy` or `pgvector` |
| `INDEX_DIR` | `.index` | where the NumPy store persists |
| `DATABASE_URL`, `PG_TABLE` | unset, `chunks` | pgvector connection and table |
| `HNSW_M`, `HNSW_EF_CONSTRUCTION`, `HNSW_EF_SEARCH` | `16`, `64`, `100` | index build and query parameters |
| `HNSW_ITERATIVE_SCAN` | `relaxed_order` | filtered-scan behavior on pgvector 0.8+ |
| `PG_STATEMENT_TIMEOUT_MS` | unset | per-transaction cap on pgvector reads; unset keeps the server default |
| `INDEX_NAME`, `INDEX_VERSION` | `knowledge`, `v1` | namespace identity |
| `DOCS_DIR`, `CHUNK_MAX_CHARS` | shared-data docs, `1200` | corpus and chunk budget |
| `SEARCH_DEFAULT_K`, `SEARCH_MAX_K` | `8`, `50` | result counts |

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

The domain model is small and has no I/O. Everything an adapter stores is a `VectorRecord`; everything a query constrains is a `SearchFilter`.

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

`SearchHit` (what a search returns: ids, tenant, cosine score, text, tags, metadata) and `DocVersion` (what the store believes is indexed for one document) complete the module on disk.

Filter semantics are defined once, in pure Python, so that the NumPy adapter uses them directly and the pgvector adapter's SQL can be tested against them.

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

The protocol names what the application needs, not what an engine offers. `replace_document` is the only write path ingestion uses; `exact=True` exists so an approximate store can be measured against itself.

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

The full file (about 220 lines, including persistence) is on disk; these are the parts that carry the design: the lazily materialized matrix, atomic document replacement under a lock, and pre-filtered exact search.

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

Persistence writes vectors to `<namespace>.npz` and records (without vectors) to `<namespace>.json` through temporary files and an atomic rename, so a crash mid-save never leaves a half-written index. Loading refuses a dimension mismatch and never unpickles (`allow_pickle=False`), because an index directory is an input file like any other.

### The pgvector adapter

The schema is rendered from a template so that the vector column's dimensions and the HNSW parameters come from configuration. Authorization fields are real columns with indexes, not keys buried in JSONB, because they are in every query.

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

The write path for one document is a single transaction; the read path sets index parameters with `set_config(..., true)`, which is the function form of `SET LOCAL`, so they cannot leak to the next request on a pooled connection.

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

`_apply_timeout` sets `statement_timeout` with `set_config(..., true)` when `PG_STATEMENT_TIMEOUT_MS` is configured, for the same reason `ef_search` is set that way: the setting dies with the transaction, so a pooled connection never carries one request's limit into the next. A search that exceeds it raises instead of holding the caller's thread, and the caller degrades (Chapter 12 skips a retriever that fails or misses its deadline). Note also that the adapter holds a single connection, which serializes concurrent searches; a service with real concurrency should give each worker its own store or put a connection pool behind the adapter, sized from the query rate times the p95 store latency.

The adapter also has `hybrid_search`, which fuses the dense candidates with PostgreSQL full-text candidates (`text_tsv`, a generated `tsvector` column) using Reciprocal Rank Fusion in one SQL statement. It enables iterative scanning for its dense branch too; without that, the dense branch would post-filter and starve selective filters. Chapter 12 owns lexical and hybrid retrieval and explains when fusion helps; the method is here because "pgvector plus `tsvector` in one database" is a large part of the case for PostgreSQL.

### Ingestion

The loader parses the subset of YAML front matter the corpus uses (no YAML dependency), fails closed on a missing ACL, and groups paragraphs under their nearest heading up to a size budget. Chapter 11 replaces this chunker with structure-aware ones; the store does not care which chunker produced its records. The pipeline is where incrementality and pruning live:

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

Authorization is derived from the principal in one method and applied on every query. The API's only job is to turn trusted headers into a principal and refuse requests without one.

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

The evaluator reports recall@k, hit rate, and MRR (Chapter 10 defines these metrics). It runs every gold question through the same `SearchService` the API uses, with a principal built from the row's tenant and groups, collapses chunk hits to documents, and scores them. Rows tagged `forbidden-doc` are scored as leak checks instead of recall, and rows whose required documents were never ingested are reported as ingestion gaps rather than retrieval misses.

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

`semsearch/bench.py` generates clustered synthetic vectors (real embeddings are clustered, which matters for IVF), sweeps `nprobe`, and runs the post-filter experiment whose numbers appear earlier in this chapter.

### The store contract test

One test file defines what "a correct vector store" means and runs against every adapter: NumPy always, pgvector when `DATABASE_URL` is set and the `integration` marker is selected.

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

Follow one query and one document through the code.

**A document enters.** `semsearch ingest` builds an embedder with `make_embedder` (the offline `FakeEmbeddings` hashing embedder by default, an OpenAI-compatible client when configured) and a store with `make_store`, both in `semsearch/config.py` on disk. The namespace comes from `namespace_for`, which combines `INDEX_NAME`, the embedder's model name, and `INDEX_VERSION`. `load_corpus` parses every Markdown file and rejects duplicate document ids, because two files claiming the same id would overwrite each other's chunks. `ingest` compares each document's `index_version` with `store.doc_versions(namespace)`; on the Northwind corpus the first run adds 24 documents and 246 chunks, and the second run reports all 24 as unchanged and embeds nothing. That second number is the one to watch in production: if a no-op ingestion run re-embeds anything, your hashes are unstable (a timestamp in the text, nondeterministic parsing) and you are paying for embeddings you do not need.

**The store writes.** `NumpyVectorStore.replace_document` validates dimensions and ownership (every record must belong to the declared document and version), then, under the store lock, removes all chunks of the document and inserts the new ones. The matrix is marked dirty rather than rebuilt; the next search materializes it once. `PgVectorStore.replace_document` does the same in SQL: advisory lock, upsert, delete the leftovers, all in one transaction.

**A query arrives.** The API's `get_principal` produces a principal or a 401. `SearchService.search` caps k, builds the authorization filter, embeds the query, calls the store, and records a span. In the NumPy store, `search` materializes the matrix if needed, computes the row mask with `matches` for every record (a Python loop, fine for tens of thousands of rows; vectorize it with per-field arrays if you go much larger), multiplies the allowed rows by the normalized query, and selects the top k with `argpartition` followed by a stable sort of just those k.

In the pgvector store, `build_where` produces the clause, the transaction sets `hnsw.ef_search` and iterative scanning, and the query orders by `embedding <=> query`, which the planner serves from the HNSW index when the filter is not too selective and from a filtered scan when it is.

**The evaluator scores it.** `evaluate` builds a principal per gold row, asks for four times as many chunks as the largest k so that enough distinct documents survive the chunk-to-document collapse (several chunks of one document count as one hit), and computes recall@k, hit rate, and reciprocal rank on document ids. It checks forbidden rows for leaks and returns a non-zero exit code from the CLI if any leak occurs, which is what makes it usable as a CI gate.

## Production considerations

**Latency.** Break retrieval latency into embedding the query (often the largest part when the embedder is a remote API, commonly tens of milliseconds, illustrative) and the store call. The span records both separately because they have different fixes: caching query embeddings and running the embedder close to the service for the first, index parameters and filters for the second. Under Northwind's 2-second time-to-first-token target, retrieval including reranking should take a small fraction of the budget; a p95 store latency that creeps up over weeks usually means index degradation or a growing filter that has tipped the planner off the index.

**Memory and cost.** Size the index before choosing it. Raw vectors cost `n * d * 4` bytes in float32: 10 million 1,024-dimensional vectors are 41 GB before any index structure (illustrative). HNSW adds edge lists per vector. Half precision halves the raw size; product or binary quantization with rescoring cuts it by an order of magnitude or more. Embedding cost is paid at ingestion and again at every re-embedding, so an embedding cache keyed by model and text hash pays for itself the first time you change the chunker.

**Security.** Treat the vector store as a copy of the documents, not as an anonymized derivative. Embeddings can leak information about their source text through inversion attacks, and the `text` column is the source text. Apply the same access controls, encryption at rest, and retention rules as the source system. Never accept tenant or group filters from the request body; derive them from verified identity in one function (Chapter 15 covers permission changes and caches on top of this). Keep the store off the public network; the search API is the only client. Parameterize all SQL, including filter values, and validate identifiers such as table names against a strict pattern, as `render_schema` (on disk, in `pg_store.py`) does.

**Replication and availability.** For pgvector, standard PostgreSQL streaming replication covers vectors and indexes; replicas serve read traffic and lag the primary by the replication delay. HNSW indexes are large, so a new replica or a restore from backup takes longer than the row count suggests, and index rebuilds on the primary generate a lot of write-ahead log traffic that replicas must replay. Dedicated vector databases replicate shards across nodes, often with eventual consistency between replicas; read the consistency settings rather than assuming. In both cases the index is derived data, so the ultimate recovery path is rebuilding from the source documents. Know how long that takes, because that number is your worst-case recovery time.

**Deletion.** Deleting a row is not the same as making it unretrievable. In PostgreSQL a deleted row's index entries remain until vacuum removes them (they are filtered out of results, but they cost scan time, and in an HNSW scan they also consume candidate slots, so heavy churn underfills results); in many dedicated stores deletes are tombstones (delete markers) applied at the next segment merge. For compliance deletions, verify with a query that the document no longer appears, purge it from every derived store (caches, evaluation snapshots, logs that contain retrieved text), and record the deletion. For high-churn corpora, schedule vacuum or compaction and watch index size per live row.

**Operations.** The metrics worth dashboards: chunks per namespace (an empty or shrinking active namespace is an outage), ingestion lag (time from source change to searchable), ingestion re-embed count per run, store p50 and p95 latency, underfilled-result rate by tenant, ANN recall from a nightly `ann-check`, retrieval recall from the gold set on every index change, and leak count, which must be zero. The `/healthz` endpoint returns 503 when the active namespace is empty, because an empty index answers every query successfully with nothing.

## Common mistakes

These are the decisions that cause the failures in the next section. Mixing embedding models, post-filtering, and skipping pruning are covered there by symptom and telemetry, so they are not repeated here.

- **Adopting a dedicated vector database for 50,000 chunks.** The team gains a new stateful system to secure, back up, monitor, and keep consistent with the source of truth, in exchange for latency it did not need.
- **Tuning ANN parameters before measuring exact-search recall.** Without the exact baseline, nobody knows whether a recall problem is the index or the embeddings.
- **Treating similarity scores as probabilities.** A fixed threshold of 0.8 that worked for one model drops every result after the model changes.
- **Chunk ids that are not deterministic.** Random UUIDs per ingestion run turn every re-ingest into duplicates, and the duplicates crowd out other documents in the top k.
- **Building the HNSW index before the bulk load.** Every row then pays for a graph search on insert, so the load runs many times slower than loading first and building the index once afterwards.
- **Training IVF centroids on an empty or unrepresentative table.** The centroids describe data that is not there, so real vectors pile into a few lists and both recall and latency degrade until the index is rebuilt.
- **Leaving `ef_search` below k without iterative scans.** The HNSW scan produces only `ef_search` candidates, so a query for 50 results with `ef_search = 40` can never return 50, and filtering shrinks it further.

## Failure modes

| Failure | How it shows up | Telemetry that reveals it | Test |
|---|---|---|---|
| Filtered-ANN starvation | Small tenant or restricted group gets few or no results; assistant abstains on answerable questions | `underfilled = true` rate by tenant; `returned` well below k | Contract test with a selective filter whose only match is far from the query |
| Model mismatch | Relevance collapses after a deploy; scores cluster in a narrow band | Span `embedding_model` differs from namespace model; top-score distribution shifts | Namespace derived from model name; startup check that query model equals namespace model |
| Stale index | Answers cite a superseded policy version | Ingestion lag; document versions in the store differ from source | Ingest, edit content without bumping version, assert a new `doc_version` |
| Zombie documents | Deleted or reclassified documents still retrieved | Doc ids in results that are absent from the source inventory | Ingest, delete source, run with prune, assert gone; run without prune and pin the bug |
| Cross-tenant or ACL leak | A user sees a document outside their tenant or groups | Leak count from the evaluator; audit of returned doc ids against principal | Forbidden-doc gold rows; API tests with headers for each tenant |
| Duplicate chunks | Same text several times in top k, crowding out other documents | Distinct doc ids per result set falls; chunk count grows on no-op ingestion | Upsert idempotency test; second ingest re-embeds nothing |
| ANN recall regression | Retrieval recall drops after an index rebuild or parameter change while embeddings are unchanged | Nightly overlap@k between ANN and exact | `ann-check` with a threshold in CI on index changes |
| Index degradation from churn | p95 latency rises at fixed `ef_search`; ANN recall drifts down | Index size per live row; dead tuples; overlap@k trend | Scheduled rebuild with before/after `ann-check` |
| Empty active namespace | Every query returns nothing, no errors | `/healthz` 503; chunk count per namespace | Health check test on an empty store |

## Tradeoffs

### PostgreSQL with pgvector versus a dedicated vector database

The choice is rarely about raw query speed on a benchmark. It is about operational surface, consistency with the rest of your data, filter behavior, and the scale at which one machine stops being enough. Examples of dedicated systems include Qdrant, Weaviate, Milvus, and managed services such as Pinecone; search engines such as OpenSearch and Elasticsearch, and Redis, also offer vector indexes. Capabilities change quickly, so treat the right-hand column as questions to verify, not facts.

| Dimension | PostgreSQL + pgvector | Dedicated vector database |
|---|---|---|
| Operational surface | One database you probably already run, back up, and monitor | A new stateful system: deployment, upgrades, backups, access control, on-call |
| Consistency with source data | Vectors, metadata, ACLs, and documents in one transaction; read-after-write on the primary | Separate system kept in sync by a pipeline; usually eventual consistency on writes |
| Filtering | Full SQL: joins, arrays, JSONB, row-level security; planner picks index or scan; iterative scans in recent versions | Purpose-built filtered ANN in many engines, often stronger with highly selective filters (small s); filter language is engine-specific |
| Hybrid retrieval | `tsvector` full-text in the same query; reciprocal rank fusion (RRF, Chapter 12) in SQL | Varies: built-in sparse or BM25 in some engines, external in others |
| Scale ceiling | Single-node memory and CPU for the index; replicas for read scale; sharding requires extra tooling | Horizontal sharding and replication as core features; designed for billions of vectors |
| Index options | HNSW, IVFFlat; half-precision and binary types | Often more: PQ, disk-resident indexes, GPU indexes, tiered storage |
| Multi-tenancy | Columns, partitions, schemas, or databases; row-level security | Collections, namespaces, or partitions; per-tenant limits vary |
| Cost model | Existing database capacity; vectors compete with transactional workload | Separate infrastructure or usage-based pricing |
| Team skills | SQL, PostgreSQL operations | Engine-specific APIs and operations |

### Decision matrix

Pick the first row whose conditions hold.

| Situation | Choose | Why |
|---|---|---|
| Under about a million vectors, modest QPS, data can live in memory or a file | NumPy or an in-process exact index | Perfect recall, trivial filters, nothing to operate |
| You already run PostgreSQL; vectors up to tens of millions; filters and joins matter; transactional consistency with documents and ACLs matters | PostgreSQL + pgvector | One system, SQL filters, consistency, hybrid with `tsvector` |
| Hundreds of millions of vectors or more, or index memory beyond one large node, or very high QPS with tight p99 | Dedicated vector database | Sharding, quantization, and filtered ANN as core features |
| Many tenants with strict physical isolation requirements | Per-tenant databases or stores, whichever engine | The contract requires physical separation, which no query filter provides |
| Retrieval is mostly exact lookups by id, code, or field | No vector index; SQL, full-text, or key-value | Structured queries are more accurate and cheaper |
| You need lexical and dense search with heavy text analytics | A search engine with vector support | One engine for BM25, aggregations, and vectors |

Keep the escape hatch open regardless of the choice: put the store behind a protocol like `VectorStore`, keep the documents as the source of truth, and keep an exact path for measurement. Then migrating between engines is a re-ingestion plus a contract test run, not a rewrite.

### When a vector database is unnecessary

RAG means retrieval plus generation; the retrieval mechanism does not have to be vectors. Several common situations need no vector index at all.

The corpus is small. A few hundred documents fit in a NumPy matrix or, often, in the model's context window directly (Chapter 10 compares RAG with long context). Adding a vector database to a corpus that small adds failure modes without adding capability.

The queries are exact lookups. "Status of ticket TCK-1042", "error code RET-002", "invoice INV-2026-0117": these are key lookups or keyword queries. Embedding models are weak at rare identifiers, and SQL or full-text search answers them exactly. The Northwind gold set tags such questions `exact-id` precisely so you can measure dense retrieval on them separately.

The data is structured. Metrics, orders, employee records, and inventory belong in a database queried with SQL, possibly generated by a model over a semantic layer (the `query_metrics` tool; Chapter 36 designs the analytics assistant built on it). Similarity search over table rows loses exact filtering, aggregation, and joins.

The content has a navigable structure. Manuals with a table of contents, legal codes with section numbers, and API references with endpoint names can be retrieved by navigating the hierarchy, by metadata filters, or by a model choosing sections from an outline. These "vectorless" approaches avoid embedding maintenance entirely and can be more precise for well-structured corpora; Chapter 37 implements structured and hierarchical retrieval.

The sensible default for a new project is: start with full-text search and an exact vector scan in the database you already have, measure on real questions, and add an ANN index or a dedicated store when measurements on your workload show you need it.

## Evaluation and testing

Project 2 tests the system at three levels, all offline.

**Contract tests** define the store's behavior once and run it against every adapter: nearest-first ordering, tenant and ACL filters, a selective filter whose only match is far from the query (the test that catches post-filtering), namespace isolation, atomic document replacement, rejection of records that do not belong to the document being replaced, deletion, idempotent upserts, dimension checks, and agreement between default and exact search on small data. The pgvector variants are marked `integration` and run when `DATABASE_URL` points at a database with the extension. Offline tests also check the SQL the adapter generates: the `WHERE` clause is parameterized, the schema has the HNSW and GIN indexes, unsafe table names are rejected, and the checked-in `sql/schema.sql` matches the renderer.

**Component tests** cover the loader (front matter subset, fail-closed ACL, heading-aware chunks, content-hash versioning), ingestion (incremental skip with zero embedding calls, update on content change, pruning, and the deliberately pinned stale-index behavior without pruning), the API (401 without identity, tenant and ACL from headers, tags narrowing, k capping, the underfill span attribute, health check), the metrics (including the worked example: two relevant documents, one found at rank 2, gives recall@5 of 0.5, precision@5 of 0.2, and reciprocal rank 0.5), and the ANN experiments (IVF with full probing equals exact; recall grows with `nprobe`; post-filtering underfills selective filters while pre-filtering does not).

**The retrieval evaluation** runs the shared Northwind gold set (`shared-data/eval/retrieval_gold.jsonl`, 40 questions) through the real service. With the offline hashing embedder, which behaves like a bag-of-words model, the run produced (illustrative, abridged):

```text
scored rows: 37  forbidden rows: 3
recall@1: 0.622   recall@3: 0.919   recall@10: 1.000
MRR: 0.776
ACL leaks: 0
ingestion gaps: []
```

Read these numbers skeptically. The corpus has 24 documents and each user sees at most about 20 of them, so recall@10 of 1.0 means little: returning half the visible corpus finds almost anything. Recall@1 and MRR are the informative numbers at this size. On a real corpus the same harness becomes meaningful, and the per-tag breakdown (`exact-id`, `paraphrase`, `multi-hop`) shows where dense retrieval is weak and where Chapter 12's hybrid retrieval is needed. The leak count is meaningful at any size: the three forbidden rows ask questions about documents the user may not read, and a correct system returns none of those documents. (One of them, RQ-037, turns out to be answerable from a public FAQ as well; Chapter 14 treats it as a gold-label defect. The leak check on it is still valid.) `semsearch eval` exits with status 1 on any leak, so it can gate a deploy. Chapter 14 builds the full retrieval evaluation methodology on top of this harness.

For ANN specifically, the test is the overlap check: run the gold questions against the index and against an exact scan with the same filters, and require overlap@10 above a threshold you choose from the cost curve (for example 0.95) before activating a new index version.

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
