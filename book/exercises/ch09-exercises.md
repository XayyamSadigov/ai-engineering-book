# Exercises — Chapter 9 — Vector Search and Vector Databases

Solutions: `../solutions/ch09-solutions.md`


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
