# Exercises — Chapter 15 — Production RAG

Solutions: `../solutions/ch15-solutions.md`


### Knowledge questions

**K1.** Why is a content hash insufficient as an ingestion idempotency key? Name two changes that would be lost.

**K2.** List every place a Northwind document's information lives after ingestion in Project 3, and say which of them the logical delete reaches immediately.

**K3.** Explain why per-tenant namespaces require rank fusion rather than score fusion across partitions.

**K4.** What does the generation counter in a cache key protect against that a TTL alone does not? What does the TTL protect against that generations do not?

**K5.** Why must the authority adjustment survive a reranker failure? What would users observe if it did not?

**K6.** State the four components of freshness lag and which one the registry cannot measure.

### Engineering questions

**E1.** Northwind wants permission tightening to take effect within 30 seconds, while normal edits keep a 5-minute SLO. Design the change to the ingestion path and say what you would measure.

**E2.** Two workers may process two upsert jobs for the same document concurrently. Compare queue partitioning by doc id with a registry row lock (`SELECT ... FOR UPDATE`), covering throughput, failure behavior, and implementation cost on PostgreSQL plus Redis.

**E3.** A tenant contract requires that their data can be deleted completely within 24 hours, including from backups. Which parts of Project 3 satisfy this, and what must you add?

**E4.** Design a cache for query embeddings that is safe to share across tenants. Then argue whether the retrieval cache could also be shared across principals with identical tenant and group sets.

### Practical exercises

**P1.** Add a per-tenant cap on in-flight ingestion jobs, so that a bulk upload from `logistics` cannot push `retail` freshness beyond its SLO. Write a test that enqueues 200 logistics jobs and 5 retail jobs and asserts the retail lag.

**P2.** Diagnose the no-reranker failure: with `RAG_RERANKER=none`, authority still runs, yet RQ-002 ranks the parental leave policy first. Use the trace to find the stage responsible and explain why the authority boost cannot fix it, implement an adjustment so the `conflicting-versions` slice passes without the reranker, then compare the gate results of the three configurations.

**P3.** Implement a source-change canary: a synthetic document whose content embeds a timestamp, edited every minute by a scheduled job. Measure end-to-end freshness (source edit to searchable) from outside the service and expose it as a metric.

**P4.** Add a `PgLexicalIndex` adapter using ragkit's `sql/lexical_tsvector.sql`, so the API no longer needs BM25 snapshots. Keep the `Retriever` contract and the pre-filter on tenant and groups, and run the existing tests against both lexical backends.

### Debugging exercises

**D1.** After a weekend restore of the vector database from Friday's backup, users report a cancelled travel policy appearing in answers. The registry shows the policy as `deleted` since Saturday. The answer cache is empty after the restart. What happened, which telemetry confirms it, and what should have prevented it?

**D2.** Retail's answer quality on paraphrased questions dropped for two hours on Tuesday, but error rates and latency were normal. The trace for a sample request shows `degraded: ["retrieve:dense#q0"]`, and the dense breaker shows `open` in `/healthz`. The vector database dashboard shows it healthy since 09:05, while degradation lasted until 11:10. Diagnose.

**D3.** An HR admin restricted the parental-leave policy to `hr` at 10:00. At 10:20 an employee still received an answer citing it. The status endpoint shows queue depth 3,400 and freshness p95 of 42 minutes, and the upsert job for the policy is still queued. The employee's response has `cache: "miss"`. Explain the chain of causes and the two fixes, one immediate and one structural.
