# Chapter 15 — Production RAG: Solutions

## Knowledge questions

**K1.** A content hash covers only the document text. Changes that leave the text identical but change what the indexes must hold are deduplicated away:

- **An ACL change.** For example `["all"]` becomes `["hr"]`. The document stays visible to everyone.
- **A tenant move.**
- **A new authority or supersession rule.** The FAQ keeps outranking the policy.
- **A chunker or pipeline change.** Old chunk ids keep being served, and the new configuration never reaches unchanged documents.
- **Turning contextual enrichment on.**

Project 3's fingerprint (`doc_fingerprint`) includes all of these.

**K2.** After ingestion, a document lives in:

- its registry row, which holds chunk ids, context keys and versions;
- the BM25 index of every version and partition holding it;
- the vector store namespace of every such version and partition;
- the BM25 snapshot files on the shared volume;
- the embedding cache (vectors keyed by text, with the per-document tag list);
- the contextual-prefix cache, if enrichment is on;
- the retrieval cache and the answer cache on every API replica;
- the blob store, if the document was uploaded;
- traces, which hold ids and hashes only under the capture policy;
- backups of all of the above.

The logical delete reaches these immediately:

- the registry, which is set to `deleting` and gets a new `deleted_seq`;
- every generation-keyed cache entry on every replica, since the generation bump makes them unreachable;
- the tagged cache entries on the replica that served the request, which are physically purged;
- retrieval visibility, because `TombstoneFilter` hides the document.

Everything else is reached by the purge job. Backups need a separate policy.

**K3.** BM25 scores depend on corpus statistics: document frequency, document count, and average length. Each namespace has its own statistics, so the same chunk would score differently in two indexes. Dense similarities are comparable across namespaces only when they come from the same embedding model, and even then their distributions differ by partition. Raw scores from different partitions are therefore not on one scale. RRF uses only ranks within each list, so it fuses heterogeneous lists without calibration.

**K4.** A generation bump invalidates every older entry at the moment a change commits, for every replica, with no waiting. A TTL alone would keep serving a deleted or re-ACLed document until expiry. A TTL bounds staleness that the registry cannot see: a source edited without a sync, an external fact that aged, or a generation bump lost because of a bug. It also bounds memory.

**K5.** The authority layer is what fixes FAQ-over-policy. If it ran inside the relevance reranker, any reranker failure would degrade to the fused order without it. Users would then get the stale "5 days" FAQ answer for PTO questions during a reranker outage. The answer would carry citations and no error, which makes it a correctness regression nobody notices. `AuthorityReranker` catches the inner failure, records `rerank:fallback`, and still applies authority and supersession.

**K6.** The four components are detection (source change to observation), queue wait, processing (parse, chunk, embed, write), and publication (for example the snapshot reload). The registry stamps `submitted_at` at observation, so it cannot measure detection lag. Measuring detection needs a source-side timestamp, such as a webhook event time or the file mtime, or an external canary (P3).

## Engineering questions

**E1.** Treat permission tightening as a priority change:

- Classify the change in the producer by comparing the new ACL with the registry row.
- Enqueue tightenings on a dedicated high-priority queue with its own worker pool, or apply a synchronous **logical restriction**. The logical restriction writes the new ACL to the registry immediately and bumps generations. Retrieval checks the registry ACL for hit documents through a small wrapper, like `TombstoneFilter`, until the worker rewrites the indexes.
- Loosening can stay on the normal path. Being briefly too strict is safe.

Measure the following:

- the lag distribution per change class (tightening, loosening, content);
- a shadow check on every response that counts cited chunks whose registry ACL no longer admits the principal (target zero);
- the depth and age of the priority queue.

**E2.** The two options compare as follows.

- **Partitioning by doc id.** Use N queues and assign each document to one by hash. Each partition is processed by one consumer at a time, so jobs for one document are serialized without database contention.
  - *Costs.* The partition count bounds parallelism. A hot partition, such as one huge document or a bulk tenant, delays everything hashed with it. Rebalancing on worker failure needs lease logic per partition. `RedisJobQueue` would need N namespaces and a lease-holder per namespace.
- **Registry row lock.** The handler opens a transaction and runs `SELECT ... FOR UPDATE` on the document row at start. It writes the indexes and commits the registry update. A second job blocks or skips with `NOWAIT` and retries later.
  - *Benefits.* Parallelism is unlimited across documents, and nothing changes in the queue.
  - *Costs.* A long transaction spans external writes to vectors and Redis. Lock timeouts must sit below the lease timeout. A crashed worker's lock is released by connection loss, which is acceptable. It does not work with the in-memory registry.

For Northwind, use the row lock with `NOWAIT` plus a retry with backoff, and keep the sequence-number guards as the correctness backstop. Both options still need those guards, because the lock serializes jobs but does not order them.

**E3.** Already satisfied:

- the logical delete takes effect immediately;
- the purge removes data from indexes, vectors, the embedding cache, the prefix cache, blobs, and request caches;
- reconciliation removes data that reappears from restored snapshots;
- tombstones prevent re-ingestion.

To add:

- **Tenant-wide deletion.** Enumerate the registry by tenant and enqueue purges. In namespace mode, drop the tenant's partitions with `drop_namespace` and remove the snapshot files.
- **Backups.** Either expire backups within the window (keep 24 hours or less), or keep a deletion log that is replayed after any restore and checked by reconciliation. Encrypting per-tenant data with per-tenant keys and destroying the key (crypto-shredding) is the usual way to satisfy "from backups" without rewriting them.
- **Traces and logs.** Confirm they hold no document text. Otherwise give them the same retention or a purge job.
- **The provider side.** Check the model provider's retention terms for prompts that contained the data.
- **An audit report per deletion** that lists each store and the count removed. The purge handler already returns those counts, so persist them.

**E4.** For query embeddings, use `CachedEmbeddings` keyed by space fingerprint plus exact query text. This is safe to share because the vector is a pure function of the text. Two precautions apply. The query text is user data, so give the entries a TTL and do not log keys with text. A cache hit does reveal that someone asked the same text before, so keep timing differences out of observable responses where that matters.

The retrieval cache could be shared across principals whose (tenant, sorted groups) sets are identical, provided that retrieval depends on nothing else about the principal. That holds in Project 3, where the key is not per-user. `scoped_cache_key(per_user=False)` already keys on the group set rather than the user. The condition breaks as soon as anything per-user enters retrieval: personal documents, per-user boosts, or conversation history in query rewriting. Then the key must be per-user.

## Practical exercises

**P1.** The expected implementation:

- **Track counts.** Keep an in-flight count per `tenant_id`, in Redis `HINCRBY` or in the job queue.
- **Cap leasing.** When leasing, skip jobs whose tenant has reached its cap, for example `max_inflight_per_tenant=2`. Alternatively, keep per-tenant ready sets and lease round-robin across tenants that have work.
- **Release on outcome.** Release the count on ack, nack, and lease expiry, which means wiring it into `_reclaim_expired`.

The test uses the in-memory queue and a manual clock:

- enqueue 200 logistics jobs, then 5 retail jobs;
- run a worker with a fixed per-job time;
- assert that every retail job completes within 5 job durations, not after the 200 logistics jobs;
- assert that logistics throughput stays at least (cap / workers) of capacity.

**P2.** Add a retriever wrapper, or a post-fusion step in `build_pipeline`, that applies the same `superseded_by` ceiling and a smaller authority boost to the fused list before it is cut to `rerank_k`. Keep `AuthorityReranker` for the reranked path.

Acceptance criteria:

- with `RAG_RERANKER=none`, the critical `conflicting-versions` hit@1 rule passes and the gate passes;
- the default configuration's metrics do not regress;
- the report compares the three configurations: lexical with the authority reranker, none with fusion-stage authority, and none without it.

Write in the report whether the higher no-reranker hit@1 survives.

**P3.** The canary has four parts:

- **The canary document.** Create `canary-freshness.md` in a dedicated `ops` tenant with `acl_groups: ["ops-canary"]`. Its body contains a token such as `CANARY-<unix-ts>`.
- **The writer.** A scheduled job rewrites the file every minute and triggers the normal detection path, either a webhook or the next sync.
- **The probe.** A probe running as an `ops-canary` principal searches for the token through `/v1/ask` (or retrieval only). It records `now - ts` when the current token first appears.
- **The metric.** Export `rag_canary_freshness_seconds` and alert when it exceeds the SLO for 3 consecutive probes.

The canary must never be visible to employees: check this with a test like the forbidden-doc tests. It must also be excluded from the evaluation set.

**P4.** Implement `PgLexicalIndex` with this shape:

- **Methods.** `retrieve(query)`, `replace_document(doc_id, chunks)`, `delete_document(doc_id)`, and `__len__`, over the table and `tsvector` column defined in `ragkit/retrieval/sql/lexical_tsvector.sql`. Use `plainto_tsquery` or `websearch_to_tsquery`, `ts_rank_cd`, and `WHERE tenant = ANY(%s) AND acl_groups && %s` applied in the same query, which pre-filters before ranking.
- **Identity.** Return `ScoredChunk` with `stage="bm25"`, so stage isolation and the trace format are unchanged.
- **Wiring.** Make `SearchIndex` accept any lexical index with that interface. Skip snapshots when the lexical index is database-backed.
- **Tests.** Run the existing suite with a parametrized fixture: the in-memory BM25 offline, and Postgres under `@pytest.mark.integration`.

Acceptance criteria: identical ACL behavior, and deletions verified through a document-listing query. Expect recall@5 differences against BM25. Report them, because PostgreSQL's ranking is not BM25.

## Debugging exercises

**D1.** **Root cause:** the vector database restore brought back Friday's vectors, including the travel policy, which was purged on Saturday. The registry was not restored and still says `deleted`. Nothing reconciled the restored store against the registry. The API process was restarted, but `Container.startup()` (refresh plus reconcile) either was not called or ran before the restore finished. Dense retrieval therefore returned the policy's chunks.

**Telemetry that confirms it:**
- the cited chunk ids belong to a doc whose registry status is `deleted`;
- a manual `reconcile()` would report removing `hr-travel-policy` from `v1/all`;
- `dense_doc_ids()` contains it while the BM25 snapshot does not, which is why it appears with dense-only signals in the hit `signals`.

**Prevention:**
- reconcile after every restore, and block traffic until it finishes;
- run reconciliation on a schedule and alert when it removes anything;
- add a retrieval-time guard that drops hits whose registry status is not `active`, generalizing `TombstoneFilter` from `deleting` to all non-active states, as a last line of defense.

**D2.** **Root cause:** the dense breaker is stuck open, or keeps re-opening, after the database recovered.

Likely causes:
- **Half-open probes fail for a different reason**, for example a connection pool still holding dead connections after the database restart. Every probe then errors, the breaker re-opens, and its open interval grows toward `max_open_s`.
- **Probes never run**, because the retrieval cache or admission level-1 traffic bypass dense retrieval entirely.

The two hours match exponential open-time growth until the pool recycled its connections.

**Telemetry:**
- the dense breaker's state transitions and probe results (`snapshot()`);
- `rag_degraded_total{reason="retrieve:dense#q0"}` staying flat while database health is green;
- probe-error messages showing connection errors rather than query errors.

**Fixes:**
- use `pool_pre_ping` or connection recycling on the vector store client;
- make probe errors visible;
- alert on any degraded rate above 0 for more than 5 minutes, independent of the error rate;
- cap `max_open_s` to a few minutes for dependencies that recover quickly.

**D3.** **Chain of causes:**
1. The ACL change was correctly fingerprinted and enqueued, but it sits behind a 3,400-job backlog in the same FIFO queue as everything else, probably a bulk import. Freshness p95 of 42 minutes confirms the backlog.
2. Until the upsert runs, the registry, the indexes and the generations still reflect the old ACL. The employee's request missed the cache (`cache: "miss"`), so this is not a cache bug. The indexes still say `["all"]`.

**Immediate fix:** apply a logical restriction for that document right now. Either write the new ACL to the registry and filter hits against it, as in E1, or run the specific job inline. An admin "process now" endpoint can do this, as can `container.handlers.upsert` on that job. Then bump the generations of the affected scopes.

**Structural fix:**
- separate queues, or priorities, for permission tightening and deletions versus bulk content;
- per-tenant fairness on leasing (P1);
- a freshness SLO for tightenings measured separately, with an alert;
- bulk imports at batch priority, which admission control can defer.
