# Exercises — Chapter 8 — Embeddings

Solutions: `../solutions/ch08-solutions.md`


**Start here:** K1, K3, E1, P1, D2 (about 4 hours). The rest go deeper.

### Knowledge questions

**K1.** Explain why cosine similarity, dot product, and Euclidean distance produce identical rankings for unit vectors, and give a concrete case where dot product and cosine disagree for unnormalized vectors.

**K2.** What does Matryoshka representation learning change about how a model is trained, and why must a truncated vector be re-normalized before a dot-product search?

**K3.** A teammate proposes reusing last year's index and embedding only new documents with the newly adopted model "to save cost." Explain precisely what goes wrong and what the correct migration looks like.

**K4.** Why does the choice of negatives in contrastive training determine what "similar" means for a model? Give an example of a distinction a general-purpose model is likely to miss.

**K5.** What is anisotropy in an embedding space, how do you detect it, and why does it make similarity thresholds non-transferable between models?

**K6.** Why should an embedding cache key include more than the model name and the text? List the fields this chapter's space fingerprint covers.

### Engineering questions

**E1.** Northwind wants to auto-close duplicate support tickets. Design the threshold selection process, including the labeled set you would build, the precision floor, what happens below the floor, and how you would re-validate after a model change.

**E2.** You must choose between a hosted embedding model and a smaller self-hosted model for the `logistics` tenant, whose documents may not leave the EU region. Describe the evaluation you would run, the metrics you would compare, and the operational costs of each option.

**E3.** The p95 time-to-first-token target for RAG answers is 2 seconds. Embedding a query currently takes 180 ms at p95 through a hosted API, and the same message is embedded separately by the router, the classifier, and retrieval. Propose changes and estimate their effect.

**E4.** Design a resumable re-embedding job for 5 million chunks under a provider rate limit, including checkpointing, double-writes, shadow evaluation, cutover criteria, and rollback.

### Practical exercises

**P1.** (about 2 hours) Add a `dimensions` parameter to the experiment: using `truncation_report`, find the smallest prefix size at which recall@3 on the Northwind labeled queries stays within 0.03 of full-dimension recall. Then extend `VectorIndex` so a truncated space can be searched with a two-stage strategy: truncated vectors for top-50 candidates, full vectors to rescore.

**P2.** (about 90 min) Extend the labeled retrieval set with 15 queries containing identifiers (error codes, ticket numbers, endpoint names) and report recall@3 on that slice separately. Add a minimal lexical boost (for example, exact token overlap on identifier-like tokens) and show its effect on the slice and on the rest of the set.

**P3.** (about 2 hours) Implement a `CentroidClassifier` variant with several centroids per class (k-means within each class) and compare it with the single-centroid and kNN classifiers using `leave_one_out` on the tickets. Report accuracy on answered items and coverage at a fixed abstention rule.

**P4.** (about 90 min) Back the embedding cache with Redis by passing a Redis-like mapping as the `store` to `EmbeddingPipeline`. Add a TTL, a per-namespace key count metric, and a test that a space change produces zero hits on the first pass.

### Debugging exercises

**D1.** After a Tuesday deploy, the router's fallback rate drops from 18 percent to 2 percent and users report being sent to the wrong knowledge base. No errors appear in logs. The embedding spans show the same model name as before. Cache hit rate on query embeddings is 94 percent, unchanged. What happened, and which telemetry confirms it?

**D2.** Recall@5 on the gold set drops from 0.88 to 0.71 overnight. The index was not rebuilt. The query service's embedding spans show a new space fingerprint, but the model name is unchanged. A diff of the deploy shows a change to the text normalization function that lowercases input. Explain the mechanism and the fix, and say what should have prevented the incident.

**D3.** The dedup job, running with a threshold of 0.82 chosen last quarter, starts merging tickets about different stores' payment terminals. The embedding model was "upgraded" by the provider under the same name a week ago. The similarity profile of the probe set shows random-pair mean cosine rising from 0.21 to 0.58. Diagnose, and describe the remediation and the monitoring that would have caught it earlier.

**D4.** The nightly re-index of the Northwind knowledge base used to send a few thousand tokens (only changed documents). For the last nine nights it has re-embedded the entire corpus, taking four hours, and on two mornings retrieval returned `SpaceMismatchError` for twenty minutes. No document count changed much. Two consecutive nights of the job's spans, abbreviated:

```json
{"name": "embed", "kind": "passage", "model": "nw-embed-v3", "space": "4be1c09a77d2e310", "texts": 225, "unique": 225, "cache_hits": 0, "cache_misses": 225, "retries": 0, "tokens_sent_total": 58210}
{"name": "index.swap", "old_space": "9f03aa5e10c4b8d2", "new_space": "4be1c09a77d2e310", "reason": "changed: post_process"}
{"name": "embed", "kind": "passage", "model": "nw-embed-v3", "space": "c7d58e01b9a3f442", "texts": 225, "unique": 225, "cache_hits": 0, "cache_misses": 225, "retries": 0, "tokens_sent_total": 58204}
{"name": "index.swap", "old_space": "4be1c09a77d2e310", "new_space": "c7d58e01b9a3f442", "reason": "changed: post_process"}
```

The release notes for the change nine days ago say: "Improve similarity spread with mean-centering (post_process = mean-center:<hash of corpus mean>)." What is wrong, why does it cost a full re-embed every night, why do the morning errors happen, and what is the fix?
