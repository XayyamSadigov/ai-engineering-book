# Exercises — Chapter 12 — Retrieval Engineering

Solutions: `../solutions/ch12-solutions.md`


**Start here:** K2, K6, E1, P2, D2 (about 4 hours). The rest go deeper.

### Knowledge questions

**K1.** Explain what each of idf, the `k1` saturation term, and the `b` length normalization contributes to a BM25 score. What does BM25 reduce to when `k1 = 0`? When `b = 0`?

**K2.** RRF with k = 60 and a document ranked 1st by one retriever and absent from the other, versus a document ranked 5th by both. Which ranks higher, and why is that usually the desired behavior? What changes with k = 0?

**K3.** Why is a cross-encoder more accurate than a bi-encoder (dense retriever) at the same model size, and why can it not replace the first stage?

**K4.** HyDE embeds a hypothetical answer instead of the question. Name two query types where it tends to help, two where it tends to hurt, and explain why the pipeline never sends HyDE passages to BM25.

**K5.** Why does PostgreSQL's `ts_rank_cd` behave differently from BM25 on a query containing one rare identifier and one common word? Does the difference matter for RRF fusion? For weighted fusion?

**K6.** What is the difference between candidate recall and final recall, and what does it mean when candidate recall is 0.95 and final recall at 5 is 0.70?

**K7.** How does a learned sparse retriever (SPLADE-style) differ from BM25 in what it stores and how it scores? Name one query type where you would expect it to beat BM25 and one where it may lose.

### Engineering questions

**E1.** Northwind adds a ticket corpus of two million support tickets to the existing documents. Propose `candidate_k`, `rerank_k`, and `final_k` values and a reranker choice for a 400 ms p95 retrieval budget, and describe the measurement you would run to validate each number.

**E2.** A team proposes to drop BM25 because their new embedding model scores higher on a public benchmark. What evidence from Northwind's gold set and query logs would you require before agreeing, and which slices would you check first?

**E3.** Design the cache key for a retrieval cache (query to candidate ids) in a multi-tenant deployment. List every component, explain what goes wrong if each is omitted, and say how the contextual-retrieval prompt version interacts with it.

**E4.** The LLM reranker reads untrusted chunks. Describe how an attacker who controls one document could try to manipulate ranking, what the current design limits, and one further control you would add.

### Practical exercises

**P1.** (about 3 hours) Add a `FieldWeightedBM25Index` (BM25F-style) that indexes the breadcrumb and the body as separate fields with configurable weights. Show on the gold set whether weighting the breadcrumb higher improves hit@1, and add a test that fails if field weights are ignored.

**P2.** (about 2 hours) Implement a `funnel_sweep` command in the comparison script that reports, for `candidate_k` in {10, 30, 100} and `rerank_k` in {5, 10, 20, 40}, candidate recall, final MRR, and p50 and p95 latency. Run it with an artificially slowed reranker (a fixed delay per candidate) and recommend a configuration for a given latency budget.

**P3.** (about 2 hours) Add a freshness-aware reranker wrapper that, among candidates whose documents share tags and disagree in version, boosts the most recently updated one by a configurable amount. Show its effect on the `conflicting-versions` slice and on every other slice.

**P4.** (about 3 hours) Implement a cheap "needs rewriting" classifier (rules or a small model) that skips `QueryRewriter` for standalone questions. Measure how many rewriter calls it saves on a synthetic conversation set and whether retrieval quality on follow-ups is unchanged.

### Debugging exercises

**D1.** After a deploy, hit@1 on the `exact-id` slice drops from 0.8 to 0.2 while paraphrase questions are unchanged. Traces show the `bm25#q0` stage returning candidates normally, with `bm25_matched_terms` of 1 for the top hits of identifier queries. The deploy changed only "text normalization utilities". Diagnose.

**D2.** Users in the logistics tenant report that answers about RoutePilot are fine but answers about the scanner guide became "I could not find this" yesterday afternoon. Traces for those requests show `bm25#q0` with no candidates, `dense#q0` returning only route-planner and tracking chunks with low scores, `acl_dropped` of 0, and no degraded stages. The same questions work for a logistics manager. Identify what to look at and the likely root cause.

**D3.** An LLM reranker was enabled for HR questions. Offline MRR improved, but in production the p95 time to first token rose from 1.6 s to 3.9 s, and 4 percent of HR answers now cite the vendor newsletter. Traces show `rerank_failed` on 3 percent of requests and the newsletter chunk with `llm_grade` 3 on the citing requests. Explain each symptom and what you would change.
