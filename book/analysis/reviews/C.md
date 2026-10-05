# Review group C: Chapters 9 to 14 (RAG progression)

Scope: chapters 09-14, their solutions, `book/projects/p2-semantic-search`, `book/projects/examples/ch10`,
`book/projects/ragkit` (ingestion, retrieval, generation, eval). Ch 15 and `p3-rag-assistant` were not edited;
their tests were run after every shared-package change.

Baseline before edits: ragkit 253 passed, p2 49 passed, ch10 20 passed, p3 45 passed; every code listing in
Ch 9 to 14 matched the file on disk (line-set check, dedented excerpts included); exercise ids matched solution
ids in all six chapters; no em dashes outside titles; test counts quoted in chapters (71, 66, 43, 73) were exact.

## Gaps found (by reviewer role)

### Principal AI Engineer
- Coverage check against the requested progression: ingestion (parsing, cleaning, normalization, metadata,
  formats, OCR, tables, PDFs), chunking (fixed, recursive, sentence, section-aware, semantic, parent-child),
  retrieval (dense, BM25, Postgres tsvector, hybrid RRF and weighted, filters, multi-query, rewriting,
  decomposition, HyDE), improvements (lexical, cross-encoder and LLM rerankers, contextual retrieval,
  parent-document, fusion), generation (contract, packing, citations, abstention, conflicts, hallucination
  reduction, streaming), evaluation (hit/recall/precision@k, MRR, nDCG, context relevance, faithfulness,
  rubric coverage, answer relevance, citation P/R, abstention, stage isolation) are all present and developed.
  Vector DB vs pgvector (comparison table + decision matrix) and "when a vector DB is unnecessary" are in Ch 9.
- Ch 11: PDF section named the layout and table problems (column interleaving, tables as space-separated
  lines) but gave no remedy beyond the text layer and an OCR seam.
- Ch 10 -> 11 -> 13 chain for the stale-FAQ failure was broken in the middle: Ch 13's packer reads
  `supersedes` / `effective_date` metadata, but no chapter said where that metadata comes from or that
  ragkit's parser already carries arbitrary front-matter keys onto chunks.
- REVIEW TODO RQ-037 (gold row labeled forbidden-doc/abstain but answerable from the public HR FAQ) was
  described in Ch 14 as undecided; Ch 9 stated that all three forbidden rows have answers only in restricted
  documents, which is false for RQ-037.

### Senior Software Architect
- Ch 12 claimed "put a timeout on every stage and degrade" but `RetrievalPipeline` had no deadlines: a hung
  retriever or reranker blocked the request indefinitely (only exceptions degraded).
- `RetrievalPipeline` created a new `ThreadPoolExecutor` per request (thread churn under load, no way to share
  a bounded pool).
- P2 `PgVectorStore` had no query timeout and holds one connection that serializes concurrent searches.
- Ch 13 / `GroundedQA`: behavior on provider failure (rate limit, outage, malformed output after repair) was
  undocumented and untested; risk that callers map it to `insufficient_evidence`.

### AI Educator
- Ch 10 repeated Ch 14's worked metric example (recall@5 0.5, precision@5 0.2) in full; Ch 14 owns metrics.
- Ch 9 cited Chapter 16 for the `query_metrics` semantic-layer tool (the analytics assistant is Ch 36), and
  did not point to Ch 37 for vectorless/structured retrieval; Ch 10 did not point to Ch 9 for "no vector index
  needed" or to Ch 37 for long-context hybrids.
- Ch 13 sent the reader to an exercise (Chapter 10, K3) for an explanation instead of the demo that contains it.
- Ch 11 said "Its 71 tests" for a package that now has 259 tests (71 are the ingestion tests).
- No answer leakage into Exercises sections; K/E/P/D present everywhere; debugging exercises present broken
  traces; mental-model callouts are used, not decorative.

### Production/SRE Engineer
- No per-stage deadline metric or alert for retrieval (only degraded rate).
- No degraded-mode guidance for generation outages; no failure-mode entry for "outage shown as abstention".
- Ch 9: no statement timeout or connection-pool sizing guidance for the pgvector path.
- Observability, cost, deletion, re-embedding, and runbook material in Ch 9 to 14 was otherwise concrete.

## Changes applied

Code (all backward compatible, defaults unchanged):
- `ragkit/retrieval/pipeline.py`: `RetrievalPipeline(..., retrieve_timeout_s=None, rerank_timeout_s=None,
  executor=None)`. First-stage jobs run under `wait(..., timeout)`; an unfinished job becomes a stage record
  with `timed_out=True` and `error="TimeoutError: ..."`, so fusion, `trace["degraded"]` (`retrieve:<list>`) and
  Ch 14's stage isolation treat it like a failed retriever. Reranker deadline degrades to fused order
  (`rerank:TimeoutError`). Optional shared executor (never shut down by the pipeline); short-lived pools are
  shut down without waiting for stragglers. `ragkit/retrieval/settings.py`: `RAGKIT_RETRIEVAL_RETRIEVE_TIMEOUT_S`,
  `RAGKIT_RETRIEVAL_RERANK_TIMEOUT_S`, wired through `from_settings`.
- `tests/test_retrieval_pipeline.py`: +4 tests (hung retriever skipped within the deadline, all-timed-out
  raises `RetrievalError`, slow reranker keeps fused order, shared executor reused and left open, settings).
- `ragkit/eval/rag_dataset.py`: `gold_row_to_case` accepts an optional explicit `forbidden_doc_ids` row field
  ("answer from required, never touch this neighbor"); rows without it convert exactly as before.
  `tests/test_rag_eval_dataset.py`: +1 test (corrected RQ-037 form; current RQ-037 unchanged).
- `tests/test_generation_generator.py`: +1 test pinning that `GroundedQA` propagates `LLMError` and that the
  packed evidence remains usable for a sources-only response.
- `p2-semantic-search/semsearch/adapters/pg_store.py`: `PgVectorStore(..., statement_timeout_ms=None)`, applied
  with `set_config('statement_timeout', ..., true)` in `search` and `hybrid_search`; `SearchSettings.pg_statement_timeout_ms`
  (`PG_STATEMENT_TIMEOUT_MS`) wired in `make_store`. `tests/test_pg_sql.py`: +1 offline test with a recording
  connection (no database needed).

Chapters:
- Ch 9: re-pasted the `PgVectorStore.search` listing; new paragraph on `statement_timeout` (transaction-local)
  and the single-connection limitation with pool sizing rule; config table row `PG_STATEMENT_TIMEOUT_MS`; test
  count 50; `query_metrics` cross-reference fixed to Ch 36; vectorless pointer to Ch 37; RQ-037 caveat next to
  the leak-count paragraph.
- Ch 10: replaced the duplicated worked metric example with a cross-reference to Ch 14 (kept the "encode
  required evidence" rule); added pointers to Ch 9 ("vector index unnecessary") and Ch 37 (long-context hybrids).
- Ch 11: intro and run commands now say 71 ingestion tests; new paragraph on PDF layout and table recovery
  (coordinate-aware extraction, table detectors, model-based page parsing) with cost, failure class, routing
  by page, per-page parser recorded as a slice, evaluation on real pages, Ch 37 for figures; new paragraph on
  the freshness/authority metadata convention (`effective_date`, `supersedes` in front matter flow to
  `Chunk.metadata`, read by Ch 13's packer; must come from the document system or a reviewed authority map;
  Project 3 supplies it).
- Ch 12: re-pasted the `retrieve()` listing; added the `_run_jobs` listing with an explanation of deadline
  semantics, why a deadline does not cancel work (client timeouts, `statement_timeout`), executor sizing for
  stragglers, and the circuit breaker (Ch 29, Project 3); config table rows for both timeouts; latency section
  and production section now name the parameters, illustrative values, and an alert on timed-out rate per
  retriever; test count 70.
- Ch 13: new "Failure recovery" production paragraph (sources only, fallback model via Ch 7 router,
  unavailable; never `insufficient_evidence`; count modes separately; Project 3 / Ch 29); new failure mode
  "Provider failure reported as abstention" with telemetry and test; supersession paragraph links to Ch 11's
  convention; K3 exercise reference replaced by the demo; test count 44.
- Ch 14: new "A known label defect: RQ-037" paragraph with the verdict (label wrong, leak probe valid), the
  corrected JSON row, why shared-data stays unchanged this edition, and how to ship the fix (dataset version
  bump, baseline rerun, changelog); the code-walkthrough paragraph now records the decision and the corrected
  abstention reading; re-pasted the `gold_row_to_case` listing and recomputed every "excerpt: lines A-B"
  marker for the eval modules; dataset-test bullet updated; test count 74.
- `analysis/07-integration-notes.md`: RQ-037 TODO marked RESOLVED with the decision; new notes for the
  retrieval deadlines/executor, GroundedQA error contract, and P2 statement timeout.

Verification after edits (from repo root, `-p no:cacheprovider`):
- `book/projects/ragkit`: 259 passed
- `book/projects/p2-semantic-search`: 50 passed, 12 deselected (pgvector integration)
- `book/projects/examples/ch10`: 20 passed
- `book/projects/p3-rag-assistant`: 45 passed
- listings vs disk: all match in Ch 9, 10, 12, 13, 14; Ch 11 has one known false positive of the checker
  (a line containing three backticks inside a string literal), verified by hand
- `build_book.py --check`: only "chapter 15 missing" and "chapter 39 missing" (being written by other agents)

## Second-pass findings

Quick second pass over all four roles after the edits:
- New prose follows style rules (no em dashes, numbers labeled illustrative, vendor neutral, chapter numbers
  match the TOC). Fixed one arithmetic slip in my own Ch 14 text: under the corrected RQ-037 label the
  abstention delta understates the candidate by two cases (candidate's answer becomes correct and the
  baseline's abstention becomes a false abstain), not one.
- The scratch listing checker was overwritten by another agent in the shared scratchpad; re-created under a
  unique name and re-ran all checks.
- Confirmed that Project 3 writes the same `supersedes` / `effective_date` keys (its `domain/authority.py`),
  so Ch 11's new convention paragraph and Ch 13 agree with Project 3. Project 3 additionally uses
  `superseded_by` for its own `AuthorityReranker`; that is Ch 15's to explain.
- Confirmed Project 3 wraps retrievers in circuit breakers but sets no per-call deadline, so the new pipeline
  deadlines are additive for it (defaults off; no behavior change in its tests).
- No remaining duplicated material found between Ch 9 to 14: hybrid SQL in Ch 9 defers to Ch 12; Ch 12's
  filter section defers to Ch 9's mechanics; parent-child (Ch 11) and parent-document (Ch 12) cross-reference;
  Ch 13 defers lost-in-the-middle to Ch 5 and judges to Ch 24; Ch 14's stage tree builds on Ch 10's debugging tree.

## Remaining open items

- `p2-semantic-search/.env.example` (and ragkit's) not updated with `PG_STATEMENT_TIMEOUT_MS` /
  `RAGKIT_RETRIEVAL_*_TIMEOUT_S`: the agent permission rules deny reading or writing `.env*` files. The
  variables are documented in the chapters' configuration tables. A human should add the three lines.
- `shared-data/eval/retrieval_gold.jsonl` RQ-037 deliberately left unchanged (lead instruction and pinned
  numbers across chapters); the corrected row is documented in Ch 14 for the next dataset version.
- `PgVectorStore` still uses one connection per store instance; the chapter now documents the limitation and
  the sizing rule, but a pooled adapter was not added because pgvector is unavailable in this environment
  (integration tests skipped), so it could not be verified against a real database. The new
  `statement_timeout` code is likewise verified only by the offline SQL-recording test.
- Suggestion for w5-ch15 (not applied, out of scope): Project 3 can pass `retrieve_timeout_s`,
  `rerank_timeout_s`, and a shared `executor` to `RetrievalPipeline`, and `statement_timeout_ms` to
  `PgVectorStore`, to complement its circuit breakers.

## Follow-up: MMR gap (raised by review B, assigned by the lead)

Gap: Ch 5 teaches maximal marginal relevance and says the diversity step belongs after reranking in
Ch 12; Ch 8 implements `mmr` in `examples/ch08/embedlab`; ragkit.retrieval had no MMR stage.

Changes:
- New `ragkit/retrieval/diversity.py`: `MMRDiversifier(lambda_=0.7, min_relevance=0.0, embeddings=None)`
  with `Reranker` shape, plus `mmr_order`, `normalized_relevance`, `jaccard_matrix`, `cosine_matrix`.
  Relevance is the incoming score divided by the pool maximum for non-negative scores (min-max only for signed
  scores). A first draft used min-max, and its test showed the flaw: the weakest pool member is pinned at 0 and
  can never be selected, which defeats MMR. Similarity is Jaccard by default (no model call) and cosine
  with an `EmbeddingClient`. Signals `prior_rank`, `mmr_relevance`, `mmr_redundancy`, `mmr_score`.
- `RetrievalPipeline(..., diversifier=None, diversify_pool_k=None)`: when set, the reranker returns a
  pool (default `min(2 * final_k, rerank_k)`), a `diversify` stage is traced (`pool_k`, `k`, ids, latency),
  failures degrade to relevance order (`diversify:<Exc>`). Off by default: trace and results are unchanged.
  Settings `RAGKIT_RETRIEVAL_DIVERSITY=none|mmr`, `_MMR_LAMBDA`, `_DIVERSIFY_POOL_K`; exported from
  `ragkit.retrieval`.
- `ragkit/eval/stage_isolation.py`: kind `diversify` is read as part of the precision stage (`rerank`), so a
  required document dropped by MMR is labeled `dropped-by-rerank` instead of being misread as a candidate list.
- Tests: +5 in `test_retrieval_pipeline.py` (near-copy swap, lambda 1 identity, relevance floor, cosine path,
  pipeline stage/pool/trace/degradation, settings), +1 in `test_rag_eval_stage_isolation.py`.
- Ch 12: new core-concepts section "Diversity: maximal marginal relevance" (mechanism; the two funnel
  adaptations; a worked example from the tests in which the crossover between lambda 0.9 and 0.8 decides
  whether the second slot goes to a near-copy or the missing facet; when it helps and hurts; guards; cost; an
  honest corpus result where it is small and mixed on Northwind; relation to the Ch 13 dedupe; cross-references
  to Ch 5 and Ch 8). Also a new implementation section with a listing, the re-pasted `retrieve()` listing, the
  architecture diagram, how-it-works step, config rows, tradeoff row, failure mode "Diversity starves the
  answer", key takeaway, and test count 75. Ch 14: test count 75 and the stage-isolation bullet.

Verification: ragkit 265 passed, p3-rag-assistant 45 passed, p2 50 passed (each run separately; running
ragkit and p3 in one pytest invocation fails at collection because both have a `tests/conftest.py`, which
was already the case and is not related to these edits). Ch 12 listings match disk (14/14). `build_book --check`
now reports only chapter 39 missing.

Open: Ch 5 (another group's chapter) still says "Chapter 12 builds the reranking stage". It could name
`MMRDiversifier`; I left it unedited and noted it in the integration notes.

## Follow-up 2: embedding-space `mmr_select` (lead's requested API)

The lead's second message asked for `ragkit/retrieval/mmr.py` with `mmr_select(query_vec, candidates, k,
lambda_=0.7, min_relevance=None)`. The pipeline stage from follow-up 1 already existed (`diversity.py`,
`RetrievalPipeline(diversifier=...)`, traced `diversify` stage, default off), so this adds the function form
on top of it rather than a second pipeline mechanism:
- `ragkit/retrieval/mmr.py`: `mmr_select(query_vec, candidates, k, lambda_=0.7, min_relevance=None, *,
  vectors=None, embeddings=None)`. Relevance is cosine to the query vector and redundancy is cosine between
  candidates, identical to Ch 8's `embedlab` `mmr`; `min_relevance` is a cosine floor. Vectors come from the
  caller (the dense retriever's) or one batched `EmbeddingClient.embed` call. It shares `mmr_order` with
  `MMRDiversifier` and is exported from `ragkit.retrieval`.
- Test: +1 (`test_mmr_select_in_embedding_space_matches_chapter_8_formula`: facet swap, caller vectors give
  the same picks with no embedding call, floor keeps the unrelated chunk out, error without vectors).
- Ch 12: implementation paragraph on `mmr_select` and when to use it (no reranker, dense-only path) versus
  `MMRDiversifier` (after a reranker); new "Interaction with rerankers" paragraph (run MMR after reranking,
  never before; the reranker's score is the relevance signal; re-tune λ when the reranker changes); layout
  tree; test count 76.

Verification: ragkit 266 passed; p3-rag-assistant 45 passed; Ch 14 integration and stage-isolation tests
(`test_rag_eval_ch12_integration.py`, `test_rag_eval_end_to_end.py`, `test_rag_eval_stage_isolation.py`) 37 passed;
Ch 12 listings 14/14 match; `build_book --check` reports only chapter 39 missing.
