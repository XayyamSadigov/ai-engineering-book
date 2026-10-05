# Review group A: Chapters 1, 2, 3, 33, 34

Scope: chapters 01, 02, 03, 33, 34, their solutions, `book/projects/examples/ch01`, `ch02`, `ch33`,
`ch34`, and `book/projects/aie_core`. Four-lens review (Principal AI Engineer, Senior Software
Architect, AI Educator, Production/SRE), fixes, second pass.

## Gaps found (by reviewer role)

### Principal AI Engineer

- Ch 1: the lineage record could not answer "which index build served the evidence". Its
  "cross-tenant" check tested permission groups only. Northwind's ACL rule is "groups intersect AND
  the doc tenant is `shared` or the caller's", so a doc open to `all` in the other tenant passed the check.
- Ch 1: the gate-bypass invariant ("deny plus non-empty output") contradicted the chapter's own
  worked example. After a deny the system shows a non-empty fallback message, so every correct
  deny was reported as a violation.
- Ch 2: prefix reuse (the mechanism behind provider prompt caching) was not explained. Chapter 3
  relies on it.
- Ch 2: "prefill grows faster than linearly" was stated unconditionally, and the prefill diagram said
  "cost grows ~ n^2". Both are wrong at typical prompt lengths, where MLP work is linear.
- Ch 3: the chapter builds `aie_core.embeddings` but never taught it. `EmbeddingClient`, batching,
  and the `CachedEmbeddings` space fingerprint were absent.
- Ch 3: the chapter told readers to "add a tenant id to the cache key", but the library offered no
  way to do so. `metadata` was excluded from `cache_key` entirely.
- Ch 33: chat APIs return no class confidence, yet the cascade depends on one. The chapter did not
  say where confidence comes from in a real deployment.
- Ch 34: "a self-hosted endpoint is just another LLMClient in its fallback chain" contradicted the
  book's split: capability-changing fallbacks belong in the Ch 7 router, same-model retries in the gateway.
- Ch 34: the 55 GiB and 56 GiB KV budgets were inconsistent between the KV section and the capacity example.

### Senior Software Architect

- `aie_core.ModelGateway` bug: one `asyncio.Semaphore` was created lazily and reused forever.
  Reusing a gateway across `asyncio.run` calls under contention raised
  "bound to a different event loop". This was reproduced against the pre-fix code.
- `aie_core` gap: no fail-closed option for response caching on a multi-tenant gateway.
- Ch 33 code: `run_fine_tune` aborted on a single transient 503 or 429 during an hours-long poll.
  The job kept training and billing at the provider. A retry of the pipeline uploaded the data again
  and started a second paid job. Nothing resumed by job id, and `max_polls` exhaustion was
  indistinguishable from a failed job.
- Ch 33 code: the OpenAI-compatible fine-tune adapter used `httpx.raise_for_status`, outside the
  book's error taxonomy, so callers could not branch on `retryable`.
- Ch 33 and Ch 34 code predated `aie_core` integration. Ch 33 never used `aie_core` embeddings or
  clients. Ch 34's `make_client` did not set `provider` (every span said "openai") and did not pass
  `supports_response_schema`, so `complete_structured` would send `response_format` to an engine
  without schema mode. Its only `aie_core` test was marked integration and always skipped.
- Code in chapters differed from code on disk:
  - Ch 33: 22 listing blocks were hand-retyped with altered signatures and `...` bodies.
  - Ch 2: two excerpts dropped imports and section banners.

### AI Educator

- Ch 1: the book anatomy said "seven cases" for Part X. Chapters 35 and 36 have nine: seven
  applications and two platforms.
- Ch 1: no map of the code base (8 libraries, 6 projects, capstone) and no pointer to the roadmap,
  glossary, references, or appendix.
- Ch 1 and Ch 34: three places referred to "the source material" or "Recipe 12 in the source",
  which the reader does not have.
- Ch 2: Mermaid labels used `<br/>` HTML, against the authoring guide.
- Ch 33: one Mermaid label contained a semicolon, which is unsafe for the parser.
- Ch 33 P1 would have been trivially answered by the new bridge code, so it was reworked.
- All five chapters: exercise ids match their solutions (K1 to K6, E1 to E4, P1 to P4, D1 to D3),
  so no fix was needed.

### Production/SRE

- Ch 3: no reference of what the gateway's `llm.complete` span records or what to derive and alert
  on from it. Missing points:
  - `OTelTracer` exports spans flat, with linkage carried in attributes.
  - Ch 31's tracer must not be mixed with this one.
- Ch 3: the rate limiter is per process. N replicas configured with the full provider quota admit N
  times the limit. This was not stated.
- Ch 3: no failure mode for a cross-tenant cache hit.
- Ch 2: no consolidated "signal to record, alert to set" guidance derived from the mechanisms.
- Ch 33: no failure mode for orphaned or duplicated training jobs.
- Ch 34: no guidance on log-probabilities as a confidence signal, or on recalibrating them after
  engine or quantization changes.

## Changes applied

### `aie_core` (backward compatible; all dependent suites rerun)

- `gateway.cache_key` includes `metadata["cache_scope"]` only when present, so existing unscoped
  keys are unchanged.
- `ModelGateway(require_cache_scope=False)` is a new keyword. When True, unscoped requests are never
  cached, so the gateway fails closed.
- The async semaphore is kept per running event loop (`WeakKeyDictionary`). Async streams release the
  semaphore they acquired, carried on `_OpenStream.async_sem`.
- `OpenAICompatibleClient(extra_body=None)`: server-specific request fields, for example
  `{"logprobs": True}`. They never override a field the adapter sets.
- Tests added: cache-scope partitioning with unchanged unscoped keys, fail-closed scope, the
  event-loop regression (fails on old code), and `extra_body`. aie_core now has 115 tests; the
  count includes tests another reviewer added concurrently. The README documents the cache scope.

### Ch 1

- `lineage.py`:
  - `EvidenceRef.tenant` (default `shared`) with a tenant check in `consistency_violations`.
  - `RequestLineage.index_version`, with a new unanswered question when evidence exists without it.
  - `RequestLineage.output_kind` (`answer`, `fallback`, `refusal`). The gate check now flags
    "deny but the model's answer was shown", so a correct deny followed by a fallback passes.
  - `from_dict` accepts older records.
- Tests went from 6 to 9: tenant leak with matching groups, missing index version, deny plus fallback.
- Chapter text:
  - Lineage questions, walkthrough, security and failure-mode text updated to match the code.
  - Listings re-pasted.
  - Part X now says nine cases.
  - New "Where the code lives" table mapping every library and project to its chapter, with a
    pointer to the companion files.
  - The "source material" reference was removed.
- E3 solution updated for the new fields.

### Ch 2

- Excerpts replaced with exact disk slices.
- HTML removed from Mermaid.
- Prefill-cost wording corrected in the prose and the diagram.
- New paragraph on prefix reuse: exact-token match from position 1, best-effort eviction, links to
  Ch 3 and Ch 34.
- New "Signals that follow from the mechanism" production paragraph: token counts with method,
  `length` share, TTFT and TPOT, cached ratio, KV utilization.
- All quoted numbers were rechecked by running the scripts: tokenizer table, sampling table, KV
  calculator.

### Ch 3

- New concept subsection "Embedding clients and the embedding space": protocol, batching and index
  checks, no-retry rationale, fakes, `CachedEmbeddings.space_fingerprint` (innermost client, frozen,
  `text_prep_version`), and the index-side equivalent in Ch 8 and Ch 9.
- New implementation excerpt for the fingerprint and key.
- Caching text documents `cache_scope` and `require_cache_scope`.
- Concurrency text covers the per-loop semaphore and the fact that the sync and async limits are
  independent.
- Rate-limit text covers the per-process limiter and replica shares.
- New span-attribute table with derived signals and three operational notes.
- New common mistakes: mixing embedding spaces, full quota per replica.
- New failure mode: cross-tenant cache hit.
- Tradeoffs mention `extra_body`.
- Test count updated. The `build_payload` excerpt was re-synced.
- E1 solution now uses `cache_scope` and `require_cache_scope`.
- Span linkage and cache-hit `avoided_cost_usd` were already described accurately and were kept.

### Ch 33

- New `aie_bridge.py`:
  - `embed_fn_from_client` adapts any `aie_core` `EmbeddingClient`, including `CachedEmbeddings`,
    to `dedupe_near`.
  - `embedding_space` returns the provenance to record in the data card.
  - `run_classifier` runs the holdout through any `LLMClient` or `ModelGateway`. It enforces the
    training prompt via the card's `system_prompt_sha256` and takes cost from the gateway's pricing.
    Invalid labels and provider errors become zero-confidence predictions, so a cascade escalates them.
  - `logprob_confidence` reads token log-probabilities from `completion.raw`.
- New `test_aie_bridge.py` with 7 tests.
- `finetune_job.py`:
  - The adapter's HTTP failures map into the `aie_core` taxonomy (`_send`).
  - `run_fine_tune` absorbs retryable poll errors up to `max_consecutive_poll_errors` and honors
    `Retry-After`.
  - `resume_job_id` lets a run resume an existing job instead of starting a second one.
  - The job id is reported to `on_status` as soon as the job exists.
  - `FineTunePollTimeout` is raised when polling gives up, and it never cancels the job.
  - `FakeProvider` records cancels.
  - 3 new tests: transient 503 then 429 with a resumed job and no upload, non-retryable 401,
    poll timeout.
- All three pipeline listings were regenerated from source with an AST-based excerpt tool, so they
  match disk exactly. The bridge is listed in full.
- Implementation intro, walkthrough, and test description rewritten.
- New failure mode "Orphaned or duplicated training job".
- Mermaid semicolon fixed.
- P1 reworked to record embedder provenance, with a matching solution. Its TF-IDF similarity claim
  was checked by running it: 0.49 against a 0.90 threshold.

### Ch 34

- `local_target.py`:
  - `envelope_for(req)` builds the envelope from a `CompletionRequest` with `aie_core`'s token counter.
  - `make_client` sets `provider="engine:target"` and `supports_response_schema` from the target,
    and accepts injectable transports.
- The always-skipped integration test was replaced by two offline tests. One streams through a
  `ModelGateway` into the fake server and checks the span's provider. The other runs
  `complete_structured` against a schema-less target and checks that no `response_format` was sent.
- Text changes:
  - Router versus gateway-fallback rule stated; key takeaway and E1 solution aligned.
  - New log-probabilities paragraph.
  - Implementation, walkthrough, and testing text updated.
  - KV budget unified at 55 GiB in text, `capacity.py` and the test; the computed figure is 0.59 replicas.
  - "Source" references removed.
- Listing re-pasted.

### Ch 30 (consistency with the new aie_core, requested by the lead)

- Text no longer claims `aie_core.CachedEmbeddings` keys on model and raw text only; it states the
  space fingerprint and that `EmbeddingCache` adds normalization on top.
- The response-caching paragraph now acknowledges `cache_scope` / `require_cache_scope` and says
  what `ScopedResponseCache` still adds (structured ACL scope, prompt and index versions, lintable
  key components).
- The `EmbeddingCache` docstring in `caching.py` was corrected the same way; it is not part of a
  chapter listing. Ch 30 tests: 34 passed; Ch 30 listings: 0 mismatches.

### Shared notes

- `07-integration-notes.md`: the stale CachedEmbeddings line was corrected. The aie_core
  `REVIEW TODO` was marked DONE, listing the new APIs.

## Second-pass findings

- After my edits, another agent overwrote Chapter 3 from a stale copy, adding truncation text.
  I reapplied all group A Chapter 3 edits with an idempotent script, kept their truncation text,
  re-synced listings, and told the team lead.
- That agent also changed `aie_core` (`Completion.truncated`, `TruncatedOutputError`,
  `CachedEmbeddings.innermost`, embedding index checks). Chapter 3's listings were rechecked
  against disk afterwards and match. The embedding concept text now mentions index checks and
  the innermost client.
- Listing check (every `# path:` block compared with disk, excerpts compared segment by segment):
  0 mismatches in all five chapters.
- Style check: no em dashes outside titles and no HTML in Mermaid in the five chapters or their
  solutions. Cross-references were spot-checked against the TOC.
- The Ch 34 E1 solution still described a hosted endpoint as a gateway fallback for `retail`.
  I rewrote it to route via the router with per-target gateways.

## Remaining open items

- `build_book.py --check` fails only because Chapter 15 and Chapter 39 do not exist yet. Their
  authors are still writing them. Nothing in group A blocks the check.
- Concurrent edits to `aie_core` and Chapter 3 by another reviewer. If they write Chapter 3 again
  from a stale copy, rerun the reapply script, which is kept in my scratchpad and is idempotent,
  or diff against this list. The team lead has been notified.
- `aie_core` embeddings still have no retry. This is deliberate and documented in Ch 3: indexing
  jobs wrap them with Ch 29 primitives. Adding retries to the embedding client would duplicate the
  gateway and was out of scope.
- `run_classifier` evaluates sequentially. Async fan-out with the gateway semaphore would speed
  large holdouts but is not needed for correctness, so I left it as a natural extension.

## Follow-up: capstone API mismatches #5 and #6 (aie_core, backward compatible)

- #5: new public `aie_core.make_provider_client(settings=None, model=None)` returns the bare
  provider adapter; `_raw_client` remains for existing callers. `make_llm_client` gained
  keyword-only `tracer=` (replaces the tracer built from `TRACE_SINK`, so the capstone can pass its
  one Ch 31 tracer into the gateway) and `wrap=` (`None` keeps the old behavior, `True` wraps the
  fake provider in a gateway for offline tests of retries, caching and spans, `False` returns the
  bare adapter).
- #6: `Settings.trace_sink` accepts `"memory"`, which builds an `InMemoryTracer` (read spans from
  the client's `.tracer`); an injected tracer is the other route.
- Tests: 3 new tests in `test_settings.py` (bare adapter and model override, injected tracer with a
  wrapped fake, memory sink from the environment). aie_core: 118 passed.
- Docs: Ch 3 factory excerpt re-synced with disk (listing check 0 mismatches), env table lists
  `memory`, code walkthrough explains `tracer=`, `wrap=` and `make_provider_client`, test count
  updated; aie_core README updated.
- Not changed: the capstone still imports `aie_core.settings._raw_client`; it keeps working, and
  the capstone owner can switch to `make_provider_client` and drop mismatches #5 and #6 from its README.
