# Review group G: Chapters 28-32

Chapters: 28 AI Application Architecture, 29 Reliability and Scalability, 30 Performance and Cost
Engineering, 31 Observability for AI Systems, 32 Engineering Practices for AI Systems. Code:
`book/projects/examples/ch28`, `book/projects/reliability`, `book/projects/examples/ch30`,
`book/projects/examples/ch31`, `book/projects/examples/ch32`. (The brief's `examples/chNN` paths live
under `book/projects/examples/`.)

## Gaps found (by reviewer role)

### Principal AI Engineer

- Ch 28: the stage plan summed to 8.2 s (0.1 + 1.5 + 0.8 + 5.5 + 0.3) while the text said it summed to the
  8 s target. K2 and its solution repeated the 5.5 s figure.
- Ch 28: stage timeouts and p95 targets were conflated. A plan that lets retrieval and rerank take
  2.4 s cannot be what meets a 2 s time-to-first-token target, and the text never said so.
- Ch 28: ports were described as if the book's packages dropped in. They do not (REVIEW TODO): no mapping
  from `ModelPort`, `RetrieverPort`, `PromptRegistryPort`, `RetrievalCachePort`, and the job ports to
  `aie_core`, `ragkit`, Ch 4 prompts, Ch 30 caches, `reliability`; no mention of `toolkit`, `agentkit`,
  `guardrails`, or the Ch 31 tracer.
- Ch 29: no hedged requests, although Ch 30 says "Chapter 29 covers them" (broken promise).
- Ch 29: no treatment of failures after a stream's first token at system level (no retry or fallback
  is possible once text reached the user).
- Ch 29: burn-rate alerting described without numbers; it said Ch 31 "turns it into alerts", which Ch 31
  did not do.
- Ch 30: the cache-hit cost text described the old `aie_core` behavior (hit spans carry the original
  cost). The current gateway records `cost_usd=0` and `avoided_cost_usd`. Retrieval, embedding, and index
  costs were one line ("infra amortized") with no worked number, so the reader could not see that
  amortized index cost overtakes model cost once the model bill is optimized.
- Ch 31: no SLO burn-rate metrics or multi-window rules, despite Ch 29 deferring to it. Otherwise
  coverage was complete: traces, prompts, responses, tool calls, retrieval results, tokens, latency,
  cost, eval results, agent trajectories, quality-degradation playbook.
- Ch 32: no gaps in must-cover items. One awkward sentence in D3 ("equals the baseline's except for
  nothing").

### Senior Software Architect

- Ch 28 (security): the stub auth read the tenant from a client `X-Tenant-Id` header, contradicting the
  chapter's own rule that the tenant originates in the credential. Ch 28 also claimed "Chapter 27
  implements JWT validation" (false; the capstone does).
- Ch 28 (code differs from text): the text said auth stores an absolute deadline in `RequestContext`;
  the code creates `Budget` inside `ChatService` and `RequestContext` has no deadline.
- Ch 28 / Ch 31 (security): the Ch 28 collector deleted `llm.prompt.text`, a key nothing writes. Ch 31's
  capture policy writes `<key>.content` (`llm.prompt.content`, `llm.completion.content`,
  `tool.args.content`, `tool.result.content`, `retrieval.query.content`). The "second line of defense" that
  Ch 31 claimed did not exist.
- Ch 28 / Ch 29: Ch 28's `JobQueuePort` (bare id `enqueue`/`dequeue`) is not a drop-in for
  `reliability.JobQueue` (lease/ack/nack). The text promised that the Redis queue would pass the
  skeleton's contract test.
- Ch 30 (bug): `chargeback` computed avoided cost from `cost_usd`, which is 0 on current hit spans, so
  avoided spend was 0 unless a pricing table was passed. The test passed only because it passed one.
- Ch 32 (security): `PromptVersion.render` stripped the delimiter with a single `replace("</ticket>",
  "")`, so `</tic</ticket>ket>` became `</ticket>` and case or whitespace variants passed untouched. The
  chapter's own checklist demands delimiters that input cannot close.
- Ch 32 (test integrity): the snapshot test wrote a missing snapshot and then passed, so a new prompt
  version with no reviewed snapshot passed CI.
- Ch 32 / Ch 31 (consistency): Ch 32 put versions on spans only as `version.*`, while Ch 31's tooling
  (`compare_versions`, triage, alerts) groups by `prompt.version`, `index.version`, `llm.model`. Ch 31
  traces and Ch 32 traces could not be analyzed with the same code. Ch 32 said "Chapter 31 owns the span
  schema" and then deviated.
- Ch 32 (CI): a canary `hold` (exit 3) fails the job and triggers rollback in both pipelines; this was
  undocumented and contradicted the text's "insufficient data holds".
- Listings: Ch 28's `docker-compose.yml` listing claimed to be the full file but lacked the file's
  header; several excerpts in Ch 28, 29, 31 were condensed (lines merged, comments added) without saying
  so.

### AI Educator

- Broken cross-references: Ch 30 to Ch 29 on hedging; Ch 29 to Ch 31 on burn-rate alerts; Ch 28 to
  Ch 27 on JWT; Ch 31 to Ch 28's collector behavior.
- Ch 28 used "the Redis queue lives in Ch 29" without telling the reader the interfaces differ, which
  would stall anyone who tried the swap.
- Exercises: all five chapters have K/E/P/D categories, debugging items present a broken system or
  trace, no answers leak into exercises, and solution ids match. No change needed beyond the K2 numbers.
- Mental-model callouts are used, not decorative. Learning order inside each chapter is sound.

### Production/SRE Engineer

- Ch 29: alerts were implied by a dashboard list but not stated (what pages, what tickets).
- Ch 29 / Ch 31: no burn-rate paging (above).
- Ch 28: no operational answer for a job dead-lettered by lease expiry while the jobs table says
  `running` (appears once jobs run on Ch 29's queue).
- Ch 30: fixed retrieval infrastructure missing from the per-task cost reasoning (above).

## Changes applied

### Chapter 28 and `book/projects/examples/ch28`

- `api_skeleton.py`: plan `model_total` 5.5 to 5.3 s, so the plan sums to 8.0 s; docstring states that
  the figures are timeouts, not p95 targets, and points to `reliability.Deadline` for cross-process
  propagation.
- `api_skeleton.py`: stub auth now reads `user@tenant:groups` from the bearer token only; `X-Tenant-Id`
  is no longer read. Tests updated; new `test_token_without_tenant_is_401` and
  `test_tenant_comes_from_the_token_not_from_headers` (spoofed header gets retail evidence).
- New `reliability_bridge.py`: `ReliabilityJobQueue` (submission half of `JobQueuePort`, refuses bare
  `dequeue`), `bridge_handler` (mirrors each attempt into the jobs table, honors cancellation, re-raises
  so the Ch 29 worker classifies), `make_worker`, `reconcile_dead_letters` (fixes rows left `running`
  after a crash on the final attempt). New `test_bridge.py`, 6 tests, skipped when `reliability` is not
  installed.
- `otel-collector.yaml`: deletes every attribute matching `^.*\.content$`, which covers all Ch 31
  capture keys.
- Chapter text: JWT sentence fixed (capstone Ch 39; Ch 27 supplies tenant-scoped keys and guardrails);
  plan arithmetic stated; "deadline created once, then carried" rewritten to match the code; new
  "Timeouts are not targets" paragraph with an illustrative p95 breakdown (first token about 1.5 s,
  completion near 6 s); tenant-origin rule tightened and tied to the new test; new subsection "Swapping
  the skeleton's adapters for the book's packages" with a port-to-package table and the bridge
  listings; Redis-queue sentences and the contract-test paragraph reconciled; file list, test counts
  (23), curl command, Budget listing, K2 updated; compose listing re-pasted from disk; condensed excerpts
  labeled.
- Solutions: K2 rewritten for the 5.3 s plan with the timeout-versus-target point; P1 count fixed.

### Chapter 29 and `book/projects/reliability`

- New `reliability.ahedged` (exported from the package): hedged requests for idempotent reads, delay
  near p95, losers cancelled, deadline respected, hedges spend from the shared `RetryBudget` so hedging
  turns itself off under overload. New `tests/test_hedge.py`, 5 tests. Backward compatible.
- Chapter text: new "Hedged requests" subsection (what, when, three hard conditions, worked tail
  arithmetic labeled illustrative, budget interaction, telemetry, listing); stream commit point at
  system level (what to do after the first token, and counting mid-stream failures as an SLI
  component); burn-rate numbers (14.4 over 1 h and 5 min, 6 over 6 h and 30 min, matching
  `should_page`); Ch 28 port reconciliation in the queues section; explicit page and ticket alerts in
  Operations; test counts (99); hedging added to a key takeaway.

### Chapter 30 and `book/projects/examples/ch30`

- `cost.py` `chargeback`: cache-hit spans use `avoided_cost_usd`, falling back to `cost_usd` for legacy
  spans. `test_ch30.py` now also runs chargeback without a pricing table, which fails on the old code.
- `attribution.py` docstring no longer claims `aie_core` spans are flat.
- Chapter text: cache-hit paragraph rewritten for current `aie_core`; new subsection "Retrieval,
  embedding, and index costs" (query versus corpus embedding cost, re-embed overlap as the real cost,
  reranker cost as candidates times tokens with a recall-driven depth, amortized index cost of about
  0.0033 USD per answer against 0.0048 USD per optimized attempt, all illustrative, with Ch 9 and Ch 15
  cross-references); listings and walkthrough updated.

### Chapter 31 and `book/projects/examples/ch31`

- `metrics.py`: SLO constants mirroring `reliability.slo.SLOTargets`; metrics
  `slo.availability_burn_rate` and `slo.completion_burn_rate`; `AlertRule.confirm_window` (multi-window
  rule: fires only if the short window also breaches; validated to need an absolute threshold and a
  shorter window).
- `alerts.yaml`: `availability_fast_burn` (14.4, 1 h confirmed by 5 min, page) and
  `completion_slow_burn` (6, 6 h confirmed by 30 min, ticket), with the arithmetic in comments.
- `dashboards.md`: SLO burn rows. New `tests/test_slo_alerts.py`, 3 tests. The incident walkthrough's
  output is unchanged except that the two new rules appear in the quiet list.
- Chapter text: collector claim corrected; architecture diagram label; new burn-rate paragraph with
  listing and the "loose objectives cannot page fast" arithmetic; walkthrough quiet list; test counts
  (47); condensed excerpts labeled.

### Chapter 32 and `book/projects/examples/ch32`

- `ports.py`: `strip_delimiters` removes opening and closing ticket tags case-insensitively, repeating
  until stable; `render` uses it. Full listing re-pasted.
- `tests/test_prompt_snapshots.py`: a missing snapshot fails instead of being written; new hypothesis
  test `test_no_input_can_forge_a_ticket_tag` over tag fragments.
- `version_manifest.py`: `as_semconv_attributes()` emits Ch 31 names; `triage_service.py` writes both
  sets on `triage.request`; manifest test extended.
- CI files: comments stating that a canary hold rolls back; both full listings re-pasted.
- Chapter text: versioning paragraph explains the two attribute sets; snapshot listing and a paragraph
  on why missing snapshots fail and what the property test caught; review-checklist delimiter item
  sharpened; practice-to-test table updated; test count (86); D3 wording fixed.

### Integration notes

Appended DONE lines and new public names to `book/analysis/07-integration-notes.md`
(`reliability_bridge`, `ahedged`, chargeback rule, SLO metrics and `confirm_window`, collector pattern,
`as_semconv_attributes`, `strip_delimiters`).

### Test results

```
book/projects/examples/ch28   23 passed
book/projects/reliability     99 passed, 10 deselected (real-Redis integration)
book/projects/examples/ch30   34 passed
book/projects/examples/ch31   47 passed
book/projects/examples/ch32   86 passed, 1 deselected (integration re-record)
build_book.py --check         only "chapter 15 missing", "chapter 39 missing" (outside group G)
```

## Second-pass findings

- All four roles re-run on the edited chapters. The coverage checklist from the lead now holds:
  retries/backoff/fallbacks/circuit breakers/malformed responses/outages/partial failures (Ch 29, plus
  stream commit point); concurrency/queues/async/workers/rate limits/load management (Ch 29, Ch 28 job
  model and bridge); latency/token budgets/caching/streaming/parallelization (Ch 30, Ch 28 budgets,
  Ch 29 hedging); cost engineering incl. tokens, requests, embeddings, retrieval, caching, routing,
  monitoring (Ch 30, with the new retrieval-cost subsection); observability of traces, prompts,
  responses, tool calls, retrieval, tokens, latency, cost, eval results, trajectories, quality
  debugging, and now SLO burn alerts (Ch 31).
- Cross-references re-checked: Ch 30 to Ch 29 (hedging) and Ch 29 to Ch 31 (burn-rate alerts) now
  resolve; Ch 28 to Ch 39 matches the TOC brief (JWT plus tenants).
- No em dashes outside chapter titles in the five chapters or their solutions.
- Listing check: every full-file listing equals its file; every excerpt either matches the file line by
  line (whitespace-normalized) or is labeled "condensed excerpt", "reformatted", or "shortened".
- Minor items found and fixed in the second pass: the Ch 28 compose listing header, condensed-excerpt
  labels.

## Remaining open items

- ruff and mypy are not installed in `.venv`, so the Ch 32 pipeline's lint and type-check stages could
  not be run locally on the edited Ch 32 files (tests pass). The new code is fully typed.
- Real infrastructure not verified: `RedisJobQueue` through the bridge was tested only with the
  in-memory queue (the bridge is backend-agnostic and the queue contract suite covers fakeredis); the
  collector's `pattern` delete was not run against a live OpenTelemetry collector (no Docker access in
  this review).
- Docker image tags in Ch 28 compose (`otel/opentelemetry-collector-contrib:latest`) are unpinned. Left
  as is, because naming a specific release would state a vendor fact that cannot be verified offline;
  instead Ch 28's Operations paragraph now says that production manifests pin images by version or
  digest.
- `book/projects/examples/ch28/.env.example` could not be read (a permission rule blocks `.env*`
  reads), so it was not checked against the new token format; it holds provider and URL settings, not
  credentials format, per the chapter's variable table.

## Follow-up: capstone mismatch 4 (AITracer streaming baggage)

Reported by the capstone integration (`book/capstone/northwind-assist/README.md`, "API mismatches"
item 4). `AITracer.span()` stamped propagated lineage keys (`tenant.id`, `prompt.*`, version manifest,
`app.route`) from baggage onto `llm.complete` spans, but `AITracer.export()` did not. `export()` is the
path the gateway's streaming spans take, because `ModelGateway.stream` builds its span by hand and
exports it when the stream ends. Streamed attempts therefore joined the trace tree but could not be
grouped by tenant, prompt version, or index version.

Changes:
- `book/projects/examples/ch31/instrument.py`: the propagation loop moved into
  `AITracer._propagate(span)`, which both `span()` and `export()` now call. Behavior of `span()` is
  unchanged.
- `tests/test_instrument.py`: new `test_streamed_attempt_gets_the_same_propagated_attributes`, which
  drives `ModelGateway.stream` inside `trace_request` and a `prompt.call` span and asserts tenant,
  prompt id, prompt version, and index version on the adopted `llm.complete` span. Verified that it
  fails with the `export()` fix removed and passes with it.
- Chapter 31: the `span()` listing shows the `_propagate` call, and a new paragraph explains the
  streaming path, the fix, the test, and the remaining limit: a stream drained after its parent span
  closed is adopted with no parent and no baggage. Test count updated to 48.

Tests:

```
book/projects/examples/ch31        48 passed
book/projects/examples/ch30        34 passed
book/projects/examples/ch32        86 passed, 1 deselected
book/capstone/northwind-assist     62 passed
```

Not changed: the capstone README still lists item 4. The capstone owner should mark it fixed, since
the capstone has no workaround code to remove.
