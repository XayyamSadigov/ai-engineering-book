# Review group B: Chapters 4-8

Scope: Ch 4 Prompt Engineering as Engineering, Ch 5 Context Engineering, Ch 6 Structured Output and
Extraction (Project 1, `book/projects/p1-extraction-api`), Ch 7 Model Selection and Routing, Ch 8
Embeddings; code in `book/projects/examples/ch04`, `ch05`, `ch07`, `ch08`.

Each chapter section below follows the four required headings: "Gaps found (by reviewer role)",
"Changes applied", "Second-pass findings", "Remaining open items". A group-level summary comes first.

## Group summary

### Gaps found (by reviewer role)

All five chapters already covered every must-cover item in their TOC briefs. The four focus checks
from the lead pass:

- Prompt engineering is taught as engineering. Ch 4 covers instruction hierarchy, a versioned prompt
  registry, golden cases, assertions, a judge rubric, CI regression, and when prompting is not enough.
  The gaps were operational: canary serving, conflict resolution inside the hierarchy, and observability.
- Context engineering is a major, complete chapter. Ch 5 covers selection, compression, ordering,
  prioritization, lost-in-the-middle, memory pointers, conversation state, ephemeral vs persistent
  context, and ablation. The gaps were long-context limits as a separate topic and the persistence and
  concurrency of conversation state.
- Structured output covers repair strategies and deterministic post-processing. The gaps were in
  Project 1's security and reliability: no auth, tenant leakage in the review queue, no deadlines, no
  request size limit, and wrong HTTP semantics for non-retryable errors.
- Model selection is vendor-independent. The gaps were router correctness (silent loss of the quality
  bar, undetected gateway fallback), a migration procedure for retired pins, and routing alerts.
- Embeddings cover all six non-RAG uses with implementations and tests. The gaps were retries,
  deletion, drift blind spots, cross-tenant cache leakage, and anomaly threshold calibration.

### Changes applied

Each chapter gained an observability table with alert signals, a degraded-mode paragraph or runbook,
new failure-mode rows, and a new debugging exercise D4 that presents a broken trace, with its solution.
Code fixes are listed per chapter below. All changes are backward compatible, and listings are
re-pasted wherever code changed.

Verification after all edits, run from the repo root:

| Suite | Result |
|---|---|
| book/projects/examples/ch04 | 64 passed |
| book/projects/examples/ch05 | 38 passed |
| book/projects/examples/ch07 | 35 passed |
| book/projects/examples/ch08 | 41 passed, 1 deselected (integration) |
| book/projects/p1-extraction-api | 83 passed |
| build_book.py --check | no problems in Ch 4-8; exits 1 only because Ch 15 and Ch 39 files are missing (other groups) |

### Second-pass findings (group level)

- Every `# path:` listing in Ch 4-8 is byte-identical to its file on disk: 27 listings checked by
  script, 0 diffs. Ch 8's `corpus.py` block is labeled as a signature summary, not a full listing.
- Exercise ids in all five chapters (K1-K6, E1-E4, P1-P4, D1-D4) match the solutions files one to one.
- There are no em dashes in prose. The only ones are in chapter titles.
- Cross-references to chapter numbers and titles match the TOC.
- The Ch 7 migration step is consistent with the Ch 4 rule: never change the model and the prompt in
  one rollout.
- Each chapter now has four debugging exercises, one more than the guide's 2-3 target. They are kept
  because each new D4 is the only exercise covering its topic (canary or trace, state race,
  indirect-injected value, hidden fallback, embedding drift).

### Remaining open items (group level)

- **Integration notes need new public names.** `07-integration-notes.md` is lead-owned, so these lines
  were not added. Proposed lines:
  - Ch 4: `prompts.rollout` gains `PromptRollout`, `ReloadResult`, and `bucket`. Sticky hash
    canary; a bad alias push keeps the last good registry.
  - Ch 5: `ConversationState` gains `version`, idempotent `add_turn` keyed by `message_id`,
    `StateSnapshot`, and `InMemoryStateStore` with compare-and-set.
  - Ch 6: see the suggested line in the Ch 6 section. It covers `api/auth.py`, tenant-scoped
    `ReviewQueue.list`, per-document deadlines, 502 for non-retryable errors, and `BatchItem.retryable`.
- **Shared-package changes were not made, by rule.**
  - `aie_core.CachedEmbeddings` fingerprints the wrapper class name instead of the real provider.
  - `embed` returns a short result silently when a provider response is partial.
  - There is no embedding gateway.
  - Truncated-response detection for Project 1 needs `aie_core` support.
- **MMR is described in Ch 5 but not implemented anywhere.** It is missing from Ch 12 and from
  `ragkit.retrieval`. Ch 5's wording was adjusted so it does not claim otherwise. Flagged for the Ch 12
  reviewer.
- **Length.** Ch 4 prose is about 9.6k words against a 7-8k target. Cut it first if the book needs
  trimming.

---

## Ch 4 — Prompt Engineering as Engineering

### Gaps found (by reviewer role)

Baseline: the chapter was already strong. It covers every must-cover item in the TOC brief (contract, instruction hierarchy, anatomy, few-shot costs, chaining, reasoning artifacts, sandboxed templates with escaping, registry with hashes and lock, regression harness with judge and gate, when prompting is insufficient). All code listings matched disk, the 57 tests passed, and the CLI output pasted in the chapter reproduced exactly. Exercise ids K1-K6, E1-E4, P1-P4, D1-D3 matched the solutions, no answers leaked, and there were no em dashes in prose.

Principal AI Engineer
- The instruction hierarchy section explained the levels but not how to handle conflicts. It gave no explicit conflict-resolution sentences paired with tests, and it did not separate conflicts across levels from collisions within one level.
- "When prompting is insufficient" listed four signals but no repeatable procedure for telling a prompt problem from a non-prompt problem, so the reader could not decide with evidence.

Senior Software Architect
- The chapter described canary rollout through aliases but had no mechanism for which requests get the canary. It did not cover sticky assignment or comparable arms.
- Runtime alias reload was mentioned conditionally, with no defined behavior when a pushed aliases.toml or prompt file is broken. The startup-versus-reload failure policy was undefined.
- tracing.py and the chapter said "Chapter 31 adds parent/child propagation; one flat span". That was outdated, because aie_core now links spans by trace and parent id.

AI Educator
- The debugging exercises described symptoms but none presented an actual trace for the reader to diagnose, as the brief requires.
- Cross-references all matched the TOC ownership map.

Production/SRE Engineer
- There was no consolidated "what to measure and alert on" table. Metrics were scattered across the Operations paragraph and Failure modes.
- The `prompt.call` span did not record `cached_input_tokens`, yet D1's solution and the "prefix cache collapse" failure mode both depend on that signal.
- Degraded modes were not stated: canary kill switch, last known good registry, and model fallback interacting with prompt versions.

### Changes applied

- **New module** `book/projects/examples/ch04/prompts/rollout.py` provides `PromptRollout` and `bucket`. Assignment is sticky, hashed on prompt id and unit key with the prompt id as salt. Reloads swap only after loading, alias validation, and lock verification all pass, otherwise the last known good registry keeps serving with `reload_failures`/`last_error`. Startup fails fast. The module is exported from `prompts/__init__.py`.
- **New tests** in `tests/test_rollout.py` (7 tests): stickiness and fraction, salting, no-canary-means-prod, broken alias push keeps last known good, good push picked up, edited published version refused on reload, fail-fast startup.
- **Tracing**: `prompts/tracing.py` now records `cached_input_tokens` on `prompt.call`, and its docstring was corrected about span linkage. `tests/test_tracing.py` asserts the cached-token attribute and that `llm.complete` is a child of `prompt.call` in the same trace. The full listing was re-pasted in the chapter, and the prose after it was rewritten so retries and fallbacks show under the prompt version.
- **Chapter, instruction hierarchy**: added a paragraph on writing conflict resolution explicitly. It gives three typical conflicts with their resolution sentences, requires one golden case per conflict rule, and separates same-level rule collisions as contract defects.
- **Chapter, when prompting is insufficient**: added an error-analysis procedure with a cause/symptom/remedy table. Each row says whether a prompt edit helps. An illustrative worked split shows that a perfect prompt edit can recover at most 7 of 30 failures.
- **Chapter, Implementation**: added the subsection "Serving versions at runtime" with the full `rollout.py` listing and wiring guidance. Wiring means a watcher or periodic reload, with the failure metric exported and alerted. The directory tree and test counts were updated from 57 to 64.
- **Chapter, Production considerations**: added an Observability table with nine signals, each giving source, alert condition, and likely cause, with thresholds labeled illustrative. Two signals are new: spans with an unknown `prompt.hash` and gateway calls without a `prompt.call` parent, which catches string-literal prompts. Also added a degraded-modes paragraph, and the Operations paragraph now references sticky assignment.
- **Exercises**: added D4, a trace-based debugging exercise. A subject variable was declared `trusted = true` in a "wording cleanup", and the injection sits outside the data block. The reader must explain why the lock, taint check, and suite all passed. The full solution in `solutions/ch04-solutions.md` covers root cause from the trace, why each control passed, a trusted-variable allowlist test, a code-owner rule, a critical subject-injection probe, and the telemetry.
- **README**: the tree now lists `rollout.py`.

### Second-pass findings

- All listings were re-verified against disk: full files are byte-identical, and every excerpt fragment is present.
- `prompts_cli.py verify` and both `compare` runs still reproduce the chapter's output, since no prompt files changed and the lock is unchanged.
- Cross-references still match the TOC. There are no em dashes in prose, and new numbers are labeled illustrative.
- Prose is now about 9.6k words against the 7-8k brief target. This is acceptable for an implementation-heavy chapter, but it is the main candidate for trimming if the book runs long. No repetition with other chapters was found.

### Remaining open items

- `book/analysis/07-integration-notes.md` should list the new public names `PromptRollout`, `ReloadResult`, `bucket` under the Ch 4 entry. The lead owns that file, so I did not edit it.
- `build_book.py --check` exits 1 only because chapters 15 and 39 are missing. Their owners are outside group B. Chapter 4 itself reports no problems.

---

## Ch 5 — Context Engineering

Files reviewed: `book/chapters/05-context-engineering.md`, `book/solutions/ch05-solutions.md`, `book/projects/examples/ch05/` (context package, demo, 3 test files). Starting state: strong chapter, all 8 full listings byte-identical to disk, 35 tests passing, exercise ids K1-K6/E1-E4/P1-P4/D1-D3 matching solutions, no em dashes in prose, cross-references (Ch 2, 4, 6, 7, 10-17, 21, 24-27, 30, 31, 33, 34) correct against the TOC. Worked numbers verified (latency ceiling 12,800 tokens; agent replay 465,000 vs 210,000 tokens; E1 solution 1.14 M tokens).

### Gaps found (by reviewer role)

**Principal AI Engineer**
- Long-context limits were covered only as "soft limit" plus lost-in-the-middle. Missing: single-needle tests overstate usable length, multi-fact aggregation degrades earlier, distractor *similarity* matters more than count, instruction decay over distance, the latency ceiling as the deciding constraint, and the architecture-change signal (pointer to Ch 37's long-context-vs-RAG comparison). The chapter's own position harness is single-needle and did not say so.
- Selection stopped at threshold + Jaccard dedupe. No graded diversity (MMR), so multi-facet questions could have the evidence cap filled by near-copies.
- Compression was presented as "summarize old turns, trim tool output". No ladder of techniques (structural trim, extractive spans, abstractive summary, learned token-level compression) with their distinct risks.

**Senior Software Architect**
- `ConversationState` is persistent context but had no persistence contract: no serialization, no version, no concurrency control. Two real failures were unaddressed: duplicate turns on client retry (no idempotency key) and lost updates when a background compaction and a request write the same session (last-writer-wins silently deletes a turn from the "append-only" log).
- Run instructions said "from the book root" while the paths are relative to the repository root.

**AI Educator**
- No exercise covered state persistence or concurrency (the new material). Exercise categories otherwise complete and answers not leaking.
- The sequence diagram omitted where state is loaded and saved, which is where the concurrency bug lives.

**Production/SRE Engineer**
- Observability was one "Operations" paragraph listing three alerts. No signal/source/alert table even though the `context.build` span already emits the attributes.
- No degraded modes: what to do when the summarizer is down, retrieval times out, or pinned content overflows.
- Failure-mode catalogue had no entry for lost or duplicated turns.

### Changes applied

Code (`book/projects/examples/ch05/`):
- `context/state.py`: `Turn.message_id`; `add_turn(..., message_id=)` is idempotent (returns the existing turn on a repeat); `ConversationState.version` bumped on every write (`add_turn`, `remember`, `forget`, accepted `compact`/`rebuild_summary`) and `loaded_version`; `snapshot()` / `from_snapshot()` with new pydantic `StateSnapshot`; new `StaleStateError` and `InMemoryStateStore` (load/save with compare-and-set on version under a lock; docstring shows the SQL `WHERE version = :expected` equivalent). All existing names and signatures unchanged (new parameters are keyword-only with defaults).
- `context/__init__.py`: exports `StateSnapshot`, `StaleStateError`, `InMemoryStateStore`.
- `tests/test_ch05_state.py`: 3 new tests (snapshot round trip renders identical items; retried turn with the same message id is not appended; stale concurrent save raises and the newer state survives).
- `README.md`: tree line for the store.

Chapter:
- New paragraph on selection beyond thresholds: MMR formula, λ range labeled illustrative, when to use and when not, placement after reranking (Ch 12), worded so it does not claim Ch 12 implements MMR.
- New subsection "Long-context limits" (after ordering): finding vs using, distractor similarity, instruction decay, worked latency number (100,000-token prompt ≈ 12.5 s prefill at the chapter's illustrative rate, about six times the TTFT target), rule "evidence cap comes from the length sweep; needing more is an architecture change", pointer to Ch 37. States explicitly that the chapter's harness is single-needle.
- Compression ladder (four rungs with risk and test for each) in "Compression and compaction".
- Persistence and concurrency paragraph in "Conversation state and history management" (idempotent appends, compare-and-set, retry rule per writer).
- Sequence diagram now shows load at version v and conditional save.
- `state.py` listing re-pasted (verified byte-identical with a listing/disk comparison script); test count updated to 38; run instructions say "repository root"; directory tree updated.
- Code walkthrough: new "State is saved with compare-and-set" paragraph.
- Production considerations: new **Observability** table (8 signals with span-attribute source and illustrative alert thresholds) and **Degraded modes** paragraph (summarizer outage, retrieval timeout, pinned overflow ladder: force compaction, larger-window route, then fail; `context.degraded` span attribute).
- Failure modes: new "Lost or duplicated turns" entry with telemetry and test.
- Exercises: new **D4** (lost update between a request and a background compaction, presented as a trace); solution added to `ch05-solutions.md` with root cause, telemetry clue (two saves with the same version number), fix, and a stronger structural alternative (insert-only turn table).
- Key takeaways: one bullet on persisted state (now 10 bullets).

Verification: `.venv/bin/python -m pytest book/projects/examples/ch05 -q -p no:cacheprovider` -> `38 passed in 0.58s`; `demo.py` runs; all chapter listings identical to disk (full files) or contained in disk (excerpts); no em dashes outside titles. `build_book.py --check` exits 1 only with "chapter 15 missing" and "chapter 39 missing" (other groups' files); nothing reported for Ch 5.

### Second-pass findings

- Re-checked new numbers (12.5 s prefill = 100,000 / 8,000; ≈ 6x a 2 s target) and that every new code name in prose exists on disk. Consistent.
- Verified that the new D4 is a diagnosis exercise (trace given, cause not stated in the prompt) and that its solution id matches.
- Chapter is now about 16,600 words including about 1,700 lines of code listings; prose is within target. No further cuts needed: the new material does not repeat Ch 2 (mechanism of lost-in-the-middle) or Ch 37 (long-context cost calculator), it cross-references them.
- Debugging exercises now number four (guide aims at 2-3). Kept, because D4 is the only exercise covering state persistence.

### Remaining open items

- `build_book.py --check` fails book-wide until chapters 15 and 39 exist (outside group B).
- MMR is described, not implemented. Neither Ch 12 nor `ragkit.retrieval` contains MMR (checked by grep). Ch 5 now says diversity selection belongs after reranking without claiming Ch 12 implements it. Suggestion for the Ch 12 reviewer: add an MMR step to `ragkit.retrieval`.
- `InMemoryStateStore` is a reference implementation; a SQL-backed session store is left to Project 3 / the capstone (Ch 15, Ch 39), which should keep the compare-and-set contract.

---

## Ch 6 — Structured Output and Extraction (Project 1)

Files reviewed: `book/chapters/06-structured-output-and-extraction.md`, `book/solutions/ch06-solutions.md`, all of `book/projects/p1-extraction-api/` (code, tests, README, env.example, Dockerfile). Baseline: 71 tests passing; every chapter listing matched disk (checked with a listing-vs-disk script: full listings byte-identical, excerpts block-by-block). Baseline content was strong: all brief must-cover items present and developed (four structuring mechanisms incl. grammar-constrained decoding, schema design, validation stack, five repair strategies plus "stop and ask a person", deterministic post-processing, classify-extract-validate-route, calibration with ECE and `choose_threshold`, entity extraction, batch economics with worked illustrative numbers, per-field evaluation). Worked numbers re-verified (cost per document, review labor, Little's law throughput and TPM). The `run_eval` output printed in the chapter was re-run and matches exactly.

### Gaps found (by reviewer role)

**Principal AI Engineer**
- Overclaim: self-consistency called "the best-calibrated signal available through any API". It can be confidently and consistently wrong.
- Input modality not addressed: nothing on scanned PDFs, OCR, or vision-model input and how each changes the evidence (quote) layer.
- Failure-mode table said "assert detection" for truncated output, but Project 1 does not detect truncation (left as exercise P2 without saying so).
- Grounding gate taught as proof of correctness, with no discussion that a quote proves presence, not authority. A value planted in the document (indirect injection) passes every gate.

**Senior Software Architect**
- No authentication or authorization on any endpoint. The review endpoints exposed full document text (PII, finances) to anyone. The chapter only said "restrict who can list it".
- Tenant leakage: `DocumentIn.tenant` was caller-asserted and never stored on review items. `GET /review` returned every tenant's items, which contradicts the book's zero-cross-tenant-leakage target.
- Reviewer identity in the audit trail was self-asserted (the `reviewer` body field).
- Every `LLMError` mapped to 503 "retry later", including non-retryable `InvalidRequestError` and `ContentFilterError`. That turns a deploy bug (rejected schema) into a client retry storm.
- No per-document deadline: worst case about 7 model calls, each with gateway retries and a 60 s default timeout. One document could hold a worker for many minutes. `CompletionRequest.timeout_s` was never set.
- No request-body size limit before parsing. The 50k-char check ran only after pydantic had parsed the whole body.
- Batch `error` entries had no retryable signal (a `DocumentTooLarge` item looked the same as a rate limit).

**AI Educator**
- Exercises well formed (K1-K6, E1-E4, P1-P4, D1-D3), with ids matching solutions and no answers leaking. All cross-references match TOC numbers. No em dashes outside titles.
- Debugging set had no security-flavored broken system, although the chapter claims the business rules defend against injection.
- Minor: Project 1 prompts are constants, not Chapter 4's `PromptRegistry`. This is acknowledged in prompts.py and kept, see open items.

**Production/SRE Engineer**
- Operations paragraph listed signals but had no alert conditions, no mapping from signal to likely cause, and no degraded-mode runbook (outage, model update, 502 spike, review backlog).
- No capacity reasoning for the human side (review staffing), though the cost section implied it.
- Truncation and deadline signals not tied to concrete span attributes.

### Changes applied

Code (`book/projects/p1-extraction-api`). Public names are unchanged. All additions are backward compatible: new keyword args with defaults, new optional fields.
- New `extraction_api/api/auth.py`: `Principal`, `Authenticator`, `ANONYMOUS`. Bearer API keys use constant-time comparison. Roles are submitter/reviewer/admin, with an optional tenant binding. Another tenant's item returns 404, not 403. Auth-off mode is logged as a warning.
- `config.py`: new `ApiKey` model plus settings `api_keys` (`EXTRACT_API_KEYS`, JSON list, SecretStr keys), `max_request_bytes`, `document_deadline_s`, `call_timeout_s`.
- `api/app.py`:
  - auth dependencies on `/extract*` and `/review*`; `/healthz` stays open
  - tenant stamped from credentials, 403 on mismatch; review list, get, and resolve scoped by tenant
  - recorded reviewer = authenticated principal
  - `Content-Length` guard returns 413 before parsing
  - retryable `LLMError` maps to 503 with `Retry-After`; non-retryable maps to 502 with `retryable: false`
- `application/service.py`:
  - per-document deadline with an injectable `clock`; every call's `timeout_s` = min(call timeout, remaining)
  - exhausted deadline before extraction raises a retryable `TimeoutError` (503 or batch error)
  - too little time before a repair degrades to review with `REPAIR_SKIPPED_DEADLINE`
  - review items carry `tenant`; `BatchItem.retryable`
- `application/classifier.py`: `classify(..., timeout_s=None)`.
- `application/ports.py`: `ReviewItem.tenant` (optional); `ReviewQueue.list(..., *, tenant=None)`.
- `adapters/review_queue.py`: tenant filter in both adapters. SQLite uses one parameterized statement with `json_extract`.
- `wiring.py`: passes the deadline and timeout settings.
- New `tests/test_security_and_limits.py` (12 tests): 401/403, tenant stamping and cross-tenant invisibility, reviewer identity, 413 body limit, 502 mapping, batch retryable flag, deadline paths with a fake clock, SQLite tenant filter.
- `env.example` and README configuration/API tables updated, including an auth section.

Chapter:
- Re-pasted full listings of `service.py` and `app.py`; added a full listing of `auth.py`; updated the classifier excerpt and the directory tree. All listings re-verified against disk.
- Configuration table gains the deadline, timeout, and API-key variables. The "How it works" steps now cover body limit, auth/tenant binding, deadline, and the degraded repair skip.
- Softened the self-consistency claim, with the caveat that agreement can be high on a wrong answer.
- New paragraph on input modalities: OCR vs vision model, and the effect on quote checks (Ch 11 and Ch 7 references).
- API section explains the 503/502 split and why reviewer identity comes from credentials.
- Production considerations:
  - new **Reliability** paragraph: worst-case call count, the deadline, three timeout outcomes, read-only retry safety vs the non-idempotent review queue
  - rewritten **Security** paragraph: what is enforced and why 404, tenant from credentials, auth-off warning, no values or text in spans, proxy body limit
  - new **Observability and alerts** table: 10 signals, each with span source, illustrative alert condition, and likely cause
  - new **Runbook for degraded modes**: outage, model update, 502 spike, review backlog (never lower the threshold to drain the queue), plus human-side capacity arithmetic
- Failure-modes table: truncation row now points to P2. New rows: planted values, cross-tenant review exposure, retry storm on a non-retryable error, slow documents holding workers.
- Evaluation section lists the new security and limit tests. New common mistake (tenant from the request body). New key takeaway (the review queue is a sensitive store).
- Fixed the extraction-pipeline paragraph to mention 502 and the retryable flag.
- New debugging exercise **D4**: a value planted by indirect prompt injection passes every gate because grounding proves presence, not authority. The solution in `ch06-solutions.md` covers root cause, the violated mental model (the model proposes, code authorizes), model-independent fixes, telemetry, and a test. K6's solution was updated for 502 and the retryable flag.

Verification: `.venv/bin/python -m pytest book/projects/p1-extraction-api -q -p no:cacheprovider` gives **83 passed** (was 71). `run_eval` output is unchanged and matches the chapter. `build_book.py --check` reports nothing for Ch 6. Its only messages are "chapter 15 missing" and "chapter 39 missing", which are other owners' files and the cause of exit code 1.

### Second-pass findings

- Checked again that every full listing is byte-identical and every excerpt is present on disk, after all edits: OK.
- No other code in the repo imports `extraction_api`, so the API additions break no dependent suites.
- One misquote of a book mental model in the new D4 solution was fixed to cite mental model 6 exactly.
- Two places still described only the 503 path (extraction-pipeline paragraph, K6 solution). Both were updated.
- No em dashes outside titles; new numbers labeled illustrative; no vendor or model names added.

### Remaining open items

- **Truncation detection is still an exercise (P2)** and is not implemented. `complete_structured` (aie_core, Ch 3) hides `finish_reason`, so a clean implementation wants an aie_core change: return or raise on `length`, for example a `TruncatedResponseError` or exposing the last completion. That is a shared package outside this fork's scope. The chapter now says plainly that P1 does not detect it yet.
- **Idempotent processing (P4) is still an exercise.** A retried `/extract` or batch can create duplicate review items (D3). Kept as the practical exercise by design. The chapter states the gap in the Reliability paragraph.
- **The deadline is approximate.** `complete_structured`'s internal schema re-asks reuse one `timeout_s`, so the budget can be overrun by up to `max_schema_repairs` call timeouts. Exact propagation belongs to Ch 29's `Deadline` or an aie_core change. The chapter states this.
- **Prompts are not in Chapter 4's `PromptRegistry`.** They are versioned constants with one `prompt_version`. Moving them would couple P1 to `book/projects/examples/ch04`, which is not an installed package. Revisit if the Ch 4 registry is packaged.
- **Integration notes not updated** (shared file). Suggested line for `07-integration-notes.md`: "Project 1 now has `api/auth.py` (Principal, Authenticator; EXTRACT_API_KEYS JSON list of {key, principal, role, tenant}), `ReviewItem.tenant`, `ReviewQueue.list(..., tenant=)`, per-document deadline (EXTRACT_DOCUMENT_DEADLINE_S / EXTRACT_CALL_TIMEOUT_S), 502 for non-retryable provider errors, `BatchItem.retryable`."

---

## Ch 7 — Model Selection and Routing

### Gaps found (by reviewer role)

**Principal AI Engineer**
- Strong baseline. All brief items are present and developed: ten selection axes including data residency, evaluation-driven procedure with Wilson lower bounds and paired counts, the reasoning-effort knob, multimodal inputs, the routing ladder (static, rules, classifier, embedding centroids, model-as-router, cascades), calibration/ECE, routing error cost in a utility function with worked numbers, capability-aware fallbacks, and pinning. The catalog is fictional and labeled illustrative, and no vendor facts are stated as truth.
- Model deprecation/migration got only one sentence ("track deprecation dates"). There was no procedure for moving off a retired pin, no treatment of capability changes between versions, no prompt portability, and no discussion of what error a retired model produces.
- "The primary has a 200k window; the fallback has 32k" was not labeled illustrative.
- The Wilson interval was used in the selection table without a one-line explanation (Ch 24 comes later).

**Senior Software Architect**
- Layering hole: the router trusted that the alias it called is the model that answered. A `ModelGateway` with its own capability-changing fallback list, which D2 describes, was undetectable in code. The solution only suggested a guard that did not exist.
- Reliability bug: when a cascade route's escalation target had a hard gap (for example an image request with a text-only strong model), escalation was disabled and the low-confidence answer was returned **without** `degraded`. The route's quality bar silently stopped applying to those requests.
- The span lacked the pinned model id, attempt count, and warning count, even though the prose said the pinned version is logged on every request.
- The listings matched the files on disk. The test excerpt also matched.
- The selection table in the chapter omitted the `errors` column that `SelectionReport.to_markdown` prints.

**AI Educator**
- Exercises were complete (K1-K6, E1-E4, P1-P4, D1-D3), with no answers in the exercises and ids matching the solutions. Cross-references (Ch 2, 3, 4, 6, 8, 24, 25, 26, 29, 30, 31, 32, 33) match the TOC.
- No debugging exercise covered deprecation, error mapping, or silent cost regressions.
- The text said the full test file has 33 tests, which needed an update after the new tests.

**Production/SRE Engineer**
- Operations advice was prose only, with no concrete signal-to-alert mapping (which attribute, what threshold, what it usually means).
- Cost control lacked budget-aware routing and a cap on reasoning-effort tokens per route.
- There was no failure mode for a retired pin or for a hidden fallback below the router.

### Changes applied
- `book/projects/examples/ch07/router.py` changes, all backward compatible because they only add fields:
  - `RouteDecision.escalation_disabled` is new.
  - `RoutedCompletion.model_mismatch` is new.
  - A disabled cascade now still runs the validator and confidence check, and marks failing answers `degraded`.
  - The router compares `completion.model` with the served alias's pinned `model_id`.
  - New span attributes: `router.model_id`, `router.model_mismatch`, `router.attempts`, `router.warnings`.
- `test_ch07.py` changes:
  - New `test_disabled_escalation_marks_low_confidence_answer_degraded`.
  - New `test_model_mismatch_below_the_router_is_detected`, which uses a stub for a gateway that silently served from another model.
  - The span test also asserts the new attributes.
  - The suite now has 35 tests.
- Chapter edits:
  - Re-pasted the router listing, which is byte-identical to disk again, and updated the test count.
  - Added the `errors` column to the selection table, plus a sentence on Wilson intervals and the p50/p95 caveat for fakes.
  - Labeled the window sizes as illustrative.
  - Added a five-step migration procedure for retired pins: separate alias, harness and cascade re-sweep, prompt porting consistent with Ch 4's one-change-per-rollout rule, shadow then canary, keep the old pin.
  - Added an error-mapping paragraph. `aie_core` maps 404 to non-retryable. A proxy returning 5xx turns an outage into a silent fallback.
  - Updated How it works step 5 for disabled escalation and step 6 for the mismatch check and new span attributes.
  - Added two Code walkthrough paragraphs: the mismatch check and the disabled cascade.
  - Production considerations: budget-aware routing and effort token caps under Cost. Added a "What to measure and alert on" table with 11 signals, each with source attribute, illustrative alert threshold, and usual cause.
  - Two new failure modes: "Hidden fallback below the router" and "Retired pin".
  - New debugging exercise D4: a retired pin behind a proxy causes a silent fivefold cost rise through fallback.
  - The key takeaway on pinning now covers mismatch alerts and early migration.
- `book/solutions/ch07-solutions.md`:
  - The D2 answer now references the implemented `router.model_mismatch` attribute.
  - Added the D4 answer: root cause, the router's behavior, telemetry, and the two preventive changes, which are the deprecation check (P3) and error mapping plus a per-alias error alert.
- Tests: `35 passed` for book/projects/examples/ch07. The demo output was re-run and still matches every number quoted in the chapter.

### Second-pass findings
- All four rewritten or added listings compare byte-identical to disk. The test excerpt functions are still verbatim in `test_ch07.py`.
- The span attribute names in the new alert table match the code (`router.route`, `router.stage`, `router.escalated`, `router.degraded`, `router.model_mismatch`, `router.substitutions`, `router.warnings`, `router.cost_usd`).
- Checked the worked numbers for consistency:
  - Cost ratio about 5x: 0.000928 versus 0.000186.
  - Latency ratio about 3.6x.
  - Cascade p95 of 2,050 ms at escalation above 5%.
  - E4's 520 ms mean matches the demo's cascade at the 0.70 threshold.
- No em dashes outside the titles. No vendor names, prices, or context sizes are stated as facts.
- The first migration draft said the prompt and the pin "move together", which contradicts Ch 4 ("never change the model and the prompt in the same rollout"). It was rewritten to evaluate the pair offline and roll out one change at a time.

### Remaining open items
- `build_book.py --check` exits 1 only because chapters 15 and 39 are missing. These are outside group B and are being written by other agents. It reports no Ch 7 problems.
- The router has no end-to-end deadline across a cascade's two stages. It relies on per-client `ModelGateway` timeouts. The chapter recommends a tight timeout on the cheap stage. A request-level `Deadline` belongs to Ch 29's `reliability` package and was not wired in, to avoid a new cross-package dependency in an early chapter.
- Exercise P3 (`deprecation_date` plus startup check) is left as an exercise on purpose. The chapter and D4 now motivate it.

---

## Ch 8 — Embeddings

Files: `book/chapters/08-embeddings.md`, `book/solutions/ch08-solutions.md`, `book/projects/examples/ch08/`.

Overall: the chapter was already strong. It covers every TOC must-cover item with a working implementation and tests: metrics and normalization with a worked example, Matryoshka truncation with a measured report, chunk vs document, query/passage asymmetry, vendor-neutral model choice, quality evaluation on own data, space fingerprint versioning, batching, caching, and all six non-RAG use cases (dedup, clustering, kNN/centroid classification, intent routing, anomaly detection, MMR recommendation), each with threshold calibration and abstention. The gaps were mostly reliability and operations, plus stale or drifted code listings.

### Gaps found (by reviewer role)

**Principal AI Engineer**
- Anomaly threshold was computed in-sample, on the same items the centroids were fit on. That is optimistic and over-flags new normal traffic. The text did not say so.
- `drift_score` (centroid cosine) only detects a mean shift. A minority new topic is invisible to it, and no complementary signal was offered.
- The text about `aie_core.CachedEmbeddings` was outdated. It said the cache keys on model and text only, but `aie_core` now salts keys with a space fingerprint. The real remaining gap is normalization, post-processing, and lazily learned dimensions.

**Senior Software Architect**
- No retries on the embedding path. `ModelGateway` wraps chat completions only, and `BatchingEmbeddings` called the provider bare. A transient 429 failed a whole re-index.
- `VectorIndex` had no delete, while the chapter requires right-to-erasure deletion of vectors.
- `plan_reembed` computed cost but not the migration window. The chapter itself argues the window, not the bill, is the binding constraint.
- A shared cross-tenant embedding cache was called "safe". It leaks membership through hit timing, so it is correct but not private.
- Several excerpt listings did not match disk: simplified signatures in quality.py, recommend.py, clustering.py, and reflowed test excerpts.

**AI Educator**
- The three debugging exercises described symptoms in prose. None presented an actual broken trace.
- The test count in the text, 36, was stale.

**Production/SRE Engineer**
- No degraded modes were defined for an embedding-provider outage, for each consumer: retrieval, router, dedup, classifier, anomaly.
- No alerting guidance. Telemetry was listed, but there was no "alert when / usually means" mapping and no probe-set job spelled out as the detector for silent provider model updates.
- Spans lacked the space fingerprint, retries, and zero-vector counts, although the failure-modes section relies on comparing fingerprints from spans.

### Changes applied

Code, all backward compatible, with public names unchanged:
- `embedlab/pipeline.py`: `BatchingEmbeddings` gains optional `retry: RetryPolicy` (reuses `aie_core.llm.gateway.RetryPolicy`) and an injectable `sleep`. It retries retryable `LLMError`s per batch, honors retry-after, and raises non-retryable errors at once. `EmbeddingStats.retries` was added. `EmbeddingPipeline` accepts and passes `retry`. The embed span now records `space` (fingerprint), `retries`, and `zero_vectors`. The `NamespacedStore` docstring was corrected to the current `aie_core` behavior.
- `embedlab/space.py`: `VectorIndex.delete(ids) -> int` for erasure. `plan_reembed(..., tokens_per_minute=None)` and `ReembedPlan.estimated_minutes`.
- `embedlab/usecases/anomaly.py`: `CentroidAnomalyDetector.calibrate(held_out_normal)` and `flag_rate(vectors)`. The `drift_score` docstring now states its blind spot.
- Tests: five new tests cover retry with a retry-after hint, a non-retryable error surfacing immediately, delete, the migration window, and held-out calibration. They also show `flag_rate` catching a minority topic while `drift_score` stays below 0.05. The span test now asserts the new attributes. The suite went from 36 to 41 passing.
- README module table updated. `demo.py` re-run and all chapter numbers are unchanged.

Chapter:
- Full listings for space.py, pipeline.py, and anomaly.py were re-pasted from disk. The quality.py, recommend.py, clustering.py, and test excerpts were replaced with exact disk text, extracted by AST. All listings now match disk. corpus.py is relabeled as a signature-form interface summary.
- Code walkthrough: the cache-key paragraph was rewritten to match current `aie_core`. The batching paragraph now explains per-batch retries, why retrying embeddings is safe, and why retries are per batch.
- Anomaly section: an in-sample threshold caveat with the held-out `calibrate` pattern, and the drift-score blind spot with `flag_rate` as the complement.
- Production considerations: a new "Failure recovery and degraded modes" paragraph. Read path gets a deadline, one retry, then lexical-only retrieval, LLM routing, and checks marked as skipped. Write path gets backoff, checkpoints, and a pause. It warns never to zero-fill a failed embedding and cross-references Ch 29.
- New "what to measure and alert on" table with 8 signals, plus the daily probe-set job.
- Cost now mentions the window estimate. Security covers the erasure path, including cache entries, and the cross-tenant cache membership leak.
- Mermaid pipeline node now shows retry. Test count fixed. A tenth key takeaway on degraded modes and alerts.
- New debugging exercise **D4** presents an actual span trace. Nightly mean-centering puts the corpus mean into the fingerprint, which causes a full re-embed every night and morning `SpaceMismatchError`s. A full solution was added to `ch08-solutions.md`: root cause, telemetry, and fix. The fix is to cache raw vectors, freeze the mean as a versioned artifact, and have the query service watch the active index version.

Verification:
- `.venv/bin/python -m pytest book/projects/examples/ch08 -q -p no:cacheprovider`: 41 passed, 1 deselected.
- `build_book.py --check`: no problems for chapter 8 (`check_chapter` returns `[]`). The global check exits 1 only because chapters 15 and 39 do not exist yet, which is outside this group.
- No em dashes outside titles.

### Second-pass findings

- Cross-references verified against chapters on disk:
  - Ch 28 has `index_versions`.
  - Ch 9 has quantization.
  - Ch 29 has degradation and circuit breakers.
  - Ch 30 has the semantic cache.
  - Ch 12 has lexical and hybrid search.
  - Ch 6 has classification.
  - Ch 27 has guardrails.
- Exercise ids K1-K6, E1-E4, P1-P4, D1-D4 all match the solutions. No answers appear in the Exercises section.
- No other code in the repo imports `embedlab`, so the API additions affect nothing else.
- Numbers in prose are labeled illustrative or come from the demo. No vendor or model facts are stated.

### Remaining open items

- **`aie_core.CachedEmbeddings` provider field.** It is fingerprinted from `getattr(inner, "provider", type name)`. When wrapped around this chapter's `BatchingEmbeddings`, that yields "BatchingEmbeddings", not the real provider. The chapter is unaffected because `NamespacedStore` adds the full space. Shared package, so this is not fixed here.
- **`aie_core.CachedEmbeddings.embed` length check.** It returns `[r for r in results if r is not None]`. If a provider returns fewer vectors than requested, the result is silently shorter instead of raising. Shared package, so this is not fixed here.
- **`ModelGateway` covers chat completions only.** An embedding gateway, with retries, rate limiting, and a circuit breaker for `EmbeddingClient`, would remove the per-project retry loop. That is an `aie_core` design change for its owner, Ch 3.

---

