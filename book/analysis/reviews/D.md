# Review group D: Chapters 16, 17, 18, 19, 23

## Gaps found (by reviewer role)

### Principal AI Engineer
- Ch 17: the decision framework (table + flowchart) was sound, but "prefer the simplest" was implied by arithmetic rather than stated as a rule. The five positions were listed in TOC order while the comparison table used complexity order, with no explanation. "Deterministic" vs "probabilistic" was not stated to refer to the path, not the outputs. There was no worked placement of concrete features.
- Ch 17: the prose claimed that resume refuses a changed draft, and that side-effecting nodes use a key from `run_id`, node, and sequence number. The engine did neither. The sequence number is also the wrong ingredient, because a failed attempt consumes a seq, so the key would change on re-execution.
- Ch 18: the freshness note described 2026 protocol revisions as course-material facts. That borders on post-mid-2026 vendor claims.
- Ch 23: the Chapter 12 `Retriever` signature was misquoted. The ledger named Ch 16's governing component as `ToolRegistry`/`ToolPolicy` instead of `PolicyEngine`/`ToolExecutor`. Code comments said "API shape as of 2026".

### Senior Software Architect
- Ch 17 engine had four problems:
  - `ResumeHandle` was not bound to the paused state, so stale approvals were possible.
  - There was no idempotency key for `send`, so the "double send after resume" failure mode had no defense in code.
  - Steps mutated the shared state object, so a failed attempt could leak partial changes into the retry and into the failed checkpoint.
  - There was no async resume, so `resume` could not be called inside a running event loop such as FastAPI.
- toolkit (Ch 16): an `IdempotencyStore` outage (`get`/`begin` raising) propagated as an unclassified exception and crashed the loop. There was no defined degraded mode for writes.
- agentkit (Ch 19): the default `CITATION` regex rejected `#`, so `doc#section` passage ids (Ch 10/13 style) failed `citations_grounded`. P6 had to work around it. Grounding is a substring test, and that limit was undocumented.
- Ch 18 P4 exercise and solution referred to "Chapter 16's `ToolPolicy`" doing approvals and to an `IdempotencyStore` used directly. In toolkit, `ToolPolicy` is an alias of `PolicyEngine`, approvals are `ApprovalManager`, and the executor drives both. The default `approval_for` also excludes `REVERSIBLE_WRITE`, so `requires_approval=True` is needed.
- Ch 23: the "Chapter 19 primitive, simplified" loop used names that do not exist (`policy.check(call)`, `registry.execute`, `ToolEvent`, `termination.check`). The ch23 code excerpts were hand-simplified and differed from disk in 25 lines. The solutions referenced a nonexistent `AgentEvent`.
- Ch 19: the pyproject listing differed from disk by its path comment.

### AI Educator
- Ch 17 K1 asked for examples "not in this chapter". Its solution reused examples that the new placement table now uses, so the solution was rewritten. There was no exercise testing the ladder rule, so K6 was added.
- Ch 17 Mermaid diagrams used HTML `<br/>`, which the authoring guide forbids. Only Ch 17 in the book did this.
- Ch 16 mental-model callout did not use the book-wide wording (model 6). There was a typo ("happy-path test) `ToolLoop`") and the article "a `issue_refund`".
- Ch 23 E3 solution: the `Workflow` port's `ResumeHandle` lacked the state binding, so the stale-approval protection was lost across adapters.
- Exercise and solution ids match in all five chapters. Cross-reference titles match the TOC (checked with a script).

### Production/SRE Engineer
- Ch 16: there were no explicit degraded modes (idempotency store, approval store, audit sink down). The `tool.execute` and `tool_loop.round` spans were not documented, and there were no starting alert thresholds.
- Ch 17: there was no per-node span in the engine, even though Production considerations asked for one. Concurrent resume by two workers was not addressed, and neither was the source of the visit counts in a shared store.
- Ch 19: the event-store outage behavior (durable-first append raises; resumable afterwards) was not stated.
- Ch 18 and Ch 23 already had adequate telemetry and runbook content. No change was needed beyond the items above.

## Changes applied

**Code (all backward compatible; every dependent suite rerun):**
- `agentkit/agentkit/dod.py`: `CITATION` now accepts `#`. It is a strict superset of the old pattern, so explicit `pattern=` callers still work. Added `test_default_citation_pattern_accepts_passage_ids`. Added the path comment to `agentkit/pyproject.toml` so the Ch 19 listing matches disk.
- `toolkit/toolkit/executor.py`: store errors in the idempotency peek or reservation now return `transient`/`idempotency_unavailable` with `retry_after_s`. The executor emits `tool.failed` and never runs the handler. Reads are unaffected. Added `test_store_outage_fails_closed_for_writes`.
- `examples/ch17/workflow_engine.py`:
  - `ResumeHandle(seq, state_hash)` with defaults; `resume` raises `StaleHandleError` (a `ValueError`) on mismatch.
  - `step_key()` contextvar returning `run_id:node:visit`, with visit counts derived from checkpoint history.
  - A deep copy of the state on each attempt.
  - Optional duck-typed `tracer` emitting `workflow.node` spans (attempts, status, next node).
  - `aresume` and `aresume_from_checkpoint`.
- `triage_domain.send` now passes an idempotency key to a `Sender(ticket_id, body, key)`. The graph uses `step_key()`.
- Four new tests: stale handle refused; ambiguous send failure plus resume delivers once with the same key; failed attempt does not leak partial state; one span per node. README updated.
- `examples/ch23/ports.py`: docstring no longer says "as of 2026".

**Chapters:**
- Ch 17:
  - Clarified path-vs-output determinism and the order of the spectrum.
  - Added the explicit ladder rule and a five-feature Northwind placement table. The Project 6 row reflects Ch 22's measured result: the team ties single agent plus verifier and costs more.
  - Corrected the idempotency-key derivation (visit, not seq, with the reason) and the stale-approval binding.
  - How it works now covers the state copy, step key, spans, and async twins.
  - Updated the code walkthrough, failure modes (they now name the real tests), and Operations (unique `(run_id, seq)` against concurrent resume; visit counts from the store).
  - Refreshed the benchmark output.
  - Re-pasted all four changed listings and extended the test excerpt.
  - Removed `<br/>` from Mermaid and quoted labels.
  - Added K6. Updated the first key takeaway. Changed "two hundred lines" to "about 250 lines" (also in Ch 23).
- Ch 16:
  - Synced the executor excerpt.
  - Added a "Degraded modes" paragraph (store, approval, audit) and a "Tracing" paragraph with spans and illustrative alert thresholds.
  - Updated test counts (49).
  - Mental model now uses the book-wide wording. Fixed the typos.
- Ch 18: P4 names `PolicyEngine`, `ApprovalManager`, `IdempotencyStore`, and `ToolExecutor`. The P4 solution was rewritten around wrapping MCP grants as toolkit `Tool`s, including the `requires_approval=True` reason. The freshness note no longer asserts specifics of 2026 revisions.
- Ch 19: documented the widened `CITATION` default, backward compatibility, and two limits (substring grounding, bracketed non-ids). Added the event-store degraded mode to Operations.
- Ch 23:
  - The Chapter 19 loop reminder was rewritten as labeled pseudocode using real agentkit event, policy, and termination names.
  - The ledger and text now use `PolicyEngine`/`ToolExecutor`.
  - The Ch 12 `Retriever` signature is now correct.
  - The ledger rows are reordered by chapter, with `AITracer` added.
  - All three ch23 modules are shown in full, identical to disk.
  - Linked the interrupt hazard to `step_key()`.
  - "As of 2026" became "at the time of writing".
- Solutions:
  - Ch 17: K1 examples replaced; K6 added; E2 now uses `step_key()`.
  - Ch 18: P4 rewritten.
  - Ch 23: `AgentEvent` removed; the E3 port gains `ResumeHandle.binding`.
- `07-integration-notes.md`: the citation REVIEW TODO is marked done, and the new Ch 17 and toolkit behaviors are recorded.

**Test results (from repo root, `-q -p no:cacheprovider`):**

| Suite | Result |
|---|---|
| agentkit | 34 passed |
| toolkit | 49 passed |
| p4-support-assistant | 17 passed |
| p5-incident-agent | 39 passed |
| p6-research-team | 30 passed |
| examples/ch17 | 22 passed |
| examples/ch18 | 16 passed |
| examples/ch20 | 24 passed |
| examples/ch23 | 14 passed |
| examples/ch25 | 59 passed, 5 skipped |
| examples/ch38 | 38 passed |

The Ch 22 code is P6 (there is no examples/ch22). `build_book.py --check` reports only that chapters 15 and 39 are missing, which is outside this group.

## Second-pass findings
- Fixed: the Ch 17 placement table first claimed the multi-agent Project 6 design was "justified". That contradicted Ch 22's benchmark. The row was rewritten so the default is single agent plus verifier, with multi-agent only for latency or permission isolation.
- Fixed: the first draft of the Ch 16 degraded-mode text claimed behavior the code does not have (an unreachable remote approval store or audit sink). It was reworded as guidance for remote replacements, separate from what toolkit does.
- Verified with scripts:
  - Every full listing in the five chapters is identical to disk.
  - Every excerpt line exists on disk.
  - No em dashes outside titles.
  - Cross-reference titles match the TOC.
  - Exercise and solution ids match.

## Remaining open items
- Ch 22 (not in this group) says "with the default pattern every baseline answer failed its DoD". That is now historical, because the agentkit default accepts `#`. The Ch 22 owner should reword it to "with the earlier default". P6 code needs no change.
- The Ch 19 practical exercise P3 (`SqliteEventStore`) overlaps an implementation that Ch 38 ships. It is left as an exercise. The solution could point to Ch 38 for comparison, which is a Ch 19/38 coordination choice.
- Word counts exceed the TOC targets: Ch 19 has about 20k words against 9-10k, and Ch 16 about 13k. Most of the excess is code listings. Cutting it would mean removing listings the brief requires to match disk, so it was not done.

## Follow-up: agentkit idempotency key policy (capstone API mismatch #1)

**Problem.** `agentkit.executor_tools` forwarded the runtime's key `run_id:request_id` to toolkit as an explicit key. toolkit prefers an explicit key to its content-bound default (tool, tenant, session, argument hash). As a result, the same `create_ticket` proposed in two runs executed twice.

**Change (backward compatible API, safer default):**
- `executor_tools(..., idempotency="content" | "run" | callable)`.
  - `"content"` (the default) passes no key, so toolkit derives its own content-bound key. This suppresses duplicates across runs of one session and still covers crash or resume re-execution, because the arguments are identical.
  - `"run"` restores the previous run-scoped key as an explicit opt-in.
  - A callable `(tool_name, arguments, ctx) -> key | None` supplies a business key. Returning `None` falls back to the content key.
- An invalid mode raises `ValueError`. `IdempotencyMode` is exported.
- Both adapter paths are covered: the `bind()` path passes an empty key that toolkit treats as "derive default", and the direct path passes `None`. `_ExecutorTool` constructed directly keeps the run-scoped key. `adapt_tool` is unchanged.
- Tests added to `agentkit/tests/test_toolkit_integration.py`:
  - the default executes once across two runs;
  - `"run"` executes twice;
  - a business key executes once and receives the agentkit ctx;
  - an invalid mode is rejected.

**Docs.**
- Ch 19 explains the two layers' notions of "same action", the three modes, the default and why it is the default, keeping `session_id` stable, and correlating the agentkit and toolkit keys by `call_id`. The failure-mode table gains "duplicate side effect across runs", and the `tools.py` listing is re-synced.
- Ch 16's idempotency section warns against forwarding per-attempt keys and names the adapter default.

**Tests:**

| Suite | Result |
|---|---|
| agentkit | 38 passed |
| toolkit | 49 passed |
| p4 | 17 passed |
| p5 | 40 passed |
| p6 | 30 passed |
| examples/ch20 | 24 passed |
| examples/ch25 | 59 passed, 5 skipped |
| examples/ch38 | 38 passed |
| capstone/northwind-assist | 62 passed |

**Open, outside group D:**
- Ch 39, around line 497, and the capstone README's "API mismatches" #1 describe the old behavior. They should note that agentkit now defaults to the content key. The capstone's direct executor call remains valid.
- The Ch 22 listing of `p6-research-team/pyproject.toml` differs from disk.
