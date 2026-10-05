# Review group E: Chapters 20, 21, 22, 37, 38

Scope: chapters, solutions, and code in `book/projects/examples/ch20`, `p5-incident-agent`, `memorykit`,
`p6-research-team`, `examples/ch37`, `examples/ch38`. Baseline before changes: all six suites green
(24, 39, 91, 28, 47, 38 tests), every chapter listing identical to its file on disk (checked with a
listing-vs-disk script: full listings byte-equal, excerpts as ordered line subsequences), exercise ids
in chapters and solutions matching (K1-K6, E1-E4, P1-P4, D1-D3 in all five), no em dashes outside titles.

## Gaps found (by reviewer role)

### Principal AI Engineer
- Ch 21: the must-cover list was complete in substance (conversation, episodic, semantic, procedural,
  profile, summarization, retrieval, consolidation, expiration, privacy, storage, poisoning, harm), but the
  standard terms "short-term" and "long-term" memory were never mapped onto the stack, so a reader could
  not connect the chapter to the vocabulary used elsewhere.
- Ch 21: the chapter claimed Chapter 5's summary guard protects conversation summaries ("the Chapter 5
  summary guard's rejection rate"), but `memorykit.ConversationMemory` accepted any summary unchecked. A
  summarizer that invented `INC-9999` would have turned a guess into session state. Claim and code
  disagreed.
- Ch 21: no description of how an episode is built from an agent run (which fields come from the harness
  versus the model), although the chapter insists episodes are "harness facts".
- Ch 37: `AgenticRAG` is a hand-written model-decides/code-executes loop, i.e. a second copy of Chapter
  19's harness (budgets, stall detection, step trace) without its event log, replay, or resume. Chapter 20
  explicitly calls this a mistake, and the review brief requires Ch 37 not to duplicate Ch 19. The chapter
  neither justified the custom loop nor showed the pattern on `AgentRuntime`.
- Ch 38: "Replay and counterfactual testing" re-defined harness replay and counterfactual replay almost
  verbatim from Chapter 19 (same framing, same limits), duplicating owned material.

### Senior Software Architect
- Project 6 (Ch 22): researchers run in a `ThreadPoolExecutor`, and `aie_core` links spans to parents via
  a context variable that thread pools do not inherit. Verified experimentally: every researcher
  `agent.run` span had `parent_span_id=None` and a fresh native `trace_id` (three orphan traces per team
  run). Only the custom `parent.span_id` attribute was correct. Ch 22's tracing docstring and prose also
  said "aie_core spans have ids but no parent links", which is outdated since aie_core gained
  `trace_id`/`parent_span_id`.
- Project 6 run ids (`<trace>.r1.sq1`, `<trace>.verify.r1`) used `.` inside a segment, contradicting
  Chapter 20's derived-id rule, where each `.` is one level (`run_hierarchy` computes depth by counting
  `.`). A depth-1 researcher read as depth 2.
- Project 5 (Ch 20): with a separately configured judge model, `IncidentService` created a second,
  independent `MeteredLLM`, so the judge had its own ceiling (doubling the "hard ceiling across all
  roles" the chapter promises) and its calls were missing from `inv.usage`.
- No layering, secrets, or tenant-isolation defects found in the group's code beyond these. Approval,
  idempotency, and ACL paths in P5, P6, and ch38 are tested.

### AI Educator
- Ch 22 was written before Ch 20 and referenced it only implicitly. Its "Coordination patterns" section
  re-taught supervisor-worker, router-specialist, map-reduce (fan-out/fan-in), and pipeline (sequential),
  all owned and implemented by Ch 20, with no pointer to Ch 20's sections, code, or fan-in policies.
- Vocabulary drift between Ch 20 and Ch 22: Ch 20's `Supervisor` (model-driven delegation via
  `delegate_<worker>` tools, `SpawnBudget`) versus Ch 22's code-driven `_Run` supervisor (`TeamBudget`
  `max_children`/`max_depth`, "spawn cap") were never related; which roles are "workers" in Project 6 was
  never stated; Ch 22 never said that Ch 20's fixed per-worker `Budget` is the fixed-slice option it
  argues against.
- Ch 20 pointed to Ch 22 only generically ("treats multi-agent systems in depth").
- Ch 22's "Self-critique in the same context" mistake duplicated Ch 20's reflection guidance without a
  reference.
- Ch 37 re-described full-text/BM25 without noting Ch 12 owns BM25 and the tsvector form.
- Ch 38 exercise ids were unbolded (`K1.`), inconsistent with every other chapter.
- Test counts quoted in Ch 20, Ch 22, and Ch 37 would go stale with the new tests.

### Production/SRE Engineer
- Ch 21 production section listed metrics but had no tracing guidance: no span names, no attributes, no
  rule to record which memory ids were rendered into a prompt (needed to trace a bad answer to a memory).
- The P6 orphan-trace defect above is also an observability gap: an OTel backend would have shown each
  researcher as an unrelated trace.
- The P5 ceiling defect is a cost-control gap.
- Ch 20, 22, 37, 38 production sections were otherwise concrete (alerts, budgets, degraded modes,
  runbook signals); no further gaps.

## Changes applied

### Code (all suites re-run, all green)
- `p6-research-team/research_team/team.py`: researchers are submitted via
  `contextvars.copy_context().run`, so native span links survive the thread pool; run ids now use
  `-` inside segments (`<trace>.r1-sq1`, `<trace>.verify-r1`).
- `p6-research-team/research_team/tracing.py`: docstring rewritten (contextvar links, thread and process
  boundaries, why attribute stamping is still needed).
- `p6-research-team/tests/test_team.py`: new `test_native_span_links_survive_the_thread_pool` and
  `test_run_ids_use_one_dot_per_level`; follow-up id assertion updated. `test_contracts_and_ledger.py`
  example id updated. README run-log tree and a run-id rule paragraph updated. Benchmark re-run: numbers
  unchanged, Ch 22 table still accurate.
- `memorykit/memorykit/conversation.py`: Chapter 5's literal guard (`literals`, `_guard`). `compact()`
  returns `False` and leaves turns in the window when a summary is empty or introduces an identifier or
  multi-digit number absent from its sources; rejections recorded in `ConversationMemory.rejected`.
  During `redact()`, a rebuilt summary that fails the guard is dropped rather than keeping the old one
  (which may contain the redacted value). `add()` now returns True only when compaction was accepted.
  Backward compatible: capstone call sites ignore the return value. Two new tests.
- `p5-incident-agent/incident_agent/adapters/metering.py`: shared `_Meter`; `MeteredLLM.share(inner)`
  meters another client against the same ceiling and totals; `calls`/`tokens` kept as properties.
  `service.py` uses `metered.share(self.judge_llm)`. New test
  `test_a_separate_judge_model_shares_the_investigation_call_ceiling` (fails without the fix).
- `examples/ch37/agentic_rag_runtime.py` (new): `RuntimeAgenticRAG`, the same search tool, evidence
  ledger, ACL filter, and coverage gate on `agentkit.AgentRuntime`. The ledger is folded from the event
  log (no in-memory state), the gate is a Definition-of-Done verifier, budgets map onto `max_steps` and
  `max_tool_calls`, repeats and dry searches are stopped by the harness. `tests/test_agentic_runtime.py`
  (7 tests: multi-hop with replay, premature answer rejected via `dod_rejected`, unknown label, ACL,
  REPEATED_ACTION, MAX_TOOL_CALLS, ledger rebuilt from the log). `pyproject.toml` and README add agentkit.

### Chapters and solutions
- Ch 22: "Coordination patterns" rewritten. Router, map-reduce, and pipeline now point to Ch 20's
  implementations and keep only what an agent boundary adds. Supervisor-worker contrasts Ch 20's
  model-driven `Supervisor` with Project 6's code supervisor and defines who the workers are. Debate is
  related to Ch 20's reflection and evaluator-optimizer. Blackboard and market kept (Ch 22-only).
  Added Ch 20 cross-references in "What counts as a multi-agent system", the justification section
  (step 8 of Ch 20's decision procedure), "Budgets at parent and child" (Ch 20's fixed slices), "Spawn
  control" (`SpawnBudget.max_agents` = `TeamBudget.max_children`), and common mistakes. "Trace
  propagation" now explains the contextvar/thread-pool problem, the two-level fix, and the run-id rule.
  `team.py` and `tracing.py` listings re-pasted; test count 28 to 30. Solutions P1 id updated.
- Ch 20: supervisor section states the book-wide separator rule (`.` between levels only, `-` within a
  segment, never `/`) and lists exactly what Ch 22 adds, with section names. Parallel section warns about
  thread pools and context variables. Cost section documents the shared judge ceiling. Test counts 39 to 40.
- Ch 21: short-term versus long-term mapping added to "The memory stack". Summary guard described in the
  compaction section, code walkthrough, and failure-mode table (with the new test name).
  `conversation.py` listing re-pasted. New "Observability" paragraph (memory.recall and memory.write
  spans, attributes, rendered memory ids, dashboards, alerts). "How it works" explains how an episode is
  built from `RunResult` and the stop reason.
- Ch 37: the agentic RAG section now states the state-in-prompt versus harness choice and recommends the
  harness version by default; new implementation subsection with the full `agentic_rag_runtime.py`
  listing; code walkthrough paragraph; file tree, install command, and test count (47 to 54) updated;
  full-text paragraph points to Ch 12 for BM25/tsvector.
- Ch 38: "Replay and counterfactual testing" reduced to a pointer to Ch 19 plus three durable-specific
  points (recovered runs remain one recording, waits replay instantly, recordings age with tool versions).
  Exercise ids bolded.
- `book/analysis/07-integration-notes.md`: RuntimeAgenticRAG, run-id rule, thread-pool contextvar rule,
  ConversationMemory guard, `MeteredLLM.share`.

### Test results after changes
| Suite | Result |
|---|---|
| examples/ch20 | 24 passed |
| p5-incident-agent | 40 passed |
| memorykit | 93 passed |
| p6-research-team | 30 passed |
| examples/ch37 | 54 passed |
| examples/ch38 | 38 passed |

Listing-vs-disk check over the five chapters: 0 problems. `build_book.py --check`: the only problems
reported are "chapter 15 missing" and "chapter 39 missing", outside this group; the five group chapters
pass the structure, fence, length, and inline-answer checks.

## Second-pass findings
- Re-read the rewritten Ch 22 sections in context: one imprecise phrase ("narrower than the router's
  caller") replaced with "no broader than the caller's, so a misroute cannot widen access". The spawn
  control paragraph initially listed the cap and depth twice; rewritten as one list of five limits with
  the Ch 20 mapping inline.
- Re-checked cross-references against the TOC with a script (chapter numbers and titles): no mismatches.
- Re-checked em dashes, exercise ids, and solution ids: clean.
- Checked capstone call sites of `ConversationMemory` (`orchestrator.py`, `memory/service.py`): they ignore
  `add()`'s return value, so the guard change is compatible. The capstone currently has no tests to run.
- No new issues found in Ch 38 or Ch 37 on the second pass; Ch 20 and Ch 22 now agree on supervisor,
  worker, spawn-limit vocabulary, and the `.` separator.

## Remaining open items
- `build_book.py --check` fails on missing chapter files 15 and 39. Not in group E; owned by the Ch 15 and
  capstone authors.
- Ch 20's pattern helpers (`make_agent`) do not accept a tracer, so the thread-pool context fix described
  in the Parallel section is guidance rather than code there. Adding a `tracer` parameter would change
  every pattern signature and listing for a demo-only benefit; left as is.
- `ConversationMemory` retries a rejected compaction on every later turn over the trigger. That is the
  same behavior as Chapter 5's `ConversationState`; a backoff would be a product decision for the
  capstone, so it is documented rather than changed.

## Follow-up: agentkit citation pattern (after review D)
- Ch 22 "Roles are data" now says the baseline failed "with agentkit's earlier default pattern, which
  rejected `#`", notes that the default has since been widened (Chapter 19), and that P6 still passes an
  explicit pattern. Chapter 19 documents the widened default. No P6 code change; P6 suite: 30 passed.

## Follow-up: p6 pyproject listing (reported by rv-D)
- The Ch 22 listing of `p6-research-team/pyproject.toml` began with a `# path:` line that the file on
  disk lacked; the rest matched. Added the header comment to the file (valid TOML, parses), so listing and
  file are now byte-identical. My earlier checker had skipped that header for files without one; a strict
  re-check of every full listing in Ch 20, 21, 22, 37, 38 (header included) finds 0 differences, excerpts
  also 0. P6 suite: 30 passed.
