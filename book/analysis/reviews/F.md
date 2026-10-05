# Review group F: Chapters 24-27 (Evaluation, Security and Guardrails)

Scope: chapters 24, 25, 26, 27; solutions ch24-ch27; code in `book/projects/evalkit`,
`book/projects/examples/ch25`, `book/projects/examples/ch26`, `book/projects/guardrails`.
Shared-data files were not edited (Ch 25 pins their hashes).

Baseline state before edits: all four suites green (evalkit 57, ch25 58+5 skipped, ch26 20, guardrails
133); every `# path:` listing matched disk except two cosmetic trims in Ch 25; the Ch 24 worked example
reproduced every number in the chapter (dataset hash `1708ea5e315d`, accuracy 0.891/0.859, macro-F1
0.867, ECE 0.063, gate FAIL on the critical rule); exercise ids matched solution ids in all four chapters.

Coverage check against the lead's checklist. Evaluation: "no evaluation, no engineering" framing, golden,
synthetic, production-sampled, adversarial and regression datasets, regression testing, deterministic,
model-based, LLM-as-judge, human, pairwise, rubrics, metrics (precision, recall, F1, groundedness,
faithfulness, correctness, relevance, task completion, tool correctness), evaluation of prompts, RAG,
agents, extraction, classification, summarization and tool use, CI/CD integration: all present and
developed before the review. Security: direct and indirect injection, jailbreaks, malicious documents,
tool abuse, exfiltration, secrets, insecure output handling, excessive agency, permission boundaries,
tenant isolation, PII, moderation, threat models: present; real-world incident patterns, sensitive
information disclosure as its own class, harmful content/misuse in the threat catalogue, and a red-team
plan were missing or thin (fixed below).

## Gaps found (by reviewer role)

### Principal AI Engineer
- Ch 24: statistics assumed independent cases; no treatment of correlated cases (turns of one
  conversation, questions from one document), so intervals on grouped datasets are overconfident.
- Ch 24: judge design lacked reference-based vs reference-free choice, judge model selection, and judge
  panels.
- Ch 24: gate section did not explain the choice between point-estimate regression tolerance and
  significance-based rules, although `evalkit` supports both.
- Ch 25: no evaluation of multi-turn conversations (replayed transcripts, simulated users), although
  Northwind Assist is conversational.
- Ch 25: online evaluation relied on feedback and canaries only; no sampled reference-free scoring of
  production traffic and no drift alerting.
- Ch 26: brief requires "real-world incident patterns"; the chapter had none.
- Ch 26: harmful content/misuse (the moderation threat) and sensitive-information disclosure (wrong user,
  provider, persistent outputs, fine-tuning data) were absent from the threat catalogue.
- Ch 26: jailbreak section missed multi-turn escalation, many-shot priming, cross-modal and
  cross-lingual carriers.
- Ch 26: STRIDE used in both worked tables but never defined; risk scale not explained.
- Ch 26: "red-team plan" from the brief existed only as a list of test assertions.

### Senior Software Architect
- Ch 25 (bug): `gates.toml` did not set `require_baseline`, so a missing or expired baseline artifact
  silently skipped every regression and slice rule and the gate could pass a regressed system. This
  contradicted the chapter's own rule that a missing run must never pass.
- Ch 24: the example `gate.toml` was not pinned to a dataset hash and did not require a baseline, while
  the prose insists gates are pinned.
- Ch 24: the worked example gates on the full 64-case set rather than the frozen holdout it just split,
  without saying so.
- Ch 27: wrong cross-reference: generated-SQL allowlisting attributed to Chapter 16; it lives in Chapter
  36, Case B.
- Ch 26: chapter showed no code listing from its own modules except one test without a path comment;
  `examples/ch26` had no README/run instructions.
- Ch 25: two listings differed from disk (a dropped `# type: ignore` in `replay.py`, an abbreviated
  `CanaryMonitor` docstring).

### AI Educator
- References to material the reader does not have: "the source material" (Ch 24 x1, Ch 25 x1, Ch 26 x4
  including exercise K1, Ch 27 x1), "Case Study 6/7" (Ch 25 and ch25 solutions), "Recipe 9" (Ch 26),
  "Workshop L" (Ch 27 and ch27 solutions).
- Ch 26: em dashes in prose (implementation file list), against the style rules.
- Ch 26: agent threat table listed A6 (risk 2) before A7 (risk 3), contradicting `by_risk()` and the
  text that says tables are risk-ordered.
- Ch 24 / Ch 26: no exercise on correlated-case statistics, incident response, or the incident patterns.

### Production/SRE Engineer
- Ch 24: no monitoring of the evaluation system itself (run cost and duration, evaluator errors, judge
  drift on a control run, dataset staleness, gate override rate) and no degraded mode when evaluation
  infrastructure is down.
- Ch 25: safety suites (Ch 27 red team) were not wired into the same release pipeline.
- Ch 26: no security telemetry specification and no incident-response runbook (containment by
  capability, poison removal from index/cache/memory, rotation, closure into regression sets).
- Ch 27: approvals treated as purely technical (no approval fatigue, approver UX, token expiry);
  per-session and per-user signals for multi-turn and slow-drip attacks missing; no link from canary
  hits to an incident runbook.

## Changes applied

### Code (all backward compatible; defaults unchanged)
- `evalkit/stats.py`: `paired_bootstrap(..., groups=None)` adds a cluster bootstrap (resample whole groups,
  sign-flip per group) via new private `_cluster_resamples`; with `groups=None` the RNG call order and
  results are identical to before. `compare_runs` passes kwargs through (`Any`).
- `evalkit/gate.py`: `evaluate_gate(..., groups=None)` forwards groups to metric regression intervals;
  added a docstring.
- `evalkit/tests/test_stats.py`: new `test_cluster_bootstrap_widens_the_interval_when_cases_are_correlated`
  (200 cases from 20 conversations; row-level CI [+0.10, +0.20] p=0.000 vs clustered [+0.00, +0.30]
  p=0.253).
- `evalkit/data/gate.toml`: `pinned_dataset_hash = "1708ea5e315d"`, `require_baseline = true` (example
  output unchanged).
- `evalkit/README.md`: API rows for `paired_bootstrap(groups=)` and `evaluate_gate(groups=)`.
- `examples/ch25/ci/gates.toml`: `require_baseline = true` in all four suites.
- `examples/ch25/tests/test_release_gate.py`: new
  `test_missing_baselines_block_instead_of_skipping_regression_rules` (empty baselines folder exits 1
  with "baseline present" in the summary).
- `examples/ch26/README.md`: new; public names, run commands, corpus location.

### Chapter 24
- Removed the source-material reference in "Why this matters" (now cross-references Ch 31).
- LLM-as-judge: new paragraph on reference-based vs reference-free judges, choosing the judge model by
  calibration, and judge panels.
- Statistics: new cluster-bootstrap paragraph with the measured table from the new test.
- Gate section: re-pasted `gate.toml`; new paragraph on the pin, `require_baseline`, and point-estimate
  vs significance regression rules (`max_regression`, `fail_on_significant_regression`, `min_ci_low`).
- Re-pasted the `stats.py` excerpt (`paired_bootstrap` and `_cluster_resamples`) from disk.
- Walkthrough: states that the toy example gates on all 64 cases and how a real pipeline uses the holdout.
- Production considerations: new "Monitoring the evaluation system itself" paragraph with alerts and the
  "no release" degraded mode.
- Common mistakes: correlated cases counted as independent. Exercise K7 (cluster bootstrap) plus solution.
  Key takeaway on statistics updated. Test counts updated to 58.

### Chapter 25
- Removed source-material and Case Study references (replay counterfactual, cheaper-classifier decision,
  evaluation platform now points to Ch 36 Case E by name).
- New subsection "Multi-turn conversations": replayed transcripts vs simulated users, their biases,
  calibration, and conversation as group key (links to Ch 24 cluster bootstrap).
- CI section: new "Safety suites in the same pipeline" paragraph (Ch 27 `guardrails-measure
  --max-effect-bypass 0.0`, quality and safety gates kept separate).
- Baselines and pins: `require_baseline` paragraph; `gates.toml` excerpt re-pasted; new failure mode
  "Gate passes with a lost baseline"; gate tests paragraph lists the new exit-code case.
- Online evaluation: new sampled-scoring paragraph (reference-free evaluators on production samples,
  three alert classes, privacy of online judging).
- Listings: `replay.py` and `CanaryMonitor` now match disk exactly. Test counts updated to 59.
- Key takeaways: multi-turn bullet; missing baselines added to the CI bullet.
- ch25 solutions: removed the "Case Study 6" reference.

### Chapter 26
- Removed all source-material/Recipe references, including exercise K1; replaced em dashes.
- How it works: STRIDE defined with AI readings, risk scale explained.
- Jailbreaks: multi-turn escalation, many-shot priming, cross-modal and cross-lingual carriers.
- New catalogue subsections: "Sensitive information disclosure and personal data" and "Harmful content and
  misuse" (moderation as a legitimate primary control for content harm, routing self-harm, invented
  commitments).
- Implementation: `threat_model.py` excerpt (Threat, ThreatModel) and the egress test with a path comment,
  both copied from disk.
- Production considerations: "Security telemetry" (fields to record, incident-class alerts, probing
  trends) and "Incident response" runbook.
- New section "Real-world incident patterns": eight generic patterns with broken assumption and the
  control that holds, linked to threat ids, plus lessons.
- Evaluation and testing: new "The red-team plan" subsection (scope, effect objectives, personas,
  technique matrix, instrumentation, metrics, closure, cadence).
- Agent threat table re-ordered to match `by_risk()`.
- Exercises K7 (capability combinations; moderation vs effect controls) and E5 (first hour of an incident)
  plus solutions. Key takeaways: incident patterns bullet; red-team plan folded into the testing bullet.

### Chapter 27
- Removed source-material and Workshop L references (chapter and solutions).
- Fixed the SQL cross-reference to Chapter 36, Case B.
- Production considerations: canary or tenant error starts the Ch 26 runbook; new "Approvals as a control
  that people operate" (approval fatigue, approver UX, rate limits on approvers, token expiry); new
  "Sessions and rates, not just requests" (per-session counters through `GuardContext.state`, tightening
  paths, gateway rate limits in Chapters 29 and 30).

## Second-pass findings

Re-read all new and changed sections in the four roles.
- Educator: new Ch 26 sections reference threat ids (R2, R4, A1, A7) that are defined earlier; the Ch 26
  jailbreak paragraph's pointer to session-level monitoring in Chapter 27 needed a target, which the new
  "Sessions and rates" paragraph now provides.
- Architect: all listings re-checked against disk with a subsequence checker (full files byte-equal,
  excerpts in order): no mismatches in Chapters 24-27.
- Educator: Ch 26 code directory had no run instructions; added `examples/ch26/README.md` and verified
  its commands and names.
- Style: no em dashes outside chapter titles; no remaining "source material", "Workshop", "Recipe", or
  "Case Study" references in the four chapters or their solutions.
- Structural check (`build_book.check_chapter`) passes for chapters 24-27.

Test results after all edits (from repo root, `-p no:cacheprovider`):

| Directory | Result |
|---|---|
| book/projects/evalkit | 58 passed |
| book/projects/examples/ch25 | 59 passed, 5 skipped |
| book/projects/examples/ch26 | 20 passed |
| book/projects/guardrails | 133 passed |
| evalkit dependents: examples/ch31, examples/ch33, p3-rag-assistant, p5-incident-agent, p6-research-team, ragkit | 44, 31, 39, 39, 30, 259 passed |

`build_book.py --check` exits 1 only because chapters 15 and 39 do not exist yet (outside this group).

## Remaining open items

- The `CaseResult` record does not carry case metadata, so `evaluate_gate` cannot derive cluster groups
  from a config key; callers pass `groups` explicitly. Adding `group_by` to `GateConfig` would need the
  runner to copy a group key into each result, a wider change to the run format that other packages read.
  Left as is.
- Approval-token expiry is described as guidance in Ch 27 but not implemented in `guardrails.tools`
  (Ch 16's `ApprovalManager` and Ch 38's `InterruptManager` own approval lifecycles). Not changed, to avoid
  duplicating those owners.
- Ch 26 is about 10.8k words including code, above the 7-8k brief, because the brief's missing items
  (incident patterns, red-team plan) were added. No content was cut to compensate, since nothing in the
  chapter repeated another owner's material.
- Ch 28's "Chapter 27 implements JWT validation" (REVIEW TODO in integration notes) is false but belongs
  to Ch 28's reviewer; not edited here.

## Follow-up: capstone integration mismatches (#2, #3, #9, #11)

Source: `book/capstone/northwind-assist/README.md`, "API mismatches". All changes are backward compatible
at the API level (no public name removed or renamed; new keyword arguments and fields have defaults).

### #2 `guardrails.presets.support_tool_rules` vs Project 4
- Root cause: the preset used argument names `q` and `id` and had no rules for `get_service_status` and
  `create_ticket`, so the fail-closed tool check blocked every valid Project 4 call except replies, and
  replies lacked P4's required `ticket_id` and `subject`.
- Fix: rules rebuilt from `p4-support-assistant/support_assistant/tools.py`. All six tools have rules.
  Argument names, required fields, and limits follow P4 (`query` 100/200 chars, `status` enum, `subject`
  120/150, `body` 2000/4000, `ticket_id` pattern, priority enum, recipient domains). The new keyword
  `require_send_approval=True` lets a stack whose tool layer already owns approvals (capstone) avoid
  asking twice. `TICKET_ID_PATTERN` is exported.
- Tests: `tests/test_tools.py` updated to P4 argument names. New `tests/test_p4_alignment.py` builds P4's
  real registry (`wiring.build_container().registry.specs()`). It asserts that every P4 tool has a rule,
  that required args equal P4's required fields, and that constraint and rehydrate keys are real
  properties. It runs a valid call per tool through the preset and checks that bad ticket ids and
  priorities are rejected. The module skips if P4 is not installed.

### #3 `RedactingTracer` scrubbed too late for OpenTelemetry
- Root cause: scrubbing only happened in `export()`. Used as the sink of Chapter 31's `OTelAITracer`, the
  OTel span had already received raw attributes in `_backend_end`. Used as a wrapper, the old class
  opened plain `aie_core` spans, which broke Chapter 31's trace tree and lacked `capture`.
- Fix (`guardrails/telemetry.py`): `RedactingTracer.span()` delegates to the wrapped tracer's `span()`
  and scrubs the initial attributes. It overrides the span instance's `set_attribute` and
  `record_exception`, so writes are scrubbed. When the body exits it scrubs attributes, events, and
  Chapter 31's `pending_content`; that runs inside the wrapped span, before its `_finalize`,
  `_backend_end`, and sink export. `export()` still scrubs hand-built spans such as the streaming path.
  `__getattr__` delegates tracer-specific API (`capture`, `capture_policy`, ...). New public constant
  `DEFAULT_DROP_KEYS`.
- Tests: two new tests in `tests/test_telemetry.py`. The first uses a Chapter 31 `AITracer` subclass
  that records attributes and events at `_backend_end`, with capture mode `full`. The second uses the
  real `OTelAITracer` with an in-memory OTel exporter. Both assert that no e-mail, card, or key reaches
  the backend or the sink, covering direct `attributes[...]` writes, `set_attribute`, captured content,
  and exception messages. The trace tree and error status are preserved.
- Ch 27: the secrets/telemetry paragraph explains wrap-not-sink; the ch27 D3 solution gains the
  sink-placement cause; one new common-mistakes bullet.

### #9 Ch 25 `NORTHWIND_TOOLS` vs Project 4
- Root cause: `create_ticket` took `title` and a model-supplied `tenant`, and the reply tools lacked
  `to` and `subject`. The evaluator therefore checked a contract the runtime does not enforce, and
  treated the tenant as a model decision, which Project 4 rightly does not allow.
- Fix: the catalog copies P4's names, required fields, enums, and patterns. Two differences are
  documented in the code: `service` stays a free string for the incident desk's adapter names, and
  `query_metrics` is kept as the running example's analytics tool. The tenant is no longer an argument:
  the sandboxed `create_ticket` in `data/build_agent_runs.py` stamps the requester's tenant, taken from
  the task's `tenant:` tag, and reports it in the result. `northwind_projection` takes the tenant from
  the tool result, so it is never a model value. Planner and tool-selector stand-ins, P-103 and P-104
  scripts, `agent_tasks.jsonl` (`tenant` dropped from `expected_args`; `send_reply` expected
  `ticket_id`/`to` added) and `tool_cases.jsonl` (TU-004, TU-012) updated. The canned search results use
  `subject`.
- Regenerated via the documented commands: `python data/build_agent_runs.py` (event logs and
  trajectories), then `python ci/run_suite.py --system baseline --suite agent|tools --out ci/baselines`.
- Pins: agent `ff705d2df50a` -> `b609f01100c1`, tools `109adcfeff9b` -> `5785c798ec86` in
  `ci/gates.toml`, the chapter listing, and `test_edited_dataset_breaks_the_pin`. The classification and
  extraction pins (shared-data) are unchanged, and no shared-data file was touched.
- Verification: suite metrics, failing case sets, and the regressed gate summary were diffed against a
  snapshot taken before the change. They are identical for baseline, candidate, and regressed. Gate
  exits: 0 for baseline, 0 for candidate, 1 for regressed. Every number in the Ch 25 walkthrough
  therefore still holds.
- New test `test_catalog_matches_project4_tool_contracts`, which skips without P4. Two tests were
  updated to the new argument names.
- Ch 25: event, trajectory, and task-spec JSON excerpts updated (new action key `6829a3c320480262`); the
  replay diagram updated; a new paragraph explains why the catalog mirrors P4 and why the tenant comes
  from the request; test counts updated.

### #11 PII tokens vs tool argument patterns
- Design: re-hydrate inside the tool boundary, never in the model's context. `ToolRule.rehydrate_args`
  (new field, default empty) names the arguments that may carry tokens: `to` on reply tools, `query` on
  `lookup_employee`. `rehydrate_arguments(call, ctx, policy, args)` swaps tokens for vault values per
  argument and per PII kind. `guard_tool_call(pipeline, call, ctx, policy=rehydrate_kinds("email"))`
  re-hydrates the rule's arguments, runs the TOOL stage on the re-hydrated call, and returns that call
  for execution. As a result:
  - constraints, the outbound PII scan, and the policy engine judge real values;
  - the approval token binds to the real recipient;
  - invented, foreign, or other-tenant tokens stay tokens and fail closed;
  - body text is not re-hydrated by default.
- `token_tolerant_schema(parameters)` builds the model-facing schema copy whose string patterns also
  accept tokens, for strict structured-output modes. The original schema is not mutated.
- All new names are exported from `guardrails` and documented in its README.
- Tests in `tests/test_p4_alignment.py`:
  - a raw token fails P4's schema and the recipient rule;
  - `guard_tool_call` re-hydrates, approval binds to the real value, and the executed call validates
    against P4's real args model;
  - re-hydration is scoped by argument, kind, forged token, and tenant (`PermissionError`);
  - the token-tolerant schema accepts tokens and real addresses but not junk.
- Ch 27: new subsection "PII tokens at the tool boundary" with a listing copied from disk, a new
  common-mistakes bullet, updated directory tree, and test counts 133 -> 144.

### Test results (from repo root, `-p no:cacheprovider`)

| Directory | Result |
|---|---|
| book/projects/guardrails | 144 passed |
| book/projects/examples/ch25 | 60 passed, 5 skipped (eval-marked: 5 passed) |
| book/projects/examples/ch26 | 20 passed |
| book/projects/examples/ch31 | 48 passed |
| book/projects/p3-rag-assistant | 50 passed |
| book/projects/p4-support-assistant | 17 passed |
| book/capstone/northwind-assist | 62 passed |
| book/projects/evalkit | 58 passed |

The guardrails measurement table in Ch 27 (FP and bypass rates, 0/5 effects with guardrails) reproduces
unchanged. Chapters 24-27 pass `check_chapter`, every listing matches disk, and there are no em dashes
in prose.

### Open items from this follow-up
- The capstone still carries its own workarounds for #2, #3, #9, and #11: `capstone_tool_rules`, the
  `_Redacting` mixin, a catalog derived from the registry, and its own `_rehydrate` with
  `model_facing_schema`. They keep working, and capstone tests pass. The capstone owner can now replace
  them with `support_tool_rules(require_send_approval=False)`, `RedactingTracer(tracer)`,
  `guard_tool_call`, and `token_tolerant_schema`, and update the README's mismatch list. Not edited here
  because the capstone is outside group F.
- Project 4 does not call `guard_tool_call` itself (it does not tokenize input), so no P4 change was needed.
