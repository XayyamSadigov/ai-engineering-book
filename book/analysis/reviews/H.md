# Review group H: Chapters 35-36 and front/back matter

Scope: `chapters/35-system-design-method-and-cases-1.md`, `chapters/36-system-design-cases-2.md`,
`solutions/ch35-solutions.md`, `solutions/ch36-solutions.md`, `projects/examples/ch35`,
`projects/examples/ch36`, `00-learning-roadmap.md`, `appendix-c-interview-preparation.md`,
`glossary.md`, `references.md`, `README.md`.

## Gaps found (by reviewer role)

### Principal AI Engineer
- All seven application cases (knowledge assistant, support copilot, document processing, coding
  assistant, research agent, analytics assistant, workflow automation) and both platform cases cover
  all ten steps, each with numbered scaling arithmetic and a failure table. No step missing.
- Neither chapter mapped design boxes to the book's packages and projects (lead's requirement). A
  reader could not go from a design to the code that implements it.
- Arithmetic errors, recomputed by hand and by test:
  - Ch 35 Case 2: speech is described as "three quarters" of the voice bill; it is 1,200 of about
    1,800 USD, two thirds. The LLM line is 597.6, not 598 via "154".
  - Ch 35 Case 3: "200 × = 5 million" (typo); annual fine-tuning saving quoted as 120,000 USD
    (340 × 365); invoices are working-day traffic, about 85,000 over 250 days.
  - Ch 36 Case A: latency paragraph treated the 24 documents per report as 24 per iteration, so
    "24 s per iteration, about 3 minutes" did not add up.
  - Ch 36 Case B: "without plan deduplication about $30" (correct about $26); "65 to 80 percent cuts
    model cost by roughly a third" (correct: 0.024 to 0.018, a little over a quarter).
  - Ch 36 Case E: smoke run quoted at $4 for 2.25 M tokens at the chapter's rates (about $5).
  - Ch 35 solutions E2: "50 percent of the savings gone" (correct: about 80 percent); K6: annual
    saving at 365 days; D3: claimed cost doubling corresponds to 0 percent cached (that triples cost;
    doubling means the cached fraction fell from 0.8 to about 0.4).
  - Ch 36 solutions E4: "average resident KV rises roughly 60 percent" (correct about 40 percent with
    the chapter's 10 percent long-context mix).
- Ch 36 Case D argues KV-based admission but never checks the replica's resident KV at the operating
  point; the throughput sizing was not cross-checked against memory.

### Senior Software Architect
- Ch 36 code excerpt (sqlglot LIMIT step) differed in line wrapping from `sql_guard.py`; the public
  interface listing omitted `guard_sql` and `GuardResult.codes`.
- Ch 35 Case 2 `send_reply` idempotency key was the draft id alone, inconsistent with Chapter 16's
  content-bound approval (an edited draft would be deduplicated or approved by id).
- The book's `SandboxRunner` (Ch 16) is a process sandbox that does not isolate the filesystem and
  only blocks network with `network="deny"` on Linux; a naive mapping of Case 4's "no network,
  repo-only FS" to it would overstate it.
- `reliability.AdmissionController` (Ch 29) admits by concurrency count and per-tenant token
  quotas, not by KV tokens; Case D's "admission by KV budget" needs that caveat when mapped to code.

### AI Educator
- Ch 36 comparison table was titled "Seven cases compared" but listed nine, promised a separator
  "below the line" that did not exist, and used short names that did not match Ch 35/36 headings
  ("Knowledge assistant", "Support copilot", "Document processing", "Workflow automation").
- Its cost-driver column contradicted Ch 35's own conclusions (Case 2 "model calls per turn", Case 3
  "pages processed", Case 4 "iterations per task" versus prefix-cache hit rate, speech minutes, and
  review hours in the text).
- Ch 36 case headings used em dashes (style rule: em dashes only in chapter titles).
- Appendix C 2.10 outlines diverged from the chapters: Case D listed "file and shell tools" (Ch 35
  has no general shell) and hybrid code retrieval (Ch 35 launches without an embedding index);
  Case C said "invoices and tickets" (Ch 35: invoices and contracts); Case B gave a consented
  customer-profile memory (Ch 35: no cross-conversation memory, facts stay in the CRM); Case E
  described Project 6's supervisor team rather than Ch 36's planner, verifier, and budgets; Case F
  cached by normalized SQL (Ch 36: normalized question, tenant, freshness marker). Workflow
  automation and the two platform cases were missing.
- README said Part X applies the method to "seven systems" (nine).
- Roadmap claimed all chapters follow one template; Part X and the capstone adapt it.
- Glossary: every chapter pointer verified by grep (259 entries; all pointers resolve, except Ch 15
  and Ch 39, which are not on disk yet). Gaps: no entries for the ten-step method, architecture
  class, back-of-the-envelope sizing, SQL guard, result equivalence, `semsearch`, or the six
  projects; `ragkit` was described as ingestion and chunking only, although Ch 12-14 add retrieval,
  generation, and eval subpackages; Little's Law, self-consistency, admission control, and trust
  boundary lacked their Part X pointers.
- References: every paper, DOI, RFC, and URL checked against what I know with certainty; none judged
  doubtful, so none removed. The source acknowledgement (Amit Shekhar's course URL) is confirmed by
  the source text itself. Missing: a text-to-SQL evaluation reference and the sqlglot docs used by
  Ch 36's code.
- Roadmap and README chapter numbers, titles, and project placement (P1 Ch 6, P2 Ch 9, P3 Ch 15,
  P4 Ch 16, P5 Ch 20, P6 Ch 22, capstone Ch 39) match `book/chapters`, `book/projects`, and the TOC.

### Production/SRE Engineer
- Failure tables in both chapters already name detection signals and mitigations; degraded-mode
  ladders are explicit. The gap was traceability: no pointer from "detection signal" to the
  tracing, SLO, admission, and spend-guard code that produces it (closed by the package maps).
- The sizing module's tests verified only Cases 1 and 2, while the chapter claimed it reproduces
  every case's numbers.

## Changes applied

- `projects/examples/ch35/test_back_of_envelope.py`: four new tests reproducing Case 1 storage, Case
  2 voice (LLM line, speech share, per-call cost, 150 concurrent calls), Case 3 (model cost on both
  tiers, review cost, 80 USD per review point), and Case 4 (autocomplete and agent mode with and
  without prefix caching). No change to `back_of_envelope.py`.
- Ch 35: "Where each piece is built" table for each of the four cases, mapping every design box to a
  verified package, class, or project path (`aie_core`, Ch 5 `ContextBuilder`, Ch 7 `Router`,
  `ragkit` retrieval/generation/eval, `semsearch`, Projects 1/3/4, `toolkit`, Ch 17 workflow engine,
  `reliability`, Ch 25 `taskevals`, `guardrails`, Ch 30/31 examples, Ch 37 `CodeIndex`, Ch 38
  `voice_gate`, `coding_harness`, `DurableRunner`); intro and sizing-module text updated; voice cost
  line and "two thirds" corrected; contract typo and annual saving fixed; `TurnGate` (Ch 38) cited
  as the code form of the no-write-from-partial rule; Case 4 tied to Ch 38's coding harness;
  `SandboxRunner` mapped with its real limits; `send_reply` key made content-bound.
- Ch 36: "Where each piece is built" table for each of the five cases (Projects 1/3/6, `agentkit`,
  Ch 37 `AgenticRAG`, `guardrails`, `evalkit`, Ch 25 gate and replay targets, Ch 17 engine, Ch 38
  `InterruptManager` and `ReconcilingTool`, `reliability` admission, deadlines and SLO burn rate,
  Ch 34 KV and load-test modules, Ch 32 canary); Project 6's benchmark result tied to the fan-out
  rule; `AdmissionController` caveat; latency arithmetic of Case A rewritten; Case B dedup and
  semantic-share numbers corrected; KV residency check per replica with Little's Law added to Case D
  (82 GB of 160 GB at 10 percent long contexts, KV-bound at 30 percent); Case E smoke cost corrected;
  sqlglot excerpt made byte-identical to disk; interface lists `codes` and `guard_sql`; case headings
  use colons; comparison table renamed "Nine cases compared", split into application and platform
  tables, names matching Ch 35/36 headings exactly, cost drivers aligned with the case text.
- Solutions: Ch 35 K6, E2, D3 and Ch 36 E4 arithmetic corrected with the derivation shown.
- Appendix C 2.10: outlines A to F rewritten to match the chapters (tools, memory, retrieval, numbers
  from step 9); G (enterprise workflow automation) added; H and I (serving and evaluation platforms)
  added as infrastructure prompts.
- README: Part X row now says nine systems and names the code artifacts. Roadmap: template caveat
  for Part X and the capstone; Part X row names the artifacts.
- Glossary: `ragkit` entry describes all four layers and adds Ch 15; new entries Architecture class,
  Back-of-the-envelope sizing, Projects (P1 to P6) with paths and package names, Result equivalence,
  `semsearch`, SQL guard, System design method (ten steps), placed alphabetically; Ch 35/36 pointers
  added to Little's Law, Self-consistency, Admission control, Trust boundary.
- References: Spider (Yu et al., 2018, arXiv 1809.08887) and the sqlglot repository added.

Tests: `.venv/bin/python -m pytest book/projects/examples/ch35 book/projects/examples/ch36 -q -p no:cacheprovider`
gives 80 passed (ch35 now 13 tests, ch36 unchanged). `build_book.py --check` reports only
"chapter 15 missing" and "chapter 39 missing"; `check_chapter` passes for Ch 35 and Ch 36.

## Second-pass findings

- All four roles re-run on the edited files. Found and fixed: the `send_reply` idempotency key (above),
  `SandboxRunner` overstated in the first draft of the Case 4 map, a misplaced glossary entry
  (Back-of-the-envelope landed under A), and a double colon in the Case C heading.
- Cross-references from other chapters to Ch 35/36 (Ch 9, 25, 30, 37) still resolve: they cite
  "Chapter 35, Case 1", "Case E (Evaluation platform)", the sizing module, and the SQL guard, all
  unchanged in name.
- No em dashes outside chapter titles in any file in scope.

## Remaining open items

- Chapters 15 and 39 and `book/capstone/northwind-assist` are not on disk yet (other authors). The
  roadmap's capstone acceptance checklist, the glossary's Ch 15/39 pointers, and the Ch 35 maps'
  references to Project 3 could not be verified against content; they match the TOC. Project 3 is
  referenced at directory level only, because its internals are still being written.
- `coverage-matrix.md`, referenced by README and the build script, does not exist; it is outside
  group H's scope.
- Ch 35 and Ch 36 are about 12,000 and 12,800 words against an 8-9k target. The overage comes from
  the case tables required by the 10-step treatment and the package maps; no filler identified to cut
  without losing a step.
