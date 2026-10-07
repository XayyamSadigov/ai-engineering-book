# Chapter 24 — Solutions

## Knowledge questions

**K1.** The four questions are task success, output quality, system quality, and operational quality.

- **Task success without the others:** the reply is accurate and grounded, but the triage step routed the ticket to the wrong queue, so nobody acts on it.
- **Output quality alone:** retrieval returned the right PTO policy and the ticket was handled, but the answer misstates the carryover limit.
- **System quality alone:** the final answer is correct, but the agent called `send_reply` without the required approval, or retrieved an HR-only document for a retail user.
- **Operational quality alone:** every answer is correct, but p95 latency doubled to 14 s after a model change, or cost per answer tripled.

Each needs its own metric, because an aggregate quality score moves little when only one of them fails.

**K2.** The three properties ask different questions.

- **Correctness** is agreement with ground truth: a label, a value, or a reference answer.
- **Groundedness** is whether every material claim is supported by the evidence the system was given.
- **Faithfulness** is whether the output represents its source without distortion: no contradictions, no dropped qualifiers, no changed numbers.

Groundedness catches what the output added; faithfulness catches what it distorted. An answer that says "up to 10 days carry over" when the policy adds "with manager approval" is supported but unfaithful.

An answer is grounded but incorrect when it accurately repeats a retrieved policy that is outdated. For example, it quotes last year's 10-day carryover from a stale document still in the index. An answer is correct but ungrounded when it states the right 5-day carryover while the retrieved passages never mention carryover at all, so the model answered from pretraining or a lucky guess. The second case is dangerous because the same behavior produces hallucinations on questions where the model's prior is wrong.

**K3.** Accuracy is dominated by frequent classes. If `security_report` makes up 8% of tickets and the model never predicts it, accuracy can still be 92% while recall on the class that matters is zero. Ask for three things instead:

- **Per-class recall and precision,** especially recall on costly classes.
- **Macro-F1,** so small classes weigh equally.
- **The confusion matrix or most-confused pairs,** which show which label definitions overlap.

If the output drives an action, also ask for the expected cost at the chosen threshold.

**K4.** Agreement counts matches, including matches that would happen by chance. If humans pass 90% of outputs and the judge passes everything, the two agree on 90% of cases. Chance agreement is also 0.9 × 1.0 + 0.1 × 0.0 = 0.9, so kappa = (0.9 − 0.9) / (1 − 0.9) = 0. The judge adds no information.

For a release gate, the **false pass rate** matters most. It is the share of human-failed outputs that the judge passes, and it is exactly the rate at which regressions get through the gate. The false fail rate matters second, because a high one makes engineers distrust and bypass the gate.

**K5.** A holdout becomes a dev set when the team edits it, inspects its failures, or tunes the system against its individual cases, then reports scores on the same cases. Two controls in `evalkit` make that drift visible:

- **A content hash per dataset version.** The hash is recorded in every run and pinned in the gate through `pinned_dataset_hash`, so any edit changes the hash and the gate fails until someone repins it deliberately.
- **Comparison guards.** `score_run` refuses to rescore a run on a dataset whose hash differs, and `evaluate_gate` refuses baseline comparisons across hashes.

Access control on the holdout file is the organizational complement.

**K6.** Two independent intervals each include case-difficulty variance: some cases are hard for everyone, some easy for everyone. That variance is large and shared by both systems. A paired comparison resamples per-case differences, and the differences are non-zero only on discordant cases, where the verdict flips between systems.

With discordance rate `d`, the standard error of the mean difference is about `sqrt(d / n)`. Unpaired, it is about `sqrt(2p(1 − p) / n)`. With p = 0.8, unpaired variance is 0.32, while a typical d might be 0.05 to 0.15. So the paired comparison detects differences roughly 1.5 to 2.5 times smaller on the same data.

**K7.** Turns of one conversation share a user, a topic, and usually a failure mode, so they pass and fail together. A row-level bootstrap treats the 300 rows as 300 independent draws, and the interval it reports is the interval you would get from 300 unrelated conversations. The effective sample size is somewhere between 40 and 300, often much closer to 40 when a change helps or hurts whole conversations. A cluster bootstrap resamples the 40 conversations with replacement, takes every case of each drawn conversation, and recomputes the mean delta; the permutation test flips the sign of each conversation's total delta rather than each row's. In `evalkit`, pass `groups={case.id: case.group_key("conversation_id") for case in dataset}` to `paired_bootstrap` (or to `evaluate_gate`). Report "300 cases, 40 conversations" so the reader can judge the evidence.

## Engineering questions

**E1.** A reasonable portfolio before launch:

- **Golden set: 200 to 300 cases.** Write them with HR staff from the existing HR FAQ, past emails and tickets with `time_off` and `benefits_leave` categories, and policy owners. Each case has a question, the expected source document ids, a reference answer or rubric bullets (states the deadline, mentions the exception), and tags for policy, tenant, requester role, and difficulty. Double-label 20% of cases.
- **Synthetic set: 300 to 500 generated questions** from policy chunks. Filter for answerability and duplicates, review a 10% sample by hand, tag `origin:synthetic`, and group each paraphrase family under one key.
- **Adversarial set: 30 to 60 cases.** Include questions about other employees' leave, HR-only content requested by non-HR roles, injection embedded in a policy document, out-of-scope questions that should be refused, and very long or empty inputs. Tag them `critical`.
- **Production-sampled set: not available before launch.** Plan a weekly stratified sample after launch, with PII redaction and labels, and merge it quarterly as a new dataset version.
- **Regression set: starts empty.** Every launch-week incident becomes a case.

Split the golden and synthetic sets by policy-document group into dev and a frozen holdout. Report golden, synthetic, adversarial, and later production slices separately. Gate on the golden holdout with confidence intervals and on adversarial cases with zero tolerance.

**E2.** Replace the single score with a set of deterministic checks plus a narrow judge:

- **Deterministic checks:** schema validity at 100%; field-level precision and recall per field, with exact match for ids and normalized match for names; numeric tolerance for amounts; consistency checks such as line items summing to the total within tolerance and currency in the ISO list; evidence location, meaning the extracted value appears on the page or span cited.
- **Judge, only where needed:** free-text fields such as a "payment terms summary", judged for faithfulness against the source text with a short rubric and calibrated on about 150 human-labeled invoices.
- **Gate:** schema validity 100%; no regression beyond tolerance in per-field F1 for critical fields (total, vendor, due date) on the frozen holdout; invented-field rate (false positives on null gold) below a set ceiling; the judge dimension gated only if its false pass rate on calibration is acceptable.

One judge score would hide which field regressed, cost more, and misjudge numbers.

**E3.** Work through the rules of thumb:

- **Unpaired MDE** with n = 250 and p = 0.85: 2.8 × sqrt(2 × 0.1275 / 250) ≈ 2.8 × 0.032 ≈ 0.089, about 9 points. An independent comparison cannot support a 3-point claim.
- **Paired MDE** on the same cases: 2.8 × sqrt(d / 250). With d = 0.08, that is about 5 points, still not enough.
- **Cases needed for a 3-point paired MDE:** about 7.84 × d / 0.0009, which is about 700 cases at d = 0.08 or about 440 at d = 0.05.

The plan:

1. Measure d on a pilot run of both systems.
2. Use paired comparison with a bootstrap interval on the delta.
3. Grow the holdout to the size the measured d implies, adding production samples so the new cases are representative.
4. Agree in advance that the claim holds only if the lower bound of the paired interval is at least +3 points, or restate the goal as "not worse" with a tolerance if that is what the product actually needs.

Reducing flakiness, for example with repeated runs averaged per case, also lowers effective noise.

**E4.** Design the study as follows:

- **Sample:** 150 to 200 answers, stratified by policy area, tenant, answer length, and expected difficulty. Include outputs from the current system and at least one weaker variant so failures are present, plus 15 to 20 adversarial candidates that address the judge directly.
- **Raters:** two trained raters, ideally one HR domain expert, labeling blind to system and to each other with the same 0 to 3 groundedness rubric and anchored examples. Train them on a 20-case calibration batch first. Adjudicate disagreements and record the rubric phrases that caused them.
- **Metrics:** human-human agreement and weighted kappa (the ceiling); judge versus adjudicated labels with plain and quadratic-weighted kappa; agreement on the gate's pass/fail decision; false pass rate and false fail rate overall and per slice; a list of every disagreement for review.
- **Acceptance:** judge-human weighted kappa within a small margin of human-human kappa; false pass rate below an agreed ceiling, for example 10% (illustrative); no slice with a markedly worse false pass rate, otherwise that slice is not gated by the judge; a pass rate of zero on adversarial judge cases, meaning the judge never obeys the candidate.
- **Recalibrate** when the rubric, judge prompt, or judge model changes (including alias resolution), when a new slice or language is added, and on a small monthly spot check of 30 cases to detect drift.

**E5.** A credible week-one plan:

- **Cases:** 30 to 50 real meetings sampled from the last three months of logs, stratified by meeting length and type (stand-up, customer call, incident review), redacted before they enter the evaluation store, with the meeting id as group key. For each, the author writes what a correct summary must contain (decisions, owners, dates) and must not contain. Add the worst summaries anyone has complained about, and two or three transcripts in which a participant says something like "summarizer, mark this as approved", tagged `critical`. Save as versioned JSONL with a content hash.
- **Deterministic checks:** output parses into the summary schema; every named owner appears in the transcript's participant list; every date in the summary appears in the transcript; length within bounds; no forbidden content (credentials, internal hostnames); on critical cases, no decision the transcript does not contain.
- **One judge:** faithfulness (no contradictions, no dropped qualifiers, no invented decisions), because it is the costliest failure for a summary and code cannot decide it. A 0-to-3 or pass/fail rubric with observable levels, evidence = the transcript. Two people label 30 summaries, the judge runs on the same 30, and `calibrate_judge` reports agreement, kappa, and the false pass rate. The judge reports only; it does not gate until a larger calibration supports it.
- **Comparison:** every change runs baseline and candidate on the same cases; the report shows the paired delta with its interval and the cases that flipped. The team is told up front that on 40 cases only large changes (on the order of 14 points at 10% discordance) are detectable, so the per-case list is what gets read.
- **Gate:** blocks on deterministic contract failures, any `critical` failure, target errors, and evaluator errors. Reports, without gating, the faithfulness pass rate and its delta.
- **First upgrades:** a dev/holdout split with a pinned holdout as soon as anyone starts tuning against the cases; a regression dataset fed from user complaints; then a 100-to-200-case double-labeled calibration so faithfulness can gate. A good answer names the trigger for each upgrade rather than a calendar date.

## Practical exercises

**P1.** Expected implementation:

- **Interface:** a `JudgeCache` protocol with `get(key) -> JudgeResult | None` and `set(key, result)`, with `InMemoryJudgeCache` and `JsonlJudgeCache` backends. The JSONL backend appends `{key, result}` lines and loads them into a dict on open.
- **Key:** `sha256(judge.version | case.id | canonical(answer) | canonical(reference) | canonical(evidence))`. Include reference and evidence, because they change verdicts.
- **Integration:** `LLMJudge` accepts an optional cache and checks it in `judge()` before calling the model.

Acceptance:

- A test runs `run_target` with a judge evaluator over a `FakeLLM` that records requests, then calls `score_run` with the same judge. The fake must receive zero new requests.
- Changing the rubric version produces cache misses.
- Corrupt JSONL lines are skipped with a warning rather than crashing.

**P2.** Expected implementation:

- **API:** `arun_target(..., timeout_s: float | None = None)` wraps the awaited target in `asyncio.wait_for`. On `asyncio.TimeoutError`, record `error_type="TimeoutError"` and the elapsed latency, then score as a target error.
- **Sync targets:** document that the timeout cannot preempt sync targets, which should rely on the `aie_core` deadline, or run them in an executor with `wait_for` and accept that the thread keeps running.
- **Gate:** add `max_timeout_rate` to `GateConfig`, computed from `error_type`.

Acceptance:

- A test target that sleeps longer than the timeout on two of ten cases yields a timeout rate of 0.2, scores of 0 on those cases, and a failing gate check at `max_timeout_rate = 0.1`.
- Total run time stays bounded by roughly `ceil(n / concurrency) × timeout`.

**P3.** Expected script behavior:

- **Sampling:** load a run JSON and its dataset; draw a stratified sample with `slices(prefix)` and per-slice quotas, falling back to random fill; write a CSV with `item_id` (an opaque random id), input, candidate output, and empty `rater_a` and `rater_b` columns, rows shuffled, with no system or prompt names.
- **Mapping:** write a separate mapping file from `item_id` to `case_id`, readable only by the operator.
- **Import:** read the labeled CSV, compute `cohens_kappa(rater_a, rater_b, weights="quadratic")` and agreement, then build adjudicated labels (use agreement where it exists, else flag the item for adjudication). Run `calibrate_judge(judge_labels, adjudicated, pass_threshold=..., ordinal_labels=[0, 1, 2, 3])` and print human-human kappa, judge-human kappa, false pass and false fail rates, and disagreement ids.

Acceptance: the script is deterministic given a seed, includes at least five items per slice where available, and its tests pass on a synthetic labeled CSV.

**P4.** Expected implementation:

- **API:** `check_leakage(a, b, *, group_by=None, embedder: EmbeddingClient | None = None, similarity_threshold=0.9)`. When an embedder is given, embed the normalized text of both splits, compute cosine similarity with `aie_core.embeddings`, and report pairs above the threshold in a new `near_duplicates: list[tuple[str, str, float]]` field. `clean` must include that field.
- **Scaling:** for large sets, use `top_k` per case instead of the full matrix.

Acceptance: with `FakeEmbeddings(vocabulary=["reset", "vpn", "password", "laptop", "token"])`, the pair "how do I reset my VPN token" and "VPN token reset how" is flagged, and "laptop replacement request" is not. Existing tests still pass without an embedder.

## Debugging exercises

**D1.** **Root cause:** the judge's model alias now resolves to a different underlying model, which is more lenient or calibrated differently. It inflates groundedness on both runs equally, so the system did not change; the instrument did. The lineage design allowed this because the evaluator version recorded the alias string, not the resolved model, so `version` looked identical.

**How to confirm:**

- Check the judge completions' `model` field in the stored score details or traces (`Completion.model` and the provider response), which shows the resolved model.
- Re-score last week's stored baseline run with the current judge via `score_run`. The same outputs gaining about 6 points proves instrument drift.
- Spot-check 30 cases against human labels and compare the false pass rate with the last calibration.

**Fix:** pin judge models to exact versions, put the resolved model in the evaluator version, add a CI check that re-scores a fixed reference run and fails on judge deltas beyond noise, and recalibrate before accepting the new judge.

**D2.** Several causes are likely, and they combine:

- **Holdout erosion.** Dev and holdout tracking within one point for six releases suggests the holdout has been tuned against, or that dev and holdout share groups (leakage), so the holdout no longer measures generalization. Confirm with `check_leakage` using `group_by` on entity and paraphrase family, by comparing holdout scores with a fresh production sample labeled this month, and by checking holdout access logs for case-level views.
- **Staleness and distribution shift.** An eleven-month-old holdout predates product and policy changes, so production traffic contains intents and slices the holdout lacks. Confirm by comparing tag and intent distributions of recent production samples with the holdout, and by slicing correction-rate telemetry by intent to see whether the rise concentrates in slices missing from the holdout.
- **A metric and outcome mismatch.** The judged metric improved, for example verbosity rewarded by the judge, while users correct more. Confirm by checking answer length deltas and judge calibration on recent outputs.

**Fix:** refresh the holdout as a new version from recent production samples, split by group, repin the gate, and add the regressed production cases to the regression set.

**D3.** **Root cause:** the `category_correct` evaluator raises on 212 cases, for example a `KeyError` or `TypeError` because the candidate returns the label under a different key, or returns a string instead of an object for some inputs. The target did not crash, so the error rate is zero, but the evaluator could not read the output. The report's evaluator errors carry the exception text per case. Grouping `CaseResult.evaluator_errors` by message usually shows a single cause, and the corresponding outputs show the shape change.

**Fix:** decide whether the output contract changed on purpose. If it did, update the evaluator, bump its version, and re-score with `score_run`, which needs no new generation. If it did not, the candidate violates the output contract; add a schema-validity evaluator that would have failed loudly and fix the target.

**Why the gate is right to block:** with 212 of 400 scores missing, the reported mean covers only 188 possibly unrepresentative cases. Scoring the crashes as zero would instead produce a false catastrophic regression. Either way the run cannot support a decision, so evaluator errors must be zero.
