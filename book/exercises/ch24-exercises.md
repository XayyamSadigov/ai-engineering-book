# Exercises — Chapter 24 — Evaluation Fundamentals

Solutions: `../solutions/ch24-solutions.md`


### Knowledge questions

**K1.** Name the four questions every evaluation should keep separate, and give one Northwind Assist failure that affects each without necessarily affecting the others.

**K2.** Define groundedness, faithfulness, and correctness, and describe an answer that is grounded but incorrect, and one that is correct but ungrounded.

**K3.** A triage classifier reports 92% accuracy. Explain why this number may be meaningless, and list the three metrics you would ask for instead.

**K4.** Explain why a judge that agrees with humans on 90% of cases can have a Cohen's kappa of zero, and which pass/fail error rate matters most when the judge feeds a release gate.

**K5.** What turns a frozen holdout into a development set? Name two technical controls in `evalkit` that make that drift visible.

**K6.** Why does a paired comparison on the same cases detect smaller differences than comparing two independent confidence intervals? Use the discordance rate in your answer.

**K7.** A multi-turn support dataset has 300 cases drawn from 40 conversations. Explain why a row-level bootstrap overstates confidence in a measured gain, what a cluster bootstrap resamples instead, and which `evalkit` argument enables it.

### Engineering questions

**E1.** Design the dataset portfolio for Northwind's HR policy assistant before launch: which of the five dataset types you can build now, which must wait for traffic, approximate sizes, slice tags, and what you will report separately.

**E2.** The extraction team wants a single "invoice quality" score from an LLM judge. Propose an evaluation design instead, stating which properties are deterministic, which need a judge, and what the gate requires.

**E3.** A product manager asks you to prove that a new model improves answer quality by "at least 3 points". Your golden set has 250 cases with a pass rate near 85%. Using the rules of thumb in this chapter, say whether the current set can answer the question, and what you would change.

**E4.** Design the calibration study for a groundedness judge that will gate releases of the RAG assistant: sample, raters, guidelines, metrics, acceptance criteria, and recalibration triggers.

### Practical exercises

**P1.** Add a `JudgeCache` to `evalkit` that stores `JudgeResult` objects keyed by judge version, case id, and a hash of the candidate output, with an in-memory and a JSONL backend. Show with a test that re-scoring an unchanged run makes no judge calls.

**P2.** Extend `run_target` with a per-case timeout for async targets that records a `TimeoutError` as a target error, and add a gate rule for a maximum timeout rate.

**P3.** Write a `tools/label_sample.py` script that draws a stratified calibration sample from a run (at least five cases per slice where available), exports it as a CSV for two raters with system identity hidden and order randomized, imports the labels, and prints a `calibrate_judge` report including the human-human kappa.

**P4.** Add an embedding-based near-duplicate check to `check_leakage` using `aie_core` embeddings, with a similarity threshold parameter. Test it with `FakeEmbeddings(vocabulary=...)` so that a paraphrase pair is caught and an unrelated pair is not.

### Debugging exercises

**D1.** After switching the judge configuration to a new model alias, the nightly report shows groundedness up 6 points on every slice for both the baseline and the candidate, and no code changed in the assistant. The run lineage shows the same evaluator version string as last week. Diagnose what happened, what in the lineage design allowed it, and how to confirm it.

**D2.** A prompt change shows a +4.1 point gain on the holdout with a paired interval of [+1.2, +7.0]. In production, correction rates rise the week after release. The holdout was last versioned eleven months ago, and the dev and holdout scores have tracked within one point of each other for the last six releases. Diagnose the likely causes and the telemetry that would confirm each.

**D3.** The release gate blocks a candidate on "evaluator errors: 212". The report shows `category_correct` missing for 212 of 400 cases and the rest scoring normally. The target's error rate is zero. Diagnose the cause and the fix, and explain why the gate is right to block.
