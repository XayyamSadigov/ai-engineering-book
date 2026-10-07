# Exercises — Chapter 32 — Engineering Practices for AI Systems

Solutions: `../solutions/ch32-solutions.md`


**Start here:** K3, K5, E2, P2, D1 (about 3 hours). The rest go deeper.

### Knowledge questions

**K1.** State the dependency rule of clean architecture and name, for each of the four layers in the triage example, one thing it may import and one thing it may not.

**K2.** What is the difference between a provider abstraction such as `aie_core` and an anti-corruption layer such as `LLMClassifier`? Why does a system need both?

**K3.** Explain why a mock of a provider SDK can keep passing after a breaking SDK change, while a recorded HTTP fixture fails. What does each still fail to tell you?

**K4.** Why must percentage rollout be deterministic by a stable unit and monotonic under ramping? What goes wrong in analysis if either property is missing?

**K5.** A team says its 5% canary "proved" that a new prompt improves routing accuracy by two points. What is wrong with the claim?

**K6.** List the artifacts a version manifest should contain for a RAG answer service (not triage), and explain why the embedding model and the index version must be recorded together.

### Engineering questions

**E1.** Northwind wants to replace the LLM classifier with a fine-tuned small model for categories it handles well and keep the LLM for the rest. Describe the changes by layer and file, and which tests change.

**E2.** Design the experiment plan for switching the triage model: hypothesis, primary metric, guardrails, randomization unit, sample size reasoning at an illustrative 1,200 tickets per day with an 80% baseline and a three-point minimum effect, duration, stop conditions, and rollback.

**E3.** Your organization forbids provider keys in any CI job triggered by a pull request. Design an offline evaluation strategy that still catches prompt regressions before merge, and state what it cannot catch.

**E4.** Write the code review checklist you would apply to a pull request that changes a prompt, a tool schema, and the parser in one change. What would you ask the author to split, and why?

### Practical exercises

**P1.** (about 2 hours) Add an `embedding_model` and `index_version` to the triage manifest by introducing a retrieval port that fetches similar past tickets as few-shot examples. Record both on spans and add a test that a changed index version changes the fingerprint and appears in `changed_components`.

**P2.** (about 90 min) Extend `FlagEvaluator` with a tenant-level override so that the `logistics` tenant can be excluded from an experiment entirely, regardless of user bucket. Add tests for precedence (kill switch, environment, tenant exclusion, user override, allocation).

**P3.** (about 2 hours) Add a `--slice tenant` option to the eval gate that reports and gates per-tenant accuracy, failing if any tenant regresses by more than the tolerance even when the aggregate improves.

**P4.** (about 60 min) Write a hypothesis property test for `RecordReplayTransport.key_for`: reordering JSON keys and changing ignored fields never changes the key; changing any non-ignored field always does.

### Debugging exercises

**D1.** After a release, the triage dashboard shows the `treatment` and `control` arms of the prompt experiment with identical category distributions and identical token counts for two weeks. The flag file shows 50% treatment. Spans show `version.flags.triage.prompt = treatment` on half of the requests. Diagnose the cause and name the check that would have caught it at startup.

**D2.** CI starts failing in the unit stage with `CassetteMiss: no recording for POST /v1/chat/completions`. The pull request only edits the docstring of `TriageDecision` and a comment in the prompt store. Explain the failure, decide whether it is a bug, and describe the correct fix.

**D3.** The nightly drift job fails on `recall.security_report` (from 0.83 to 0.67). No commits landed in a week. The manifest fingerprint of the nightly report is identical to the baseline's. Spans from production over the same week show `llm.served_model` changing from one dated identifier to another on Tuesday. What happened, what is the immediate mitigation, and what would you change so that this is caught on Tuesday rather than days later?
