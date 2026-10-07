# Chapter 32 — Solutions

## Knowledge questions

**K1.** Source-code dependencies point inward: outer layers depend on inner layers, never the reverse, and inner layers declare what they need as interfaces (ports) that outer layers implement.
- Domain: may import the standard library and pydantic. It may not import `aie_core`, httpx, FastAPI, or any application or adapter module.
- Application: may import the domain and the pure libraries (`flags`, `version_manifest`). It may not import adapters or provider SDKs. It defines `TracerPort` instead of importing `aie_core.observability`.
- Adapters: may import anything, including `aie_core`, httpx, FastAPI, and application ports. They may not be imported by the domain or the application. Use cases must not call adapters directly.
- Composition root: may import every concrete class. Nothing else should import it except entry points such as the HTTP factory and the eval gate CLI.

**K2.** A provider abstraction hides wire-format differences between vendors. It is still LLM-shaped: messages, roles, tokens, finish reasons, and an LLM error taxonomy. An anti-corruption layer translates another model's concepts into the application's own concepts, so that `Completion` and `RateLimitError` never appear in use-case code. A system needs both. The provider abstraction avoids writing an HTTP client per vendor. The anti-corruption layer keeps the use case independent of the fact that an LLM is involved at all. A fine-tuned classifier, a rules engine, or a different vendor can then replace it by changing only the adapter and the composition root. The anti-corruption layer is also where LLM-specific failure semantics, such as `finish_reason = "length"`, become domain-meaningful errors.

**K3.** A mock returns an object shaped the way the test author believed the SDK behaves. When the SDK renames a field or changes an error class, nothing in the mock changes, so the tests still pass. A recorded fixture replays the actual bytes the provider returned, beneath the adapter. The adapter's real decoding then runs against them, and a decoding change surfaces as a failure. A changed request produces a `CassetteMiss`. Neither one tells you about output quality, because a recording is one sample from one day. A recording also cannot detect that the live provider changed after it was made. That is the job of the nightly drift evaluation and the integration re-record test.

**K4.** Determinism by a stable unit means a user always gets the same variant on every replica, retry, and replay. Without it, users see inconsistent behavior within a conversation. Analysis by user then mixes exposures, and replays of past traces cannot reproduce the assignment. Monotonicity means that ramping from 5% to 20% keeps the original 5% in treatment. Without it, users flip between variants. Early exposure data then cannot be pooled with later data, and carryover effects such as user learning contaminate both arms.

**K5.** A 5% canary is sized and designed to detect guardrail breaches, not small improvements. At an 80% baseline, detecting two points with conventional significance and power needs several thousand units per arm. A 5% arm accumulates that slowly, so the canary almost certainly lacks the power to support the claim. The comparison may also not have been pre-registered. It may have been read repeatedly, which inflates false positives. It may also have counted requests instead of users, which ignores correlation. The claim requires an A/B test with a planned sample size and duration.

**K6.** For a RAG answer service, the manifest should contain:
- code identity: app version and git sha;
- the answer prompt version and hash, plus any query-rewrite and judge prompts;
- the chat model, both requested and served;
- the embedding model;
- the index version, which covers corpus snapshot, chunker configuration, and embedding model;
- the reranker model, if any;
- retrieval parameters, such as k and hybrid weights;
- the policy and guardrail versions;
- the tool schema versions, if any;
- the evaluator and dataset versions, for evaluation runs;
- the flag assignments.

The embedding model and index version must travel together because vectors from different embedding models live in different spaces. A query embedded with model B against an index built with model A returns near-random neighbors without raising any error. The index is defined by the embedding model, so recording only one of the two cannot explain a retrieval regression.

## Engineering questions

**E1.**
- Add `adapters/small_model_classifier.py`, which implements `ClassifierPort` over the fine-tuned model's HTTP endpoint and maps its errors to `ClassifierUnavailable`.
- Add a routing adapter, `CascadeClassifier`, which also implements `ClassifierPort`. It calls the small model first and falls back to the LLM classifier when the predicted category is outside the supported set or confidence is below a threshold. Routing is an adapter concern (Chapter 7), so the use case does not change.
- In the composition root, build the cascade and add a `triage.classifier` flag with variants `llm` and `cascade`.
- In the manifest, record `models.classifier_small` and `models.classifier_llm`, plus the cascade threshold in `config`.
- Tests: add replayed fixtures for the small-model endpoint, unit tests for the cascade's routing decisions with two fakes, and an eval gate run with the cascade that also reports the fraction routed to each model and cost per ticket.
- No domain or application test changes, which confirms that the boundary works.

**E2.**
- Hypothesis: the candidate model improves correct-routing rate without hurting guardrails.
- Primary metric: correct routing per agent feedback, on the first assignment.
- Guardrails: error rate (unavailable plus parse errors) increases by no more than one point; p95 latency is at most 1.2 times baseline; cost per ticket is at most 1.15 times baseline; zero safety violations; security-report recall does not drop.
- Randomization unit: the requesting user, with a dedicated salt.
- Sample size: 2,629 users per arm for three points at an 80% baseline. At an illustrative 1,200 tickets per day and 50/50, that is about 4.4 days of tickets. Because users file several tickets, inflate by the design effect, or count users and run about two weeks to cover weekly seasonality.
- Sequence: offline gate, then shadow for a day (agreement and errors), then a 5% canary with the canary policy for 24 hours, then 50/50 for the fixed duration with no early stopping on the primary metric.
- Stop conditions: any guardrail breach rolls back immediately through the kill switch.
- Rollback: kill `triage.model`, which takes effect without a deploy. Document the plan in an ADR with the evidence section filled at the end.

**E3.**
- Run the gate on pull requests with a simulated model, as the example does. This catches wiring, parser, rule, prompt-rendering, and lock regressions, but not model quality.
- Add replayed cassettes for the golden set. Record the candidate prompt's responses on a protected job: either a maintainer-triggered job on a protected branch with keys, or a scheduled re-record. Merge request pipelines then replay real model outputs for the exact request bytes. If the prompt changed and no recording exists, the job fails with `CassetteMiss` and requires a maintainer to run the protected recording job before merge.
- Run the real-provider gate on the default branch after merge, as a blocking step before build.

This approach cannot catch quality changes for prompts that have not been recorded yet, provider drift between recording and merge, or behavior on inputs outside the golden set.

**E4.** Checklist: the items in the chapter's checklist, applied to each component. Specifically:
- Is the prompt a new version with an updated lock and snapshot?
- Does the tool schema bump its version and lock, and is the schema generated from the validator?
- Does the parser keep totality, with new `@example`s?
- Is there an eval report for this manifest fingerprint?
- Is the behavior flagged?
- Do descriptions count as prompt changes?

Ask the author to split the change into three pull requests: parser first, because it is a pure robustness change gated by unit tests; then the tool schema; then the prompt. Three components changing at once lose attribution. If the eval moves, nobody knows which change caused it, and a rollback of one requires reverting all three. The parser change may also be safe to ship immediately while the prompt needs a canary.

## Practical exercises

**P1.**
- Add `RetrieverPort.similar_tickets(ticket, k) -> list[Example]` with attributes `embedding_model` and `index_version`.
- Add an in-memory adapter using `FakeEmbeddings(vocabulary=...)` over `eval/golden.jsonl`.
- `TriageService` calls it and renders examples into the user message as data.
- `with_updates` sets `embedding_model` and `index_version` on the per-request manifest.

Acceptance criteria:
- Spans carry `version.embedding_model` and `version.index_version`.
- A test builds two services with index versions `idx-1` and `idx-2` and asserts different fingerprints and `changed_components == {"index_version"}`.
- The architecture test still passes, because the retrieval port lives in the application and the adapter lives in adapters.
- The prompt snapshot updates through a new prompt version, not an in-place edit.

**P2.**
- Add `excluded_tenants: tuple[str, ...]` to `FlagConfig` and add `tenant` to `evaluate(name, unit_id, tenant=None)`. Keep the port signature backward compatible with a default.
- Precedence: unknown flag, then kill switch, then disabled environment, then tenant exclusion (returns the safe variant with reason `tenant_excluded`), then user override, then allocation.

Tests:
- A killed flag beats an override.
- An excluded tenant beats a user override, or document the opposite choice explicitly and test it.
- Allocation is unchanged for non-excluded tenants, so bucket assignments for other tenants are identical before and after the change.
- An invalid exclusion config is rejected at load.

**P3.**
- `evaluate` accumulates hits per tenant, as `slice.tenant.<name>.category_accuracy`.
- `gate` iterates over tenant slices present in the baseline and fails when `current < baseline - max_regression` for any tenant, even if the aggregate improves.
- Small slices need a minimum count, for example 10, below which the slice reports but does not gate. Otherwise noise blocks every merge.

Acceptance criteria:
- A test with a fake LLM that is perfect for `retail` and degraded for `logistics` shows an aggregate within tolerance but exits 1 with a failure naming `logistics`.
- The report JSON contains every slice.

**P4.**
- Strategy: generate a JSON object body with `st.dictionaries` over text keys and JSON values. Build two requests: one with keys reordered (rebuild the dict from reversed items) and different values for keys in `ignore_body_fields`, and one with one non-ignored key's value changed to a different value.
- Assert `key_for(a) == key_for(reordered)` and `key_for(a) != key_for(changed)`.
- Use `assume` to skip cases where the changed value is equal to the original, or where the only keys are ignored ones.

Acceptance criteria: the test runs at least 200 examples offline. It also covers non-JSON bodies, where the raw bytes are hashed.

## Debugging exercises

**D1.** Root cause: the flag sends 50% to `treatment`, but no treatment prompt version was configured (`TRIAGE_PROMPT_TREATMENT_VERSION` was unset in that deployment). `_select` therefore fell back to the control version for treatment users. Both arms ran identical prompts, so the experiment compared control with control. Telemetry: spans with `version.flags.triage.prompt = treatment` show `version.prompts.triage.classify` equal to the control label (`1.0.0#…`). The two arms also share an identical manifest fingerprint distribution apart from the flag attribute. The check that catches this is `check_consistency` in the composition root. It fails at startup when a flag variant with non-zero traffic has no configured value. The deployment must have bypassed it, for example with an older image or a custom wiring. Restore it, and add an alert on "same prompt label across experiment arms."

**D2.** The adapter sends `TriageDecision`'s JSON Schema as `response_schema`, and pydantic includes the class docstring as the schema `description`. Editing the docstring therefore changed the request body. Its hash no longer matches the committed recording, and replay mode correctly raises `CassetteMiss`. This is not a bug in the transport. It is a real prompt change: the model reads that description. The comment in the prompt store is irrelevant. The fix is to treat the change as a prompt change. Review whether the new description is intended, re-record the cassette deliberately with `CASSETTE_MODE=record` in the protected recording job, review the cassette diff, and run the eval gate. A team that wants docstrings decoupled from the request can set an explicit schema `description` or `title` in the model config so that documentation edits do not change the wire format. That trade-off belongs in an ADR.

**D3.** The provider updated the model behind the alias in use on Tuesday. The application did not change, which is why the manifest fingerprint matches. The served model changed underneath, which the span attribute `llm.served_model` shows. Security-report recall dropped as a result.

Immediate mitigation:
- Pin the previous dated model identifier, if the provider still serves it, through configuration. No code deploy is needed.
- If it is not available, raise the human-review routing for low-confidence and security-adjacent tickets. For example, lower `AUTO_ROUTE_MIN_CONFIDENCE` through a flagged config.
- Open an incident linked to the nightly report.

Changes so this is caught on Tuesday:
- Pin dated identifiers instead of aliases.
- Alert on any change in the distinct values of `llm.served_model` in production spans.
- Include the served model in the manifest's comparison for the nightly gate, so a served-model change is reported as a changed component.
- Run the drift job on a schedule frequent enough for the risk, or trigger it automatically when a served-model change is detected.
