# Chapter 31 — Solutions

## Knowledge questions

**K1.** A RAG assistant's worst failures are semantic. The service completes normally and returns a fluent answer that happens to be wrong. HTTP status records whether the code path raised, and duration records how long it took. Neither records whether the answer was grounded in the right evidence. Two examples of a 200 with normal latency:

- *Context truncation.* Gold evidence is retrieved but dropped by the packer, and the model answers from a superseded document.
- *Unsupported claim.* The evidence is in context, and the model asserts something it does not say.

Others include retrieval miss, citation mismatch, bad refusal, and cross-tenant contamination. Detecting them needs quality labels and stage decisions on the trace: evidence ids, citations, and eval verdicts.

**K2.** `llm.generate` is one logical generation. It carries prompt identity, the captured prompt and response, and final usage. `llm.complete` is one provider attempt, emitted by the gateway. It carries provider, served model, attempt number, queue time, latency, per-attempt cost, and the error class of a failed attempt. Merging them causes one of two problems. If you keep one span per attempt, prompt identity and content are duplicated and token totals are double-counted across retries. If you keep one span per generation, retries, fallbacks to another model, and per-attempt latency disappear. You can then no longer tell recovered rate limits from clean calls, or attribute cost to the model that actually served.

**K3.** A plain hash of low-entropy content is reversible by guessing. Email addresses, user ids, employee numbers, and short tool arguments can be enumerated, hashed, and matched against telemetry. An HMAC with a secret salt cannot be computed without the salt. You keep equality, deduplication, and joinability, and you lose reversibility, as long as the salt stays in a secret store and out of the telemetry pipeline.

**K4.** Absolute survival rates propagate losses downstream. If retrieval loses gold evidence in 20% of traces, every later stage's absolute rate also drops by at least that much, so every stage looks guilty. The conditional rate, the probability of surviving a stage given survival of the previous one, isolates the stage where the new loss begins. In the chapter's incident, the absolute `cited` rate fell 0.25. Its conditional drop was 0.03, which shows the model was not at fault.

**K5.** Head sampling decides at the root, typically by trace-id ratio. It is cheap and consistent across services, but it keeps rare failures at the same low rate as everything else, so a low ratio misses them. Tail sampling decides after the trace completes, so it can keep all errors, slow traces, and traces with feedback. It needs a collector that buffers whole traces, adds cost and delay, and has a specific trap in AI systems: rules such as "keep errors" miss semantic failures, which are not errors. Include probes, feedback-flagged traces, and a uniform sample, and remember that feedback often arrives after the sampling decision.

**K6.** Probe traffic consists of synthetic golden questions with known gold evidence. It is the only production traffic with stage-level ground truth, so it must be identifiable for the funnel. It must also be excluded from user-facing metrics. Probes inflate request counts, skew topic and tenant mixes, can bias quality metrics in either direction, and must not be billed to tenants or used as training data as if they were real user behavior. The `traffic.source` attribute on the root does both jobs.

## Engineering questions

**E1.**

- **Capture.** Set `tenant_ceiling={"logistics": "off"}`, or `"hashed"` if keyed hashes are contractually acceptable. The ceiling is applied after sampling and capture-on-error, so no logistics span ever carries `.content`.
- **Collector.** Add an attribute processor that deletes every `*.content` key for `tenant.id=logistics`, as defense in depth. Route logistics telemetry to an in-region pipeline and store if residency applies.
- **What is still recorded.** Debugging relies on structure: evidence ids and ranks, dropped ids, token counts, prompt and index versions, error classes, guardrail decisions, tool names, argument fingerprints, and statuses. The funnel, version diffs, loop detection, and cost analysis all work on this data, as the chapter's agent trajectory shows.
- **When content is truly needed.** Reproduce inside the tenant boundary. Replay the request in an in-region environment from the recorded ids and versions: fetch evidence by id from the index manifest and render the template by version. Alternatively, use logistics-owned probe questions whose content is non-sensitive and can be captured.
- **Process.** Record the access decision in the incident ticket.

**E2.**

- **Producer side.** The agent process opens the `tool.call` span and injects W3C `traceparent` (plus `tracestate` and baggage carrying tenant and the version manifest) into the message headers.
- **Worker side.** The worker extracts the context and opens a child span, for example `tool.execute`, with the worker's own attributes: queue wait time, which is enqueue-to-dequeue, execution status, and retries. It then returns the result with the same call id.
- **Idempotency.** The idempotency key travels in the message body and in both spans.
- **Detecting a broken hop.** Worker spans arrive with a trace id that has no matching root, or they start new traces whose `tool.call_id` matches a tool span elsewhere. The agent's `tool.call` spans have no worker children. `completeness()` shows a rising orphan rate. A contract test should run one tool call through the real queue and assert a single trace id with a parent link.

**E3.** The need is drill-down from an aggregate to examples. Metrics must stay low-cardinality, and a prompt hash or document id label multiplies series by thousands. Two changes serve the need instead:

- **Keep metric dimensions to tenant, route, traffic source, and the few live versions.** `prompt.version` is fine because it has a handful of values. `prompt.hash` is not.
- **Do drill-down on traces.** Dashboards link a metric panel to a trace query with the same filters, for example "labeled traces after 09:00 where `context.dropped_ids` contains the gold id". Use exemplars, meaning trace ids attached to histogram buckets, where the backend supports them.

For per-document questions, such as which documents are most often dropped, run a scheduled aggregation over traces into a small table. Do not use a live metric label.

**E4.**

| Signal | Join key | Latency | Storage |
|---|---|---|---|
| Nightly offline eval | `trace_id` written by the eval runner, for example evalkit's `CaseResult.trace_id`, plus `case_id` and dataset version | hours | eval results store; joined rows for dashboards. It is replay traffic, so tag `traffic.source=eval` and keep it out of production metrics |
| Sampled online judge | `trace_id`; inline as an `eval.score` child span only if it gates the response, otherwise as an async record | seconds to minutes | joined records, with judge model and version in `eval.name` |
| Ticket reopened | the ticket id, recorded as an attribute on the trace that created or answered the ticket at action time, resolved to `trace_id` | up to 7 days | delayed-label table keyed by trace id. Trace retention for content-free spans must exceed 7 days, or the needed attributes must be copied into the joined row at request time |

## Practical exercises

**P1.**

- **Where the check runs.** In `join_signals.join`, or as a post-join pass, for each labeled trace whose `evidence_path(...)["retrieved"]` is false, add `retrieval_miss` to the joined row's error classes. Alternatively, write a derived record with `error.class=retrieval_miss` that the store merges into `error_classes`. Do not mutate the original span; derived classes should be marked as derived.
- **The alert.** Add a rule `errors.class_rate:retrieval_miss` with `group_by: tenant.id`, a 6 h window, `baseline_ratio: 1.5`, and `min_samples` matching the probe rate.
- **Acceptance.** A unit test builds a labeled trace without gold in `retrieval.returned_ids` and asserts the class. The simulated incident does not fire the new alert, because retrieval was healthy. A simulation variant with a degraded retriever does fire it.

**P2.**

- **Signature.** `compare_index_versions_by_replay(store, replay: Callable[[TraceTree], tuple[list[str], list[str]]])`. It takes the labeled traces, calls `replay` to get retrieved and packed ids for the candidate index, and computes the funnel from the recorded spans for the baseline and from the replay output for the candidate.
- **Fixture.** In the test, `replay` reuses the simulator's ranking and packing logic with chunker `c3`.
- **Acceptance.** Run against the pre-deploy probe traces, the candidate funnel shows `in_context` dropping by more than the tolerance while `retrieved` holds. Wire it into the index-build release gate.

**P3.**

- **Function.** `tail_sample(store, uniform: float, seed) -> TraceStore` keeps traces with error classes, negative feedback, or `traffic.source=probe`, plus a seeded uniform fraction of the rest.
- **Measurement.** Run the playbook on each sampled store and compare `triage_by_stage(...).first_failing_stage`, the count from `retrieved_not_cited`, and the fired alerts with the unsampled result.
- **Expected result.** Because all probes are kept, stage triage survives at every rate. Feedback-based metrics need the feedback join to happen before the sampling decision, or buffering long enough for feedback to arrive. This is the practical argument for a delayed tail decision or for always keeping probe and feedback traces.
- **Report.** Show stored bytes against diagnostic fidelity.

**P4.**

- **Wrapper.** Wrap `gateway.stream(req)` in a generator inside an `llm.generate` span. Record the clock at the first `text_delta`, or the first `tool_call_delta`, as `llm.ttft_ms` relative to span start. Accumulate the text for capture and set usage from the `usage` event. `aie_core` is not modified: its hand-built streaming span is adopted by `AITracer.export` into the current tree.
- **Metric.** Add `latency.ttft_p95_ms` to `METRICS` as the 95th percentile of `llm.ttft_ms` over `llm.generate` spans, add a row to the latency panel in `dashboards.md`, and optionally add an alert against the 2 s target.
- **Test.** Use `FakeLLM` streaming with an injected clock advanced inside the handler.

## Debugging exercises

**D1.**

- **Root cause.** The deploy changed how the root span is opened, so the version manifest no longer reaches `trace_request`. For example, a refactor builds the manifest after the root opens, or passes it under the wrong key. The retrieval stage still sets `index.version` on its own span, which is why the store's fallback lookup hides the gap from some queries but not from `completeness()`, which checks the root.
- **Risk.** Quality is unchanged today. The next incident's version diff will be impossible or wrong, and the `llm.complete` spans lose the propagated `index.version`, so queries grouping provider attempts by index break silently.
- **Confirmation.** Filter traces by `app.version`: the missing-key fraction is near 1.0 for the new version and 0 for the old. The 0.31 overall rate matches the rollout fraction. Inspect one new-version root and see the absent key while its retrieval child has it.
- **Fix.** Restore the manifest on the root and add the schema contract test to CI.

**D2.** These are rate limits absorbed by retries. The provider attempts failed with `rate_limited`, the gateway retried, and the logical generations succeeded. That is why `llm.generate` is `ok` and quality is flat. The metric is counting recovered attempt errors as trace failures, which inflates the outcome error rate and trains people to ignore it. The dashboard should show `error_classes`, which excludes recovered attempts, for the outcome error rate. It should show `recovered_errors`, retry rate, and `cost.llm_calls_per_request` on a separate reliability panel, because rising retries do matter: they add latency and cost, and they are an early warning of quota exhaustion (Chapter 29). Also check whether the retry delay is pushing p95 latency up.

**D3.** This is a knowledge gap, not a system regression. Users are asking about a product launched that morning, and its documentation is not in the index yet, either not written or not ingested. The telemetry distinguishes the two cases as follows:

- **System health.** Probes pass at the usual rate and the funnel is unchanged, so retrieval, packing, and generation behave as before on known questions. No version changed.
- **Traffic.** The sampled traces cluster on a new topic, the retrieval results contain no relevant document (low top scores, if you track the score distribution), and the judge flags ungrounded or abstaining answers.

The fix is in the content pipeline: ingest the launch documentation and check index freshness for that source. Add a freshness signal, such as time since the last successful ingest per source, plus a probe for the new product. Consider an alert on an abstention or low-retrieval-score rate by topic cluster. Prompt changes would not help.
