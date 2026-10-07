# Exercises — Chapter 31 — Observability for AI Systems

Solutions: `../solutions/ch31-solutions.md`


**Start here:** K2, K4, E1, P1, D2 (about 4 hours). The rest go deeper.

### Knowledge questions

**K1.** Why is HTTP status plus duration insufficient to detect the most important failures of a RAG assistant? Name two failure classes that produce a 200 response with normal latency.

**K2.** Explain the difference between the `llm.generate` and `llm.complete` spans, and what goes wrong if you merge them into one.

**K3.** What problem does a keyed hash (HMAC) solve that a plain SHA-256 of the same content does not?

**K4.** Why does stage triage compare conditional survival rates rather than absolute ones?

**K5.** Distinguish head sampling from tail sampling, and give one failure each is prone to in an AI system.

**K6.** Why must probe traffic be tagged and filtered separately from user traffic?

### Engineering questions

**E1.** Northwind's logistics tenant forbids any prompt or response content from leaving the request path, including in redacted form. Specify the capture configuration, collector configuration, and debugging workflow that still let you diagnose a logistics quality incident.

**E2.** Design trace context propagation for an agent whose tool calls are executed by a separate worker service via a message queue. Say which ids travel where, and what telemetry reveals a broken hop.

**E3.** The team wants to add `prompt.hash` and `document_id` as dimensions on the quality metrics "to make drill-down easier". Respond with a design that serves the underlying need without the cost.

**E4.** Decide how evaluation labels should arrive for each of these: a nightly offline eval run, an online groundedness judge sampled on 5% of traffic, and a ticket-reopened signal that arrives up to seven days later. For each, give the join key, the latency, and where the label is stored.

### Practical exercises

**P1.** (about 90 min) Add a `retrieval_miss` semantic check: when a probe trace's gold ids are absent from the retrieval results, the joined record should carry `error.class=retrieval_miss`. Then add an alert rule for its rate.

**P2.** (about 3 hours) Extend `analysis.py` with `compare_index_versions_by_replay`. It takes the labeled traces from before an index change, and a function that re-runs retrieval and packing against a candidate index, and reports the funnel for both. Show that it would have caught the chapter's incident before the release.

**P3.** (about 2 hours) Implement tail-sampling logic as a function over completed trees: keep every trace with an error class, negative feedback, or probe traffic, plus a configurable uniform fraction of the rest. Measure what fraction of the simulated incident's diagnostic findings survive at 1%, 5%, and 20% uniform rates.

**P4.** (about 2 hours) Add time-to-first-token to the instrumentation for streamed generations, without modifying `aie_core`. Add a `latency.ttft_p95_ms` metric and a dashboard row.

### Debugging exercises

**D1.** After a deploy, `telemetry.incomplete_rate` rises from 0 to 0.31, and the missing key is `index.version`. Quality metrics are unchanged. Retrieval spans still carry `index.version`, but root spans do not. What is the likely cause, what is the operational risk if nobody fixes it, and how would you confirm the cause from traces?

**D2.** An on-call engineer reports that `errors.trace_error_rate` doubled overnight to 4%, while quality, feedback, and latency are flat. `error_distribution` shows the growth is entirely `rate_limited`, and the affected traces' `llm.generate` spans have status `ok`. What is happening, what is wrong with the metric, and what should the dashboard show instead?

**D3.** Negative feedback for retail rises 60% on Monday morning. Probes pass at the usual rate, the funnel is unchanged, no version changed, and the judge's pass rate on user traffic falls. `sample_for_review` returns traces whose questions mention a product launched that morning, and their retrieval results contain no document about it. Diagnose the incident and say which telemetry distinguished it from a regression in the system.
