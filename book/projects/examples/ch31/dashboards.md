<!-- path: book/projects/examples/ch31/dashboards.md -->
# Northwind Assist dashboards: metric definitions

Every metric is computed in `metrics.py` (`METRICS`, plus the parametric family
`errors.class_rate:<error_class>`). In production, emit the same quantities as counters and
histograms at request time, with the dimensions listed, and keep traces sampled. Alert rules
live in `alerts.yaml` and may only reference names defined here. Thresholds are illustrative.

Dimensions available on every metric: `tenant.id`, `app.route`, `traffic.source`,
`prompt.version`, `index.version`, `llm.model`, `app.version`. Keep version dimensions low
cardinality: versions are a handful of values at a time; user ids and trace ids are never
metric labels.

## Panel 1: Quality (top row: it is the reason the system exists)

| Metric | Definition from spans | Unit | Notes |
|---|---|---|---|
| `quality.eval_pass_rate` | labeled traces with every eval `passed=true` / labeled traces | ratio | Labels come from probes (golden questions on a schedule), sampled judges, and human review. Show the sample size next to the value. |
| `quality.negative_feedback_rate` | traces with feedback -1 / traces with any feedback | ratio | Biased toward unhappy users; trend it, never read the level as quality. |
| `quality.feedback_rate` | traces with any feedback / traces | ratio | A falling rate can mean a broken feedback widget, not happier users. |
| `quality.context_truncation_rate` | `context.build` spans with `context.truncated=true` / all | ratio | Leading indicator of evidence loss. |
| `quality.abstention_rate` | roots with `response.abstained=true` / all | ratio | Spikes after index or ACL changes. |
| `quality.guardrail_block_rate` | `guardrail.check` spans with decision `block` / all | ratio | Split by `guardrail.name`. |

## Panel 2: Latency

| Metric | Definition | Unit | Notes |
|---|---|---|---|
| `latency.request_p50_ms`, `latency.request_p95_ms` | percentiles of root `request` duration | ms | Northwind target: p95 under 8 s for answers. |
| `latency.retrieval_p95_ms` | p95 of `retrieval.search` duration | ms | |
| `latency.llm_p95_ms` | p95 of `llm.complete` duration | ms | Plot next to input-token p95: decode and prefill scale with tokens. |

Add time-to-first-token (`llm.ttft_ms`) for streaming routes; it is the latency users feel.

### SLO burn (Chapter 29)

| Metric | Definition | Unit | Notes |
|---|---|---|---|
| `slo.availability_burn_rate` | share of roots with error status / (1 - 0.995) | ratio | 1.0 spends the 30-day budget exactly; paged at 14.4 over 1 h confirmed over 5 m. |
| `slo.completion_burn_rate` | share of roots over 8,000 ms / (1 - 0.95) | ratio | Ticket at 6 over 6 h confirmed over 30 m; a 95% objective cannot reach a 14.4 burn without 72% slow requests. |

## Panel 3: Cost

| Metric | Definition | Unit | Notes |
|---|---|---|---|
| `cost.per_request_usd` | sum of `llm.cost_usd` over `llm.complete` spans / requests | USD | Illustrative pricing table. Cache hits count 0; their original cost is in `llm.avoided_cost_usd` (Chapter 30). |
| `cost.input_tokens_per_request` | sum of `llm.usage.input_tokens` on `llm.generate` / requests | tokens | Context growth shows here first. |
| `cost.llm_calls_per_request` | `llm.complete` spans / requests | count | Retries and agent steps multiply cost. |
| cost per successful request | see `analysis.cost_by_tenant` | USD | The number to compare architectures with. |

## Panel 4: Errors and safety

| Metric | Definition | Unit | Notes |
|---|---|---|---|
| `errors.trace_error_rate` | traces with any outcome-affecting `error.class` / traces | ratio | Recovered retries are excluded; see `recovered_errors`. |
| `errors.class_rate:<class>` | traces with that class / traces | ratio | One line per class from `semconv.ErrorClass`; stacked area. |
| `errors.tool_failure_rate` | `tool.call` spans with status not `ok` / all | ratio | Split by `tool.name`. |

## Panel 5: Telemetry health

| Metric | Definition | Unit | Notes |
|---|---|---|---|
| `telemetry.incomplete_rate` | request traces missing a root, with orphans, or missing a required key | ratio | Required: tenant, route, prompt version, index version. |
| unmatched feedback | feedback events with no trace (from `join_signals.py`) | count | Rising = traces dropped or ids not propagated. |
