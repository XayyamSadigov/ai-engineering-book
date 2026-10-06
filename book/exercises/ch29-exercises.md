# Exercises — Chapter 29 — Reliability and Scalability

Solutions: `../solutions/ch29-solutions.md`


### Knowledge questions

**K1.** Explain why availability, time to first token, completion latency, task success, and degraded share are separate SLIs for Northwind Assist. What goes wrong if degraded answers are simply counted as available and degraded share is not measured?

**K2.** A request passes through four layers that each make up to three attempts. How many calls can one request make to a dead dependency? Describe two mechanisms that bound this and what each one bounds.

**K3.** Why does `default_is_failure` ignore `InvalidRequestError`, `MalformedResponseError`, and `DeadlineExceeded`? Give the incident each exclusion prevents.

**K4.** Why does a deadline travel between services as remaining milliseconds rather than as an absolute timestamp, and why does the receiver cap it?

**K5.** What is the difference between a lease expiring and a `nack`, and why must expired leases count as attempts?

**K6.** Why is estimated wait a better load-shedding signal than accelerator utilization? Use Little's law in your answer.

### Engineering questions

**E1.** The backup provider in Northwind Assist has half the primary's quota. Design what happens, step by step and with which primitives, when the primary goes down at peak, so that the backup is not immediately overwhelmed.

**E2.** You run 6 API replicas and 3 worker pods against a provider limit of 240 concurrent requests and 2 million tokens per minute. Propose per-replica admission capacity, bulkhead sizes, and gateway rate limits, and say what must change when the API autoscales to 12 replicas.

**E3.** An agent in the incident-research workflow calls `search_tickets`, `query_metrics`, and `get_service_status`, and the metrics backend is down. Specify how the agent loop should treat the failure, which limits must be hard, and what the final result looks like.

**E4.** Product wants "exactly one reply email per ticket, guaranteed". Explain what is achievable with the queue and worker in this chapter, where the remaining gap is, and what you would require of the email provider to close it.

### Practical exercises

**P1.** Add per-dependency latency tracking to `CircuitBreaker.snapshot()`: p50 and p95 over the rolling window, computed from bucketed histograms rather than stored samples. Add tests that drive known latencies through a `ManualClock`.

**P2.** Implement an `AsyncWorker` that runs up to N handlers concurrently on one event loop, preserves graceful shutdown (drain all in-flight jobs, release those that checkpoint), and heartbeats leases automatically at a third of the visibility timeout. Reuse the queue contract tests and add concurrency tests.

**P3.** Extend `AdmissionController` to count estimated KV-cache memory instead of request count for a self-hosted model (Chapter 34): each request reserves memory proportional to prompt plus maximum output tokens, and capacity is a memory budget. Show with a test that one 64k-token request blocks as much as sixteen 4k-token requests.

**P4.** Write a chaos test for the fallback overload cascade: primary out, backup rate-limited above a threshold of concurrent calls. Make the test fail on the current code if the degrade policy does not reduce load, then make it pass.

### Debugging exercises

**D1.** During a provider incident, the dashboard shows the `llm:primary` breaker closed on every replica, provider error rate at 60%, and outbound request rate at four times normal. Each replica sends about 3 calls per second to the provider, retries included, and the breaker is configured with `min_calls=50` and `window_s=10`. Diagnose why the circuit never opened and what the request-rate increase tells you.

**D2.** After a deploy, the dead-letter queue fills with `ingest_document` jobs whose `last_error` is "lease expired on final attempt", although the documents are small and the parser has not changed. The new deploy raised the worker termination grace period from 30 to 120 seconds and lowered `WORKER_VISIBILITY_TIMEOUT_S` from 300 to 60. Traces show handlers taking 70 to 90 seconds because of a new embedding step. What happened?

**D3.** p95 latency for interactive requests is fine at 10:00 and doubles every day at 02:00 and 14:00, although traffic at those times is low. Admission control shows no rejections and utilization below 40%. The nightly and midday evaluation runs start at those times. Identify the missing protection and the telemetry that confirms it.
