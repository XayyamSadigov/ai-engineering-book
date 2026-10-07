# Exercises — Chapter 3 — Working with LLM APIs

Solutions: `../solutions/ch03-solutions.md`


**Start here:** K3, K7, E1, P2, D1 (about 3 hours). The rest go deeper.

### Knowledge questions

**K1.** Why does the assistant message containing tool calls have to be replayed verbatim before the tool result messages? What happens at the provider if it is omitted?

**K2.** Explain the difference between an exact-match response cache and provider prompt caching along three axes: where it lives, what the key is, and what it saves.

**K3.** A gateway has `max_attempts=3`, `base_delay_s=0.5`, `max_delay_s=8`, full jitter, and a request deadline of 2 seconds. The provider returns 503 on every attempt. What is the maximum number of attempts the gateway can make, and why might it make fewer?

**K4.** Which of the six error classes should trigger a fallback to another provider, and why is `MalformedResponseError` excluded even though the call failed?

**K5.** Why does the adapter buffer tool-call argument fragments instead of emitting them as they arrive, while it emits text fragments immediately?

**K6.** Streaming reduces neither server-side time to first token nor total generation time. What does it reduce, and which user-facing experience do TTFT and total completion time each govern?

**K7.** A ticket classifier moves to a reasoning model and keeps `max_tokens=150`, sized for a 20-token visible answer. A few percent of responses now come back empty with `finish_reason` `length`, and the monthly bill rises far more than the price-per-token difference suggests. Explain both symptoms and name two fixes.

### Engineering questions

**E1.** Northwind runs the same gateway for the `retail` and `logistics` tenants, with tenant permissions enforced by the tool layer rather than in the prompt. The response cache is enabled. Describe the bug, then propose two different fixes and the tradeoff between them.

**E2.** A batch job classifies 50,000 tickets nightly with temperature zero. Design the gateway configuration (limiter, concurrency, cache, retries, timeout, fallback) and justify each number against the provider's published per-minute limits, which you may treat as illustrative.

**E3.** You must add a provider whose API is request/response only, with no streaming. How would you satisfy the `LLMClient` protocol's `stream` method without lying to callers about latency? Discuss what the `done` event should carry.

**E4.** The team wants semantic caching: a cache hit when a new question is "close enough" to a cached one. Specify the key, the similarity threshold policy, the invalidation rules, and the evaluation you would run before enabling it.

### Practical exercises

**P1.** (about 2 hours) Implement `IdempotentGateway`, a thin wrapper around `ModelGateway` that accepts an idempotency key in `req.metadata`, stores in-flight and completed results keyed by it, and guarantees that concurrent or retried calls with the same key produce exactly one provider call. Write tests with `FakeLLM` and threads.

**P2.** (about 90 min) Write a streaming tool-calling loop for Northwind Assist: stream text to stdout, collect tool calls, execute them against a dict of fake tools (`lookup_employee`, `search_tickets`), append results, and continue until the model stops calling tools or a step limit is hit. Use `FakeLLM(handler=...)` to script a two-step conversation.

**P3.** (about 90 min) Add a `RedisResponseCache` that implements the `ResponseCache` protocol with TTLs, serializing `Completion` via pydantic. Provide an in-memory fake Redis for tests and verify the gateway behaves identically with both caches.

**P4.** (about 45 min) Build a `PrefixStabilityCheck` test helper: given a function that renders a prompt for a request, call it twice with different user content and assert the leading N bytes are identical. Apply it to a Northwind system prompt that currently embeds the current date, and fix the prompt.

### Debugging exercises

**D1.** Traces show, for one logical request, four `llm.complete` spans: attempts 1, 2, 3 on provider A with status error and `RateLimitError`, then attempt 1 on provider B with status ok. Total span time is 41 seconds although the request's `timeout_s` was 10. The injected clock is the real one. List the candidate causes in the gateway or its configuration and the single log line or attribute that would confirm each.

**D2.** After a deploy, cost per request rose 35 percent with no change in traffic or token counts; `cached_input_tokens` dropped to near zero on every span. The diff touched only the system prompt template. What happened, and what test would have caught it?

**D3.** Users report answers that stop mid-sentence roughly once in two hundred requests. Spans for those requests have `stream=True`, status ok, `finish_reason="stop"`, and `output_tokens` well below `max_tokens`. No error events were emitted. Where in the stack is the bug most likely, and what two experiments narrow it down?
