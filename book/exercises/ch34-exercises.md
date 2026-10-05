# Exercises — Chapter 34 — Inference and Serving Essentials

Solutions: `../solutions/ch34-solutions.md`


### Knowledge questions

**K1.** Define queue time, TTFT, TPOT, end-to-end latency, throughput, and goodput. For an interactive chat product and for a nightly batch classification job, name the one metric each cares about most and explain why the other metrics matter less for that product.

**K2.** Explain in two or three sentences, without kernel detail, why the decode phase is limited by memory bandwidth and why that makes batching during decode nearly free in throughput terms.

**K3.** State the KV-cache formula and name each variable. For a model with 48 layers, 8 KV heads, head dimension 128, and a BF16 cache, compute the cache per token and per 16k-token sequence. How does the result change with an FP8 cache?

**K4.** Contrast static batching and continuous batching. What property of LLM requests makes static batching waste capacity, and what scheduling problem does continuous batching introduce for interactive latency?

**K5.** Why is "measure task quality, not perplexity alone" the right rule for evaluating a quantized model? Name three task slices that commonly regress while aggregate scores look stable.

**K6.** State Little's Law and give one example of using it in each direction: deriving in-flight requests from traffic, and deriving a sustainable arrival rate from a fixed concurrency budget.

### Engineering questions

**E1.** Northwind's logistics tenant requires that incident reports never leave the corporate network, while the retail tenant has no such constraint. Design the routing so both tenants use the same application code and the same gateway. Which `aie_core` components are involved, what does the `ServingTarget` list look like, and what must appear in every trace?

**E2.** A team proposes raising the engine's maximum model length from 16k to 64k so that one rare long-document feature works. Using the illustrative 8B shape on an 80 GiB device, quantify the effect on maximum concurrency and propose two alternatives that serve the feature without paying that cost for every request.

**E3.** Your prompt registry adds a `request_id` to the first line of every system prompt for debugging. Explain the effect on prefix caching, estimate the TTFT impact for a 2,000-token system prompt relative to a 300-token user message, and propose a layout that keeps the debugging value without the cost.

**E4.** A load test shows p95 TTFT crossing the 2 second SLO at concurrency 12, with TPOT flat and KV utilization at 60 percent. Queue depth is rising. Which of the three constraints in `replicas_needed` is binding, what does that imply about the fix, and why would quantizing the KV cache not help here?

### Practical exercises

**P1.** Extend `loadtest.py` to record the engine's Prometheus-style metrics (queue depth, KV utilization, running sequences) at the end of each concurrency level, add the columns to `LevelSummary` and the report, and extend `fake_server.py` to expose a `/metrics` endpoint so the feature is tested offline.

**P2.** Write `plot_sweep.py` that reads a list of `LevelSummary` objects (serialize them to JSON from `sweep`) and produces the two protocol plots: p50 and p95 TTFT against requests per second, and output tokens per second against p95 E2E. Mark the SLO lines and the operating point.

**P3.** Build an admission-control function on top of `kv_cache.tokens_that_fit` that, given free KV bytes reported by the engine and a request's input tokens plus `max_tokens`, decides between admit, queue (with a bounded wait), and reject with a retryable error. Test it with a sequence of mixed-length requests against a fixed budget and assert that it never admits more tokens than fit.

**P4.** Run the load generator against a real local OpenAI-compatible server of your choice, once with the default configuration and once with a quantized variant of the same model. Record the full configuration from the protocol section, find both operating points, and run the same small structured-output evaluation against both. Write a one-page comparison that reports goodput at the SLO and the quality deltas by slice.

### Debugging exercises

**D1.** After an engine upgrade, p50 TTFT for the RAG endpoint doubled across all replicas while TPOT, error rate, GPU utilization, and traffic were unchanged. The prompt registry shows no changes in the last week. Traces show input token counts identical to last week's. Diagnose the most likely cause and name the single engine metric that would confirm it.

**D2.** A dashboard shows output tokens per second at an all-time high and GPU utilization at 85 percent, yet support tickets about "the assistant hangs before answering" tripled this afternoon. KV utilization is pinned at 99 percent and the engine's preemption counter is climbing. Explain what is happening, why the throughput number is misleading, and what two changes, one immediate and one structural, address it.

**D3.** Overnight, a self-hosted replica's KV utilization climbed steadily to 95 percent and stayed there with almost no client traffic; the gateway shows a handful of open connections, the engine shows forty running sequences. In the morning every request queues. Identify the fault, the layer most likely responsible, and the test that should have caught it.
