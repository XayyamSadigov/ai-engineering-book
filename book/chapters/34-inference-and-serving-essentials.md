# Chapter 34 — Inference and Serving Essentials

This chapter is the serving math behind a model endpoint: what limits how many users one GPU serves, which latency numbers users feel, and how to plan capacity from a load test instead of a spec sheet. You need it the day a workload moves from a hosted API to open weights on hardware you control, or the day a hosted endpoint's latency becomes your problem.

**You will be able to:**
- Decide, with numbers, which slice of traffic (if any) should run on a self-hosted or managed open-weight endpoint instead of a hosted API.
- Define TTFT, TPOT, end-to-end latency, throughput, and goodput, and read a serving benchmark for the one your users feel.
- Size the KV cache for a model and context length, including grouped-query, sliding-window, latent-attention, and mixture-of-experts models, and turn it into a concurrency limit.
- Plan replicas with Little's Law and a load-tested operating point, and explain why headroom is not waste.
- Run a reproducible closed-loop load test that measures TTFT through streaming and finds the operating point.
- Put a self-hosted, OpenAI-compatible server behind the same `aie_core` client and gateway the rest of the book uses.

**Prerequisites:** Chapters 2 (tokens, prefill and decode, the KV cache as a mechanism), 3 (the `aie_core` client and `ModelGateway`), and 5 (prompt layout for prefix caching). | **Code:** `book/projects/examples/ch34/` (run: `.venv/bin/python -m pytest book/projects/examples/ch34 -q` from the repository root) | **Builds:** a KV-cache and concurrency calculator, Little's Law and replica-estimate helpers, an async streaming load generator, an in-process fake server, and a serving-target router over `aie_core`.

## Why this matters

Most of this book treats the model as a service behind an HTTP call. That abstraction holds until one of four things happens. Volume grows until the per-token bill becomes a line item the finance team asks about. A customer or regulator requires that prompts never leave a region or a network. A product needs a model the hosted providers do not offer, such as a fine-tuned open model from Chapter 33 or a small specialist for classification. Or latency needs to be controlled rather than observed, because a voice product (Chapter 35, Case 2) cannot tolerate a provider's queue during a traffic spike.

At that point an application engineer inherits a problem that looks like infrastructure: GPUs, serving engines, memory budgets, batch schedulers. The temptation is to treat it as someone else's domain. That is a mistake for two reasons. First, the serving layer's behavior leaks straight into application behavior: the prompt layout you chose in Chapter 5 decides whether prefix caching (reusing computation for a shared prompt opening) helps, the context lengths your RAG pipeline produces decide how many users fit on a GPU, and the quantization (lower-precision weights) an operator picks silently changes your structured-output success rate.

Second, the decision to self-host is often made badly in both directions: teams pay per-token prices for workloads that would be several times cheaper on dedicated hardware, and other teams buy GPUs for a workload that never gets near the utilization needed to break even.

Engine internals are a deep subject, and this chapter compresses them deliberately. You do not need to know how a paged attention kernel works to run a serving platform well, but you do need the arithmetic of memory, the vocabulary of metrics, and a disciplined benchmark protocol.

## Mental model

> **Mental model:** Serving an LLM is a memory and scheduling problem wrapped around matrix multiplication. Weights are a fixed cost; the KV cache (the per-request store of attention keys and values, Chapter 2) is the variable cost that decides how many users share the GPU; the scheduler decides who waits.

Three consequences follow and recur throughout the chapter. Because the KV cache grows linearly with context length, long contexts buy capacity at the expense of concurrency, and "the model fits in memory" says nothing about how many requests fit. Because token generation reads the entire weight set once per decode step, shared by every sequence in the batch, the generation phase is limited by memory bandwidth, and batching several requests into one weight read is almost free throughput. Because queues form in front of a saturated GPU, average latency is meaningless near capacity and the tail is what users feel.

> **Mental model:** Capacity is an empirical envelope found by load testing, not a division of peak tokens per second by tokens per request.

## Core concepts

### Hosted API or self-hosted engine

The decision is rarely binary. Many production systems route most traffic to a hosted provider and send a slice to a self-hosted model: regulated tenants, high-volume cheap tasks, or a fine-tuned specialist. The question is which slice, if any, justifies the operational burden. Chapter 7 frames this as part of model selection; this section adds the serving-side detail. Five factors decide it.

**Cost at volume.** Per-token pricing has no fixed component; a self-hosted GPU costs the same per hour whether it serves one request or a thousand. Self-hosting wins only when utilization is high and sustained. The cost section later gives the arithmetic; a GPU idle 60 percent of the day often costs more per useful token than the API it replaced.

**Data residency and confidentiality.** Some requirements can only be met by keeping prompts inside your own network: air-gapped environments, contractual prohibitions on third-party processing, or source code the security team refuses to send anywhere. These settle the decision for the affected traffic.

**Latency control.** A hosted API gives you an observed latency distribution; a self-hosted engine gives you a controlled one. You decide batch size, admission policy, headroom, and whether a batch job may delay an interactive user. You also own every outage. For voice and latency-budgeted agent loops, control can be worth the burden.

**Model choice.** Fine-tuned open-weights models (Chapter 33), particular licenses, or small specialists no provider hosts require self-hosting. The strongest general models are often hosted-only, which is why the hybrid is common: self-host the specialist, call the API for the hard cases.

**Operational burden.** GPU procurement or reservation, driver and engine upgrades, capacity planning, on-call for a stateful service whose failure mode is memory exhaustion, and a quality-suite rerun after every engine change. Budget a fraction of an engineer permanently.

| Factor | Favors hosted API | Favors self-hosting |
|---|---|---|
| Traffic | Spiky, low, or unpredictable | Sustained, high, predictable enough to size |
| Data | Can leave the network under contract | Must stay in-region or on-premises |
| Latency | Observed p95 is acceptable | Need control over batching, admission, headroom |
| Model | Frontier general model required | Fine-tuned, licensed, or specialist open model |
| Team | No one to own GPUs and engine upgrades | Platform team exists, or workload justifies hiring |
| Change rate | Want new models without migration work | Want pinned behavior and reproducible outputs |
| Typical outcome | Everything hosted, gateway for retries and cost (Chapter 3, 30) | Hybrid: hosted default, self-hosted slice behind the same gateway |

**The middle option: managed open-weight endpoints.** Between a hosted proprietary API and your own GPUs sits a third choice: a cloud or inference provider serves open weights for you, billed per token or per reserved GPU-hour. You get the model choice and portability of open weights, often including your own fine-tune or LoRA adapter, without procurement, driver upgrades, or an on-call rotation for memory exhaustion. You give up most of the latency control and part of the residency guarantee, and you still inherit the quality questions of this chapter: the provider picks the engine, the quantization, and the upgrade schedule, so ask which quantization is served and rerun your evaluation suite when it changes. It is the usual first step when a team wants open weights, and the right end state when volume is too spiky to keep dedicated GPUs busy. A dedicated-capacity variant (reserved replicas run by the provider) moves the cost model toward self-hosting: you pay per hour again, so the utilization arithmetic in Production considerations applies.

Whatever the split, both sides speak the same `LLMClient` protocol from Chapter 3, so retries, caching, cost accounting, and tracing come from the same `ModelGateway`. Where each target sits is a design choice. Interchangeable replicas of the same model belong in one gateway's fallback chain. A different model with a different envelope (shorter context, no tool calling, another quantization) belongs behind a capability-aware router like the one in Chapter 7 (this chapter's `check_fit` and `choose_target` are a minimal version) that chooses a target per request and then calls that target's gateway; putting it in a fallback chain would silently change capability in the middle of an incident.

### The metrics, defined precisely

Serving benchmarks are full of numbers that sound interchangeable and are not. Define each one before you measure it, and know which one your users experience.

| Metric | Definition | Who feels it |
|---|---|---|
| Queue time | From request arrival at the server to the moment the scheduler starts prefill | Nobody directly; it is inside TTFT, and it is the first thing to grow under load |
| Time to first token (TTFT) | From request send to first content token received. Includes network, queue time, prefill, and first scheduling step | Interactive users: this is the pause before anything appears |
| Time per output token (TPOT), also inter-token latency | Average gap between consecutive content tokens after the first | Interactive users as the streaming pace; readers notice above roughly 100 ms per token |
| End-to-end latency (E2E) | From send to final token, including all of the above plus post-processing | Everyone; agents and batch pipelines feel only this |
| Throughput | Aggregate work per second across the server: output tokens/s, or requests/s | The operator and the bill |
| Goodput | Throughput counting only requests that met their SLO | The product; it is the capacity you can advertise |
| p50 / p95 / p99 | Percentiles of any of the above across requests | p50 is the demo; p95 and p99 are what one in twenty and one in a hundred users get |

Two distinctions matter in practice. First, TTFT is where load shows up. A saturated GPU does not generate tokens more slowly; it makes new requests wait. TPOT stays flat while TTFT climbs, so a dashboard showing only tokens per second looks healthy while users stare at a spinner. Second, goodput, not throughput, is capacity. A benchmark reporting 4,000 tokens per second at a p95 TTFT of 9 seconds reports a number you cannot use.

Northwind Assist's targets are p95 TTFT under 2 seconds and p95 completion under 8 seconds for RAG answers. Every benchmark in this chapter is judged against those two numbers.

### Prefill and decode

A request passes through two phases that stress hardware differently. **Prefill** processes every prompt token at once: one large matrix multiplication per layer over the whole prompt, which keeps the GPU's arithmetic units busy and is therefore compute-bound. Prefill produces the first token and, as a side effect, the KV cache for the prompt. **Decode** then generates one token per step per sequence. Each step reads the entire weight set and the sequence's KV cache from GPU memory to produce a single token's worth of arithmetic.

That ratio, many bytes moved for very little arithmetic, is the whole intuition for why decode is memory-bandwidth-bound. Moving tens of gigabytes of weights from memory to the compute units takes a fixed time per step regardless of how many sequences are in the batch; the arithmetic for one extra sequence is tiny by comparison. A GPU decoding one sequence is mostly waiting on memory, and decoding sixteen sequences in one step costs nearly the same time. This is why batching is close to free throughput during decode, why memory bandwidth predicts single-stream TPOT better than FLOPS, and why quantizing weights to fewer bytes speeds up decode even when the arithmetic is unchanged.

The two phases map onto the two user-facing metrics. Long prompts lengthen prefill and therefore TTFT; long outputs lengthen decode and therefore E2E. A 6,000-token RAG prompt and a 60-token chat message have similar TPOT and very different TTFT, which is why traffic shape, not model size alone, determines what users feel.

### The KV cache and the concurrency it allows

Chapter 2 introduced the KV cache as the mechanism that avoids recomputing attention keys and values for every earlier token on every decode step. Here it matters as a budget. For a conventional attention model the cache for one sequence is

```
kv_bytes = 2 * layers * kv_heads * head_dim * tokens * bytes_per_element
```

The factor 2 is keys plus values. `kv_heads` is the number of key/value heads, which under grouped-query attention is smaller than the number of query heads: a model with 32 query heads and 8 KV heads stores one quarter of the cache a full multi-head model would. This single architectural choice, invisible to the application, can quadruple the users one GPU serves.

Take an illustrative 8-billion-parameter model with 32 layers, 8 KV heads, head dimension 128, and a BF16 cache (2 bytes per element). Per token that is 2 × 32 × 8 × 128 × 2 = 131,072 bytes, or 128 KiB. Two worked contexts:

| Context (tokens) | KV cache per sequence | Why that length |
|---|---|---|
| 8,192 | 1.00 GiB | A typical RAG answer: system prompt, five or six evidence chunks, a question, and a few hundred output tokens |
| 32,768 | 4.00 GiB | A long document in context or a many-turn agent loop |

Doubling the context doubles the cache. Now place it on an illustrative 80 GiB GPU. Weights at BF16 take about 15 GiB; reserve 10 percent of the device (8 GiB) as headroom for fragmentation and bursts, and 2 GiB for the runtime's own buffers. That leaves 80 − 15 − 8 − 2, roughly 55 GiB, for cache. At 8k tokens that is about 55 concurrent sequences; at 32k it is about 13. Same model, same hardware, a quarter of the users, purely because of context length. If the engine also quantizes weights to INT8 and the cache to FP8 (8-bit formats, covered under Quantization trade-offs below), the 32k figure rises to about 31 (`kv_cache.py` reproduces all of these numbers).

Three consequences follow. Admission control must count KV memory, not requests: one 64k-context request is worth sixteen 4k requests (Chapter 29 owns the admission implementation; the budget comes from here). Context engineering (Chapter 5) is also capacity engineering: every thousand tokens trimmed from a prompt is cache given to another user. And an engine's "maximum model length" is a knob that trades concurrency for length, not a free parameter.

The formula is a floor. Real footprints add block metadata, fragmentation, draft-model caches, and per-request workspaces. Treat the estimate as the plausibility check and the load test as the truth.

### When the architecture changes the arithmetic

The formula above, and `kv_cache.py`, describe a dense model with conventional (grouped-query) attention in every layer. Many current open-weight models depart from that in ways that change both the fixed and the variable cost. Read the model's configuration file before you plan, because the model card's headline parameter count can mislead in either direction.

**Mixture-of-experts (MoE).** An MoE model replaces each feed-forward block with many "expert" blocks and a small router that sends each token to a few of them. The model card then quotes two numbers: total parameters (every expert) and active parameters (what one token passes through). They govern different resources. **Memory scales with total parameters**: every expert must be resident, because the next token may need any of them. **Compute per token, and decode bandwidth at small batch, scale with active parameters**: a single sequence reads only the experts its token was routed to. Take an illustrative MoE with 48B total and 8B active parameters. At BF16 its weights need about 96 GB and do not fit on the illustrative 80 GiB device, where the dense 8B model left 55 GiB for cache; at FP8 they take about 48 GB and leave much less cache than the dense model. Yet its single-stream TPOT resembles the dense 8B model's, since each step reads roughly 8B parameters' worth of weights.

Two consequences follow. First, the "batching is nearly free" argument weakens: as the batch grows, its tokens route to different experts, so a decode step reads more of the total weight set, and per-step time rises toward that of a dense model of the total size until the batch is large enough to amortize it. Second, the KV cache is untouched by the experts: it depends on the attention layers alone, so size it with the attention shape from the config, not the parameter count. MoE models suit fleets with plenty of memory and steady, high concurrency, and spread naturally across devices with expert parallelism (see the parallelism paragraph below). On a single small device they are often the wrong choice: you pay memory for the total and get the quality of something between the two numbers.

**Attention variants change the KV formula.** Three are common:

- **Multi-query and grouped-query attention** shrink `kv_heads`, which the formula already captures.
- **Sliding-window (local) attention** layers attend only to the last W tokens, so they keep at most W tokens of cache regardless of context length. Models that interleave local and global layers pay the full formula only for the global layers; for long contexts the real footprint can be a fraction of the naive estimate, provided the engine implements the window and does not allocate full-length cache for every layer.
- **Latent attention** (multi-head latent attention, used for example in the DeepSeek-V2 and V3 model families) caches one compressed latent vector per token per layer instead of separate keys and values for every head. Per token the cost becomes roughly `layers × latent_dim × bytes` (plus a small positional component), which removes the factor 2 and the `kv_heads × head_dim` product and is several times smaller than a grouped-query cache of similar model size.

Hybrid models that mix attention with linear-attention or state-space layers go further: those layers keep a fixed-size state per sequence that does not grow with context at all. In every case the practical rule is the same. Engines report the KV capacity they actually allocated at startup, as a number of tokens or blocks that fit; that number, divided by your p95 sequence length, is the concurrency ceiling to check your arithmetic against. `kv_cache.py` deliberately models only conventional attention; exercise K7 asks you to extend the reasoning.

### Static versus continuous batching

Early serving systems used **static batching**: wait until N requests have arrived, run them together, return when the longest finishes. Because output lengths vary, most of the batch sits idle waiting for the one request still generating, and a request arriving a millisecond after the batch started waits for the whole batch. Utilization is poor and TTFT is erratic.

**Continuous batching** schedules at the granularity of a decode step. At each step the engine admits new requests whose prompts fit in the remaining cache budget, runs one step for every active sequence, and retires those that produced an end token, freeing their cache blocks for the next arrival. This keeps the GPU full under mixed-length traffic and is why modern engines reach several times the throughput of static batching on the same hardware.

Scheduling policy still matters. A long prompt admitted into a busy batch does a large prefill that stalls everyone's decode step, so interactive users see a TPOT hiccup. Engines mitigate this by chunking long prefills across steps or separating prefill and decode priorities; at very large scale some deployments run the two phases on separate worker pools and ship KV state between them, an optimization called disaggregated prefill and decode, listed in the comparison table below; it is not a starting point. The knob you will touch is the maximum number of batched sequences or tokens, which trades throughput for TPOT. Tune it against your SLO, not the engine's default.

Paged attention is the memory-management idea underneath continuous batching: cache is allocated in fixed-size blocks mapped to sequences like virtual-memory pages, so sequences grow without reserving a contiguous maximum-length region and freed blocks are reused with little fragmentation. You do not configure it; you benefit from it and read its utilization metric.

### Prefix caching and prompt layout

When many requests share an identical token prefix, the KV cache computed for that prefix during one request's prefill can be reused for the next. Prefill for the shared part is skipped and TTFT drops, sometimes dramatically for a long system prompt or a cached document. Hosted providers expose the same idea as prompt caching with a per-token discount; self-hosted engines implement it as automatic prefix caching over the paged cache.

The application controls whether it helps. A hit requires an identical token prefix: same model, tokenizer, adapter, and text. Anything that varies early in the prompt breaks the match for everything after it. The layout rules from Chapter 5 and the caching layers in Chapter 30 follow directly: stable system prompt and tool definitions first, tenant and session context next, the user's message and freshly retrieved evidence last. A timestamp or request ID in the system prompt defeats caching entirely. Measure the hit rate before designing around it; a workload of unique long documents gets nothing from prefix caching except the memory it consumes.

Two cautions. Cached activations derive from user content; engines key on exact token match, which is safe for content, but timing can leak: a fast first token reveals that someone recently sent the same prefix, so review cross-tenant sharing with your security team (Chapter 26). And the prefix cache competes with active sequences for the same blocks, so a high hit rate under low load can vanish under high load.

### Quantization trade-offs

Quantization stores tensors in fewer bits. Three targets exist and should never be conflated.

**Weight-only quantization** (INT8, INT4, FP8) shrinks the model's fixed memory cost. The illustrative 8B model drops from about 16 GB (15 GiB) of weights at 16 bits to about 8 GB at 8 bits and 4 GB at 4 bits, before scales and metadata. The freed memory becomes KV cache, which becomes concurrency; and because decode is bandwidth-bound, reading half the bytes per step also speeds up single-stream TPOT when the kernels for that format are efficient on your hardware. A 4-bit format with a slow dequantization kernel can be slower than 8-bit, so the speedup is empirical.

**Activation quantization** (typically FP8 or INT8 for both weights and activations) also accelerates the arithmetic on hardware that supports those tensor-core formats, which matters for prefill. It is more sensitive to outlier channels and needs calibration data.

**KV-cache quantization** (FP8 or INT8 cache) halves the variable memory cost and therefore doubles concurrency at a given context. Quality effects are position- and task-dependent and show up most on long-context retrieval tasks, exactly the tasks a RAG system cares about.

One rule deserves emphasis: **measure task quality, not perplexity alone.** Perplexity is an aggregate language-modeling score that can move by a fraction of a percent while structured-output validity, tool-call argument accuracy, multilingual quality, or long-context faithfulness regress badly. Run the release evaluation suite (Chapters 24 and 25), sliced by task type, before and after any quantization change. If INT4 doubles concurrency but breaks structured output for one task family, it is not a win; if a small benchmark dip leaves the product evaluation unchanged and halves cost, it is.

The memory saving is usually worth more than the kernel speed. Doubling batch size on a bandwidth-bound decode nearly doubles throughput; a 20 percent faster kernel does not.

### Speculative decoding

Decode is sequential: one token per step per sequence, each step reading all the weights. Speculative decoding attacks that sequential bottleneck. A cheap **draft** mechanism proposes several future tokens; the full **target** model then verifies all of them in a single forward pass, which costs about the same as generating one token because the pass is bandwidth-bound anyway. The verifier accepts the longest prefix of the draft that passes the acceptance test (under greedy decoding, that matches the target's own choice) and generates one more token itself. Under the standard acceptance rule the output distribution is identical to the target model's; speculation changes speed, not results.

The speedup is governed by the **acceptance rate**. If the draft guesses four tokens and three are typically accepted, each expensive step advances about four positions instead of one. If the draft is poorly matched to the target or the text is unpredictable, most proposals are rejected and the system pays the draft cost for nothing; a bad draft makes the system slower. Acceptance is high for predictable text (code, structured output, boilerplate) and low for creative or novel content.

Drafts come in several forms. A separate small model from the same family is the classic choice and needs its own weights and cache. **Medusa** adds prediction heads to the target so it proposes several tokens from one hidden state, at the cost of training those heads. **EAGLE** predicts at the level of hidden features with a small auxiliary network and tends to reach higher acceptance than a tiny standalone draft. **N-gram speculation** proposes continuations from token patterns already in the prompt or output, costs almost nothing, and suits tasks that copy from context, such as extraction and code editing. All are engine configuration choices; benchmark TTFT and TPOT at realistic concurrency, because speculation helps most at low batch sizes and its benefit shrinks when the GPU is already busy.

### Serving many fine-tunes on one base: multi-adapter serving

Chapter 33 recommends parameter-efficient fine-tuning, usually LoRA: the base weights stay frozen and the fine-tune is a small set of low-rank matrices, the adapter. That shape has a serving consequence. Instead of one replica per fine-tuned model, an engine can load the base weights once and keep many adapters resident, applying each request's adapter inside the same batch with kernels designed for mixed-adapter batches. The client selects an adapter by model name, so to the application each adapter looks like a separate model on the same endpoint.

**When it pays.** Many fine-tunes with modest traffic each: one per tenant, per task family, or per language. Ten adapters on one base cost roughly one replica plus the adapters' memory (illustratively tens to hundreds of megabytes each, depending on rank and which layers they touch), where ten merged models would cost ten replicas, nine of them mostly idle. It also makes adapter rollout cheap: loading a new adapter version is a configuration change, not a new deployment.

**When not.** A single high-volume fine-tune is better merged into the base weights and served as a plain model, because applying an adapter at runtime adds a little work to every step. Full fine-tunes cannot share a base at all. And adapters for different base models, or the same base at different quantizations, need different pools.

**What to watch.** Engines cap the number of adapters resident on the GPU and the maximum adapter rank; a request for a cold adapter waits for it to load, which shows up as a TTFT outlier. The prefix cache is per adapter, because the same text produces different keys and values under different adapters, so ten adapters split the cache hit rate ten ways. Each adapter is its own model release: evaluate it on its own slice (Chapter 33), record the adapter name and version in every trace, and add a contract test that an unknown adapter name is rejected rather than silently served by the base model.

### Parallelism in one paragraph

When a model does not fit on one GPU, or one GPU cannot deliver the required throughput, the engine spreads work across devices. **Tensor parallelism** splits each matrix operation across GPUs and synchronizes after every layer, so it needs a fast interconnect and is used within one machine. **Pipeline parallelism** assigns layers to stages on different devices and passes activations along; it tolerates slower links but introduces idle bubbles. **Data parallelism** runs independent replicas of the whole model and is the right answer whenever the model fits on one device group, because replicas add capacity with no communication. **Expert parallelism** places the experts of a mixture-of-experts model on different devices and routes tokens to them, which requires all-to-all communication. The practical guidance for an application team is simple: fit the model on the smallest device group that holds weights plus a useful KV budget, scale with data-parallel replicas behind a load balancer, and treat multi-node tensor parallelism as a specialist's problem.

### Serving options compared

The table lists the common choices as options, not recommendations. The cells reflect engine documentation as of mid-2026, and features change quickly across versions; verify every cell against the version you deploy, and pin that version.

| Option | Target hardware | Batching | Prefix cache | Structured output | Quantization formats | Operational maturity and notes |
|---|---|---|---|---|---|---|
| vLLM | Data-center and workstation GPUs, several vendors | Continuous, paged KV | Automatic prefix caching | Grammar and JSON-schema constrained decoding via pluggable backends | Many weight-only and FP8 formats, KV quantization | Widely deployed OpenAI-compatible server; large model catalog; supports adapters, tensor and pipeline parallelism, speculative decoding; optional disaggregated prefill/decode at scale |
| SGLang | Data-center GPUs | Continuous, paged KV with radix-tree prefix sharing | Core design goal; strong for shared prefixes and multi-call agent programs | Constrained decoding built in, plus a frontend language for multi-step programs | Common weight-only and FP8 formats | OpenAI-compatible server; emphasizes scheduling and cache reuse for agentic and structured workloads; speculative decoding supported |
| TensorRT-LLM | One GPU vendor's hardware only | Continuous, in-flight batching | Supported | Supported via guided decoding integrations | Vendor-optimized FP8, INT8, INT4 kernels | Often the highest peak performance on its hardware in published benchmarks; requires a per-model build step and tighter coupling to the vendor stack; typically fronted by a separate inference server |
| llama.cpp (GGUF) | CPUs, consumer GPUs, laptops, edge; many backends | Limited parallel slots; not a high-concurrency design | Prompt cache per slot | Grammar-constrained sampling | GGUF with many low-bit schemes; GGUF is a file format, llama.cpp is the runtime | Excellent for local, offline, and single-user deployments; OpenAI-compatible server included; not intended for server-scale concurrency |
| Managed open-weight endpoints | Provider's | Provider-managed | Varies by provider | Varies; often JSON-schema mode | Provider's choice; ask which is served | Open weights and often your own adapters without running GPUs; per-token or dedicated-capacity billing; engine and quantization upgrades on the provider's schedule |
| Hosted endpoints | Provider's | Provider-managed | Prompt caching with discounts, provider-specific rules | Native structured outputs on most providers | Not exposed | No operations; regional options; capacity and latency are observed rather than controlled; model behavior can change on provider schedule |

Benchmark the shortlist on your exact model, your GPU generation, your prompt and output distributions, and your structured-output needs. A generic leaderboard measures none of those.

## How it works

### A request's path through a serving stack

A production stack has more parts than the engine. The application's gateway (Chapter 3) sends an HTTP request to a load balancer that routes it to a replica, usually by least outstanding requests or by prefix affinity so cache hits land on the replica that holds the prefix. The replica's admission controller checks that the request's token budget fits in free KV memory and either queues it, admits it, or rejects it fast with a retryable status. The tokenizer converts text to IDs. The scheduler places the prompt into the next decode step's batch for prefill, possibly chunked. The first token streams back; the sequence stays in the batch for every decode step until it emits an end token, hits `max_tokens`, or the client disconnects. On completion or cancellation its cache blocks return to the pool, and the response's usage block reports token counts for cost accounting and tracing.

### Capacity planning with Little's Law

Little's Law is the queueing identity behind every capacity estimate; Chapter 35 applies it to whole-system sizing, and this section applies it to the serving fleet. For any stable system,

```
L = λ × W
```

where L is the average number of requests in the system (queued or running), λ is the arrival rate, and W is the average time a request spends in the system. It holds regardless of distribution, scheduling, or batching. A first use: peak arrivals of 20 requests per second with a 4 second mean end-to-end time means about 80 requests in flight. If each needs 1.5 GiB of KV cache at its typical length, the fleet needs 120 GiB of cache for those requests alone, before weights and headroom. Capacity planning starts there, not from a single-request benchmark.

The second idea is the **queueing knee**. For an idealized single-server queue, mean response time is `S / (1 − ρ)`, where S is the service time and ρ the utilization (the fraction of time the server is busy). At 50 percent utilization requests take twice the service time; at 90 percent, ten times; at 99 percent, a hundred times. Real LLM servers batch and have heavy-tailed service times, so the real curve is not that formula, but it has the same shape: flat, then vertical. The point where it turns is the knee, and the operating point (the load you plan to run at, chosen in the protocol below) must sit before it. This is why **headroom** is not waste. A replica loaded to 60 percent of its measured goodput capacity has room for bursts, for the long-tail prompt, and for the replica next to it failing; one loaded to 90 percent has already crossed into the region where p99 is unbounded.

**Worked example (illustrative).** Northwind Assist's RAG endpoint peaks at 6 requests per second. Mean input is 3,000 tokens (p95 6,000), mean output 300 tokens (p95 600), mean E2E 4 seconds. A load test of one replica of the 8B model found it sustains 1,500 output tokens per second and 20,000 prefill tokens per second while meeting the p95 TTFT and completion SLOs.

Little's Law puts 24 requests in flight at peak. Decode demand is 6 × 300 = 1,800 tokens per second; at a 60 percent target utilization one replica is good for 900, so decode needs 2.0 replicas. Prefill demand is 18,000 tokens per second against 12,000 usable, so 1.5 replicas. KV demand is 24 requests at a p95 sequence of 6,600 tokens, about 0.8 GiB each, 19 GiB total, well inside the 55 GiB cache budget of a single replica computed in the KV section (0.59 replicas at 60 percent utilization). Decode is the binding constraint; round up to 2 and add one replica for failover and rolling upgrades: 3 replicas. `capacity.py` encodes this computation, and if you change the p95 context to 30,000 tokens the binding constraint flips to KV, which is exactly the lesson of the KV section.

### A load test and benchmark protocol

A benchmark that cannot be reproduced is an anecdote. Record, for every run, the model and its exact weights or quantization, the engine and version, the hardware, the scheduler settings (maximum batched sequences and tokens, chunked prefill, speculative configuration), the sampling parameters, and the prompt and output distributions. Then follow this protocol.

1. **Warm up.** The first requests pay for kernel compilation, graph capture, and cold caches. Discard them.
2. **Use realistic distributions.** Sample prompts from production traces or a representative mix: short chat turns, long RAG prompts, structured extraction with tool schemas. Sample `max_tokens` from the observed output distribution. A benchmark of identical 128-token prompts with 128-token outputs tells you nothing about mixed traffic, because queueing and KV pressure come from the tails.
3. **Sweep concurrency.** Run a closed loop at 1, 2, 4, 8, 16, 32 concurrent clients (and beyond, until something breaks). Closed loop means each client sends its next request when the previous completes, which keeps offered load bounded and the sweep interpretable.
4. **Record per request:** input and output tokens, concurrency level, TTFT, TPOT, E2E, success or error or cancel status. Record per level: GPU memory and utilization, KV-cache utilization, batch size, and queue depth from the engine's metrics endpoint.
5. **Plot two curves.** p50 and p95 TTFT against offered load, and throughput (requests per second or output tokens per second) against p95 E2E. Throughput rises and flattens; TTFT stays flat and then climbs. The knee is where p95 TTFT leaves the floor.
6. **Pick the operating point.** The highest concurrency at which p95 TTFT and p95 E2E meet the SLO with no errors. Its goodput is your capacity per replica. Plan replicas from that number with headroom, as in the worked example.
7. **Repeat for every variant** you are comparing: quantization, engine version, scheduler setting, speculative decoding on or off. Keep workload and SLO constant and compare goodput at the operating point, not peak throughput.

The load generator in this chapter implements steps 1 to 3, the per-request half of step 4, and step 6, and prints the data behind step 5's curves as a text table; exercises P1 and P2 add the engine metrics and the plots.

## Architecture

The first diagram shows the request path and where each metric is measured. The trust boundary matters: everything past the gateway is your infrastructure when self-hosting, and the application's authorization decisions (Chapter 16) must already have been made before a request reaches the engine.

```mermaid
flowchart LR
    subgraph App["Application (trusted)"]
        GW["ModelGateway: retry, fallback, cost, trace"]
    end
    subgraph Serving["Serving platform (self-hosted)"]
        LB["Load balancer: least-load or prefix affinity"]
        ADM["Admission control: KV budget check"]
        Q["Queue"]
        SCH["Scheduler: continuous batching"]
        ENG["Engine replica: weights + paged KV cache"]
        MET["Metrics: queue depth, KV util, batch size"]
    end
    GW -->|"HTTP, stream=true"| LB --> ADM
    ADM -->|"fits"| Q --> SCH --> ENG
    ADM -->|"reject fast, retryable"| GW
    ENG -->|"first token: TTFT"| GW
    ENG -->|"token deltas: TPOT"| GW
    ENG -->|"usage, finish: E2E"| GW
    SCH -.-> MET
    ENG -.-> MET
```

The second diagram shows one request's life inside a replica, with two concurrent sequences sharing decode steps. Request B arrives during A's decode and is admitted at a step boundary, which is continuous batching; A's cache is freed the moment it finishes.

```mermaid
sequenceDiagram
    participant C as Clients
    participant S as Scheduler
    participant G as GPU
    C->>S: Request A (6k-token prompt)
    S->>G: Prefill A (compute-bound)
    G-->>C: A first token (TTFT A)
    loop Decode steps, one token per sequence per step
        S->>G: Step: A
        G-->>C: A token
    end
    C->>S: Request B (short prompt) arrives mid-stream
    S->>G: Prefill B, batched with A's decode
    G-->>C: B first token (TTFT B includes queue wait)
    loop Decode steps, batched
        S->>G: Step: A + B (one weight read)
        G-->>C: A token, B token
    end
    G-->>C: A finishes, KV blocks freed
    loop Decode steps
        S->>G: Step: B
        G-->>C: B token
    end
    G-->>C: B finishes (E2E B)
```

## Implementation

Five modules, all under `book/projects/examples/ch34/`. The listings below are excerpts that carry the ideas; every file is complete on disk, including each module's `__main__` demo and the full test file. The calculators and the load generator depend only on `httpx` and `pydantic`. `local_target.py` is the bridge to `aie_core`: it imports the library lazily, so the capacity helpers stay importable on their own, and its tests drive a real `OpenAICompatibleClient` and `ModelGateway` against the in-process fake server.

```
book/projects/examples/ch34/
├── kv_cache.py        # KV sizing, weight memory, concurrency estimate
├── capacity.py        # Little's Law, M/M/1 knee, headroom, replica estimate
├── loadtest.py        # async closed-loop load generator with streaming TTFT/TPOT
├── fake_server.py     # in-process OpenAI-compatible SSE fake for offline tests
├── local_target.py    # ServingTarget capability envelope + aie_core client factory
├── conftest.py        # registers the integration marker
└── test_ch34.py
```

Run the tests from the repository root:

```bash
.venv/bin/python -m pytest book/projects/examples/ch34 -q
```

Run the load generator against any OpenAI-compatible server (the engine's own `/v1/chat/completions`):

```bash
cd book/projects/examples/ch34
python loadtest.py --base-url http://localhost:8000/v1 --model my-model \
    --levels 1,2,4,8,16,32 --requests 32 --prompt-file prompts.txt --ttft-slo 2 --e2e-slo 8
```

### KV-cache sizing

```python
# path: book/projects/examples/ch34/kv_cache.py (excerpt; full file on disk)
class ModelShape(BaseModel):
    # ... docstring: read these from the model's config file
    name: str = "illustrative-8b"
    layers: PositiveInt
    kv_heads: PositiveInt
    head_dim: PositiveInt
    params_billion: float = Field(gt=0, description="total parameters, in billions")


def kv_bytes_per_token(shape: ModelShape, kv_bytes: float = 2.0) -> int:
    """Bytes of KV cache that one token occupies across all layers.

    Formula: 2 (keys and values) * layers * kv_heads * head_dim * bytes_per_element.
    """
    return int(2 * shape.layers * shape.kv_heads * shape.head_dim * kv_bytes)


def kv_bytes_per_sequence(shape: ModelShape, tokens: int, kv_bytes: float = 2.0) -> int:
    """KV cache for one sequence of ``tokens`` tokens (prompt plus generated so far)."""
    if tokens < 0:
        raise ValueError("tokens must be non-negative")
    return kv_bytes_per_token(shape, kv_bytes) * tokens


def weight_bytes(shape: ModelShape, weight_bytes_per_param: float = 2.0) -> int:
    """Memory for the weights alone. Scales and metadata for quantized formats add a few percent."""
    return int(shape.params_billion * 1e9 * weight_bytes_per_param)


# ... ConcurrencyEstimate (fields and summary) on disk


def max_concurrent_sequences(
    shape: ModelShape,
    tokens_per_sequence: int,
    gpu_memory_bytes: int,
    precision: Precision | None = None,
    headroom_fraction: float = 0.10,
    runtime_overhead_bytes: int = 2 * GiB,
    tensor_parallel: int = 1,
) -> ConcurrencyEstimate:
    """How many sequences of a given length fit in KV memory once weights are resident.

    ``tensor_parallel`` devices share weights and cache evenly, so the estimate is for the whole
    group. ``headroom_fraction`` is memory you refuse to plan into: fragmentation and bursts.
    ``runtime_overhead_bytes`` is per device (CUDA context, workspaces), so it scales with the group.
    """
    precision = precision or Precision()
    if not 0 <= headroom_fraction < 1:
        raise ValueError("headroom_fraction must be in [0, 1)")
    total = gpu_memory_bytes * tensor_parallel
    weights = weight_bytes(shape, precision.weight_bytes)
    headroom = int(total * headroom_fraction)
    runtime = runtime_overhead_bytes * tensor_parallel
    usable = total - weights - headroom - runtime
    per_seq = kv_bytes_per_sequence(shape, tokens_per_sequence, precision.kv_bytes)
    max_seq = 0 if usable <= 0 or per_seq == 0 else usable // per_seq
    return ConcurrencyEstimate(
        gpu_memory_bytes=total,
        weight_bytes=weights,
        headroom_bytes=headroom,
        runtime_overhead_bytes=runtime,
        usable_kv_bytes=max(usable, 0),
        per_sequence_bytes=per_seq,
        max_sequences=int(max_seq),
    )


def tokens_that_fit(shape: ModelShape, kv_budget_bytes: int, kv_bytes: float = 2.0) -> int:
    """Inverse question: given a KV budget, how many total tokens can be resident at once?"""
    per_token = kv_bytes_per_token(shape, kv_bytes)
    return 0 if kv_budget_bytes <= 0 else kv_budget_bytes // per_token
```

Running `python kv_cache.py` (its `__main__` block is on disk) prints the numbers used earlier in the chapter (illustrative hardware):

```
illustrative-8b: 128.0 KiB per token
    8192 tokens -> 1.00 GiB per sequence
   32768 tokens -> 4.00 GiB per sequence
  ctx 8192:  ... 55.1 GiB for KV; 1.00 GiB per sequence -> about 55 concurrent sequences
  ctx 32768: ... 55.1 GiB for KV; 4.00 GiB per sequence -> about 13 concurrent sequences
  ctx 32768 with INT8 weights + FP8 cache: ... 62.5 GiB for KV; 2.00 GiB per sequence -> about 31
```

### Capacity helpers

```python
# path: book/projects/examples/ch34/capacity.py (excerpt; full file on disk)
def in_flight(arrival_rate_per_s: float, mean_time_in_system_s: float) -> float:
    """L = lambda * W. Average number of requests inside the system (queued or running)."""
    _non_negative(arrival_rate_per_s, "arrival_rate_per_s")
    _non_negative(mean_time_in_system_s, "mean_time_in_system_s")
    return arrival_rate_per_s * mean_time_in_system_s


# ... arrival_rate, mean_time_in_system, utilization on disk


def mm1_response_time(service_time_s: float, rho: float) -> float:
    """Mean time in system for an M/M/1 queue: S / (1 - rho).

    Idealized (Poisson arrivals, exponential service, one server). Real LLM servers batch and
    have heavy-tailed service times, so the knee is sharper and arrives earlier. The shape of
    the curve is still the right intuition: latency is flat, then vertical.
    """
    if not 0 <= rho < 1:
        raise ValueError("rho must be in [0, 1) for a stable queue")
    return service_time_s / (1.0 - rho)


# ... headroom, Workload, ReplicaProfile, ReplicaEstimate on disk


def replicas_needed(
    workload: Workload,
    profile: ReplicaProfile,
    target_utilization: float = 0.6,
    failover_replicas: int = 1,
) -> ReplicaEstimate:
    # ... docstring and argument validation
    decode_demand = workload.peak_requests_per_s * workload.mean_output_tokens
    prefill_demand = workload.peak_requests_per_s * workload.mean_input_tokens
    decode_r = decode_demand / (profile.decode_tokens_per_s * target_utilization)
    prefill_r = prefill_demand / (profile.prefill_tokens_per_s * target_utilization)

    l_in_flight = in_flight(workload.peak_requests_per_s, workload.mean_e2e_s)
    # Size KV for the p95 sequence when known: tail requests are what exhaust cache.
    seq_tokens = (workload.p95_input_tokens or workload.mean_input_tokens) + (
        workload.p95_output_tokens or workload.mean_output_tokens
    )
    kv_per_request = seq_tokens * profile.kv_bytes_per_token
    kv_r = (l_in_flight * kv_per_request) / (profile.kv_budget_bytes * target_utilization)

    constraints = {"decode": decode_r, "prefill": prefill_r, "kv": kv_r}
    binding = max(constraints, key=constraints.__getitem__)
    recommended = math.ceil(constraints[binding]) + failover_replicas
    return ReplicaEstimate(
        decode_bound_replicas=decode_r,
        prefill_bound_replicas=prefill_r,
        kv_bound_replicas=kv_r,
        target_utilization=target_utilization,
        failover_replicas=failover_replicas,
        recommended_replicas=max(recommended, 1),
        in_flight_requests=l_in_flight,
        binding_constraint=binding,
    )
```

### The load generator

```python
# path: book/projects/examples/ch34/loadtest.py (excerpt; full file on disk)
async def run_one(
    client: httpx.AsyncClient, cfg: LoadTestConfig, prompt: str, max_tokens: int, concurrency: int
) -> RequestResult:
    """One streamed chat completion with TTFT, TPOT, and end-to-end timing."""
    body = {
        "model": cfg.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": cfg.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    result = RequestResult(concurrency=concurrency, ok=False, prompt_chars=len(prompt), max_tokens=max_tokens)
    start = time.perf_counter()
    first_token_at: float | None = None
    last_token_at: float | None = None
    tokens = 0
    reported_tokens: int | None = None
    try:
        async with client.stream("POST", "/chat/completions", json=body, timeout=cfg.timeout_s) as resp:
            result.status_code = resp.status_code
            if resp.status_code != 200:
                await resp.aread()
                result.error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                result.e2e_s = time.perf_counter() - start
                return result
            async for line in resp.aiter_lines():
                chunk = parse_sse_line(line)
                if chunk is None:
                    continue
                usage = chunk.get("usage")
                if usage and usage.get("completion_tokens") is not None:
                    reported_tokens = int(usage["completion_tokens"])
                if _delta_text(chunk):
                    now = time.perf_counter()
                    if first_token_at is None:
                        first_token_at = now
                    last_token_at = now
                    tokens += 1
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        result.e2e_s = time.perf_counter() - start
        return result
    # asyncio.CancelledError is not caught: Ctrl-C or an outer deadline must stop the sweep.

    end = time.perf_counter()
    result.e2e_s = end - start
    # Content deltas are an approximation of tokens; prefer the server's usage count if given.
    result.output_tokens = reported_tokens if reported_tokens is not None else tokens
    if first_token_at is None:
        result.error = "no content tokens received"
        return result
    result.ttft_s = first_token_at - start
    if last_token_at is not None and result.output_tokens > 1:
        result.tpot_s = (last_token_at - first_token_at) / (result.output_tokens - 1)
    result.ok = True
    return result

# ... run_level (closed-loop workers draining a shared queue) on disk

def summarize(results: list[RequestResult], wall_s: float, cfg: LoadTestConfig) -> LevelSummary:
    ok = [r for r in results if r.ok]
    ttft = [r.ttft_s for r in ok if r.ttft_s is not None]
    tpot = [r.tpot_s for r in ok if r.tpot_s is not None]
    e2e = [r.e2e_s for r in ok]
    passed = sum(1 for r in results if r.meets_slo(cfg))
    wall = max(wall_s, 1e-9)
    return LevelSummary(
        # ... nearest-rank p50/p95/p99 for TTFT, TPOT, and E2E
        requests_per_s=len(ok) / wall,
        output_tokens_per_s=sum(r.output_tokens for r in ok) / wall,
        slo_pass_fraction=passed / len(results) if results else 0.0,
        goodput_requests_per_s=passed / wall,
    )

# ... sweep (warm-up, pooled client, one run_level per concurrency level) on disk

def find_operating_point(summaries: list[LevelSummary], cfg: LoadTestConfig) -> LevelSummary | None:
    """Highest concurrency whose p95 TTFT and p95 E2E meet the SLO with zero errors."""
    passing = [
        s for s in summaries
        if s.errors == 0 and s.ttft_p95 <= cfg.ttft_slo_s and s.e2e_p95 <= cfg.e2e_slo_s
    ]
    return max(passing, key=lambda s: s.concurrency) if passing else None
```

### The offline fake server

```python
# path: book/projects/examples/ch34/fake_server.py (excerpt; full file on disk)
@dataclass
class FakeServer:
    capacity: int = 4  # concurrent sequences the fake "GPU" decodes without queueing
    prefill_s: float = 0.01  # fixed prompt-processing delay
    tpot_s: float = 0.002  # per-token decode delay
    queue_slot_s: float = 0.02  # scheduler overhead added when a request had to wait for a slot
    fail_above: int | None = None  # return HTTP 429 when more than this many are active
    # ... counters and the per-event-loop slot semaphore on disk

    # ... handle(): routes /chat/completions, returns 429 above fail_above, streams _SSEStream

class _SSEStream(httpx.AsyncByteStream):
    # ...
    async def __aiter__(self) -> AsyncIterator[bytes]:
        s = self.server
        s.active += 1
        s.peak_active = max(s.peak_active, s.active)
        slots = s.slots()
        try:
            queued = slots.locked()                              # every decode slot is busy
            async with slots:                                    # wait for a slot: the real queue
                s.decoding += 1
                s.peak_decoding = max(s.peak_decoding, s.decoding)
                try:
                    await asyncio.sleep(s.prefill_s + (s.queue_slot_s if queued else 0.0))  # prefill => TTFT
                    created = int(time.time())
                    for i in range(self.n_tokens):
                        if i:
                            await asyncio.sleep(s.tpot_s)
                        yield _chunk(self.model, created, f"tok{i} ")
                    usage = {"prompt_tokens": 10, "completion_tokens": self.n_tokens, "total_tokens": 10 + self.n_tokens}
                    yield _chunk(self.model, created, None, finish="stop", usage=usage)
                    yield b"data: [DONE]\n\n"
                finally:
                    s.decoding -= 1
        finally:
            s.active -= 1
```

### Serving targets and the `aie_core` client

```python
# path: book/projects/examples/ch34/local_target.py (excerpt; full file on disk)
class ServingTarget(BaseModel):
    name: str
    base_url: str
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    max_context_tokens: int = Field(gt=0)
    supports_tools: bool = True
    supports_response_schema: bool = True
    supports_stream_usage: bool = True
    engine: str | None = None  # e.g. "vllm", "sglang", "llama.cpp", "hosted"; recorded in traces


# ... RequestEnvelope: input_tokens, max_output_tokens, needs_tools, needs_response_schema


def check_fit(target: ServingTarget, env: RequestEnvelope, reserve_tokens: int = 64) -> list[str]:
    """Return a list of human-readable problems; empty means the request can be sent as is."""
    problems: list[str] = []
    needed = env.input_tokens + env.max_output_tokens + reserve_tokens
    if needed > target.max_context_tokens:
        problems.append(
            f"context: needs {needed} tokens (input {env.input_tokens} + output {env.max_output_tokens} "
            f"+ reserve {reserve_tokens}) but {target.name} allows {target.max_context_tokens}"
        )
    if env.needs_tools and not target.supports_tools:
        problems.append(f"tools: {target.name} does not support tool calling")
    if env.needs_response_schema and not target.supports_response_schema:
        problems.append(f"schema: {target.name} does not support JSON-schema constrained output")
    return problems


# ... choose_target and envelope_for on disk


def make_client(
    target: ServingTarget,
    *,
    timeout_s: float = 60.0,
    transport: Any = None,
    async_transport: Any = None,
) -> Any:
    # ... docstring; aie_core is imported lazily
    from aie_core.llm.providers import OpenAICompatibleClient  # noqa: PLC0415

    client = OpenAICompatibleClient(
        base_url=target.base_url,
        api_key=os.environ.get(target.api_key_env, "not-needed"),
        default_model=target.model,
        provider=f"{target.engine or 'openai-compatible'}:{target.name}",
        timeout_s=timeout_s,
        transport=transport,
        async_transport=async_transport,
    )
    client.supports_response_schema = target.supports_response_schema
    return client
```

## Code walkthrough

**`kv_cache.py`** is the formula with guard rails. `ModelShape` holds the four architecture facts that matter and tells you where to find them in a model's configuration file. `max_concurrent_sequences` subtracts weights, a headroom fraction, and a fixed runtime allowance from device memory and divides by the per-sequence cache. The headroom default of 10 percent is a planning posture, not a measurement: it is memory you refuse to count on. The `tensor_parallel` argument pools memory across a device group, which is the only parallelism arithmetic an application engineer needs; the runtime allowance is per device, so it scales with the group. `tokens_that_fit` answers the inverse question an admission controller asks: given the free cache right now, what is the longest request I can admit?

**`capacity.py`** separates what is exact from what is a sketch. The three Little's Law functions (`in_flight` above, `arrival_rate` and `mean_time_in_system` on disk) are exact identities and are safe to use on any measured system. `mm1_response_time` is labeled an idealization in its docstring, and it exists to produce the knee curve, not to predict your server. `replicas_needed` computes three independent constraints and reports which one binds, because the remedy differs: a decode-bound fleet wants more replicas or quantized weights, a prefill-bound fleet wants shorter prompts or prefix caching, a KV-bound fleet wants shorter contexts, cache quantization, or a GQA model. The `ReplicaProfile` docstring (on disk) insists that its token rates come from a load test at the operating point, not from a spec sheet, because a spec sheet's tokens per second is measured past the knee.

**`loadtest.py`** does the one thing most ad hoc benchmarks skip: it streams. `run_one` requests `stream_options.include_usage` so the server's final chunk reports the real completion token count; content deltas are counted as a fallback because servers may coalesce tokens into one chunk. TTFT is measured to the first non-empty content delta, not the first response byte, because engines send role-only or empty deltas first. TPOT is the span between first and last content token divided by tokens minus one. Errors become result rows with `ok=False` rather than exceptions, so an overloaded server shows up as an error rate instead of aborting the sweep; cancellation is not caught, so Ctrl-C or an outer deadline stops it. `run_level` (on disk) runs a closed loop: `concurrency` workers drain a shared queue of jobs, each starting its next request when the previous one finishes. `sweep` (on disk) warms up and then runs one level after another; the client's connection pool is sized to the highest concurrency level, because a default pool of 100 would queue requests on the client and report the wait as server TTFT. `summarize` computes nearest-rank percentiles and goodput as requests that met both SLOs divided by wall time; `find_operating_point` applies the protocol's rule: highest concurrency with zero errors and both p95s under the SLO.

Against the fake server configured with a capacity of 8 (prefill 0.02 s, 5 ms per token, 32 or 64 output tokens, 32 requests per level, and SLOs scaled to the fake: 0.15 s TTFT and 1 s end to end), the sweep produces this table. The timings are fake and vary a little between runs; the shape is what you will see against real hardware, and `test_walkthrough_shape_has_a_knee_at_capacity` checks it:

```
 conc    n  err  ttft p50  ttft p95  tpot p50  e2e p50  e2e p95   req/s    tok/s   slo%  goodput
------------------------------------------------------------------------------------------------
    1   32    0     0.022     0.023    0.0058     0.20     0.40    3.51    161.6  100.0     3.51
    4   32    0     0.021     0.022    0.0058     0.38     0.41   12.56    627.9  100.0    12.56
    8   32    0     0.021     0.022    0.0057     0.21     0.38   27.02   1161.9  100.0    27.02
   16   32    0     0.253     0.433    0.0058     0.60     0.79   23.02   1150.8   25.0     5.75
   32   32    0     0.423     0.856    0.0059     0.61     1.20   22.78   1025.0   25.0     5.69
```

Read it the way you will read a real one. TPOT is flat at every level: the GPU did not get slower. Throughput climbs to concurrency 8 and then levels off, because the fake decodes at most 8 sequences at once and the rest wait for a slot. TTFT is flat until capacity, then p95 jumps about twentyfold, and goodput collapses from 27.0 to 5.8 requests per second even though raw throughput barely moved. The operating point is 8. A benchmark that reported only the 1,162 tokens per second peak would have been accurate and would have hidden the collapse.

**`fake_server.py`** implements the server-sent-events framing of an OpenAI-compatible endpoint with a per-token sleep so TTFT and E2E differ, a real capacity limit (at most `capacity` sequences decode at once; the rest wait for a slot) so TTFT grows and throughput levels off when more requests are active than the fake can batch, and a `fail_above` knob that returns HTTP 429 to exercise the error path. As an `httpx.MockTransport`, it lets the generator run its real HTTP and SSE parsing code with no sockets.

**`local_target.py`** is the routing side of self-hosting. `ServingTarget` records what an endpoint can do; `check_fit` returns every problem rather than the first so the trace (Chapter 31) explains a reroute; `choose_target` (on disk) walks a preference list. `envelope_for` (on disk) computes the envelope from the `CompletionRequest` the application already built, using `aie_core`'s token counter, so routing needs no second representation of the request. `make_client` builds the book's `OpenAICompatibleClient` with the target's `base_url` and two settings that matter in production: `provider` becomes `engine:target`, so every gateway span and cost report says which deployment answered, and `supports_response_schema` mirrors the target, so `complete_structured` switches to prompt-and-repair on an engine without schema mode instead of sending a `response_format` it will reject. That is the entire integration: the same `CompletionRequest`, the same `ModelGateway`, the same tests. One test streams through a gateway into the fake server and checks the span's provider; another runs `complete_structured` against a mocked schema-less target and checks that no `response_format` was sent.

### What changes when you route between hosted and local

The transport does not change; the envelope does. When a request that ran on a hosted endpoint is routed to a self-hosted engine, check five things before you assume parity.

**Tool support.** Tool calling on open models depends on the engine's parser for that model family's tool-call format and on the model having been trained for it. A target that reports `supports_tools=False` should receive requests with `tools=None`, and the application should fall back to a text-protocol tool format or route tool-using requests elsewhere. Test with the actual tool definitions from Chapter 16; argument formatting errors are common and specific to model and parser version.

**Schema support.** Constrained decoding for JSON schemas is widely available on self-hosted engines, often more strictly than hosted structured outputs, but the supported subset of JSON Schema varies (recursive schemas, `anyOf`, string formats). `complete_structured` in `aie_core` already falls back to prompt-and-repair when `response_schema` is unsupported; make sure the target's capability flag is accurate so it does.

**Context length.** A self-hosted engine's maximum model length is a deployment knob chosen for concurrency, often far below the model's trained maximum and far below a hosted endpoint's limit. `check_fit` reserves a margin because tokenizers differ between what the application counted with `count_tokens` and what the engine counts. Requests that do not fit should be compacted (Chapter 5) or routed, never silently truncated.

**Log-probabilities.** Many self-hosted engines return token log-probabilities on request, which hosted endpoints may restrict. They are the cheapest confidence signal a classifier cascade has (Chapter 33). The neutral `CompletionRequest` has no field for them, so build the target's client with `extra_body={"logprobs": True}` and read them from `completion.raw`; `extra_body` never overrides a field the adapter sets. Treat the numbers as engine- and quantization-specific: recalibrate the escalation threshold after any engine or quantization change.

**Model identity and behavior.** The model name the server expects is whatever the operator registered; the behavior is whatever quantization and engine version are deployed. `make_client` already puts the engine and target into the span's `provider`; Production considerations lists the remaining trace fields.

## Production considerations

**Latency.** Set two SLOs, TTFT and completion, and alert on p95 of each per replica. Alert on queue depth and KV-cache utilization as leading indicators, because they rise before the latency percentiles do. Autoscaling on GPU utilization alone reacts too late: a GPU at 70 percent utilization can already be past the knee if its cache is full. Scale on queue depth, admission rejections, and SLO violations, and add replicas before the knee, not at it.

**Cost model.** For self-hosting, cost per useful token is `(GPU hours × hourly cost + operations) / tokens served within SLO`. The denominator makes or breaks the case. Illustrative arithmetic: a GPU at 3.00 USD per hour running the 8B model at its operating point of 1,500 output tokens per second could produce 5.4 million output tokens per hour, about 0.55 USD per million if saturated around the clock. Real traffic has a daily cycle; at 35 percent average utilization the same GPU produces 1.9 million useful tokens per hour and the cost becomes about 1.60 USD per million, before engineer time, the failover replica that is idle by design, and load tests. Compare that to the per-token price you actually pay for the same traffic with caching discounts applied. The comparison favors self-hosting for sustained, high-utilization workloads or when a non-cost factor decides. Chapter 30 owns the full cost model; feed it goodput from your load test, not peak throughput.

**Health checks and rolling upgrades.** A replica's readiness check must do a real tiny completion, not just return 200 from the HTTP server: an engine can accept connections while its weights are still loading or after its GPU context has died. During a rolling upgrade, drain a replica by stopping admission and waiting for in-flight sequences to finish or hit a drain timeout; a hard kill discards every active user's generation. Keep at least one replica of headroom permanently so an upgrade never drops capacity below the operating point.

**Cancellation.** When a client disconnects or an agent loop abandons a step, the engine must notice and free that sequence's KV blocks. The zombie-generation failure mode below gives the test; some gateway configurations keep the upstream connection open after the client has gone, and the GPU keeps generating for nobody. Chapter 29 covers timeouts and cancellation plumbing at the gateway.

**Model version in traces.** Every span for a self-hosted call should carry the model name, weights or quantization identifier, adapter name and version when one is used, engine name and version, and the replica. Outputs change when any of these change, and without the fields in the trace you will debug a prompt for a day before discovering the operator upgraded the engine.

**Quality suite after every engine or quantization change.** Treat a change of quantization, engine version, kernel configuration, speculative decoding setting, or even maximum batch size as a model release. Rerun the golden-set evaluation and the slice reports from Chapters 24 and 25 before promoting it. Some settings change sampling numerics enough to alter outputs at temperature zero.

**Security.** The engine is now inside your trust boundary, which removes a third party and adds responsibilities: authenticate calls to the engine even on an internal network, because an OpenAI-compatible server will happily serve anyone; isolate tenants at the gateway, since the engine does not know what a tenant is; review prefix-cache behavior for cross-tenant timing leakage; and keep model weights, which may be licensed, out of public buckets.

## Common mistakes

**Planning from peak tokens per second.** Dividing an engine's advertised throughput by tokens per request gives a capacity you can never run at. Use goodput at the operating point, then add headroom.

**Benchmarking with uniform prompts.** Identical 128-token prompts hide the queueing and KV pressure that long-tail requests create. Sample from production traces.

**Measuring TTFT without streaming.** A non-streaming benchmark reports only E2E, so you cannot see the metric that saturates first.

**Treating "the model fits" as "the users fit."** Weight memory is the fixed cost; the KV cache budget is what determines concurrency, and it depends on your context lengths.

**Sizing a mixture-of-experts model by its active parameters.** The active count predicts single-stream speed; the total count decides whether the weights fit and how much memory is left for cache.

**Self-hosting a spiky workload.** A GPU busy two hours a day costs more per useful token than the API; do the cost arithmetic before buying hardware.

**Forgetting the failover replica.** N replicas sized exactly to peak become N minus one during every rolling upgrade and hardware fault.

## Failure modes

Each failure below names what breaks, how it looks in telemetry, and how to test for it before production does.

**KV exhaustion under long contexts.** A burst of long-context requests fills the cache; new requests queue or are preempted mid-generation, and TTFT climbs while TPOT stays flat. Telemetry: KV utilization near 100 percent, queue depth rising, preemption or recompute counters incrementing on the engine. Test: run the load generator with a prompt mix whose p95 is your real p95, not your mean, and confirm admission rejects fast rather than queueing without bound.

**Head-of-line blocking by a long prefill.** One 30k-token prompt enters a busy batch and every active user's next token is delayed by its prefill. Telemetry: periodic TPOT spikes correlated with admissions of long prompts. Test: mix a few very long prompts into a stream of short ones and plot per-request TPOT over time; enable chunked prefill or a prompt-length cap if the spikes breach the SLO.

**Quantization regression on one task family.** Aggregate quality looks unchanged but structured-output validity drops or tool arguments acquire formatting errors. Telemetry: repair-loop rate in `complete_structured`, tool-call validation failures, sliced evaluation scores. Test: the release suite with slices, run before promotion.

**Prefix cache miss storm.** A prompt change moves a variable field earlier in the prompt; cache hit rate drops to zero and TTFT doubles across the fleet. Telemetry: engine prefix-cache hit ratio, TTFT p50 step change aligned with a prompt registry version bump (Chapter 4). Test: assert in CI that the registered system prompt's first N tokens are stable across the versions you intend to be cache-compatible.

**Zombie generations after client disconnect.** Abandoned streams keep decoding; KV utilization stays high with no corresponding client connections. Telemetry: active sequences on the engine exceed open connections at the gateway. Test: open streams, close them early, watch the engine's active-sequence metric fall within a bounded time.

**Replica flapping during rolling upgrade.** Readiness passes before weights finish loading and the load balancer sends traffic to a replica that times out every request. Telemetry: error spikes aligned with deployment events, one replica with zero completions. Test: a readiness check that performs a real completion; staged rollout, one replica first.

**Speculative decoding that slows the system.** A draft with low acceptance for your workload adds cost at every step and TPOT worsens at high concurrency. Telemetry: acceptance rate below break-even for the draft length; TPOT regression versus baseline. Test: the same sweep with speculation on and off, compared at the operating point.

**Silent context truncation.** An engine configured with a shorter maximum length than the application assumes truncates or rejects, and the model answers without the evidence that was cut. Telemetry: engine-side input token counts below the gateway's, or a spike in `InvalidRequestError`. Test: `check_fit` with the engine's real limit, plus a contract test that sends a request one token over it.

## Tradeoffs

Every lever in this chapter trades one resource for another, and the right setting is workload-specific.

- **Context length versus concurrency.** Longer maximum contexts let single requests claim cache that many users could have shared: a 128k setting admits rare requests whose cache crowds out dozens of ordinary ones, and worst-case planning has to assume them. Set the engine's limit to your p99 plus margin, not to the model's trained maximum, and route the rare long request to a pool configured for it.
- **Batch size versus TPOT.** Larger batches raise throughput almost linearly during decode and raise per-token latency slightly; past the cache budget they raise TTFT sharply. Tune the maximum batched tokens against the SLO.
- **Quantization versus quality.** Fewer bits buy concurrency and bandwidth; the price is task-specific and only visible in a sliced evaluation.
- **Speculation versus utilization.** Speculative decoding helps most when the GPU is underutilized (low concurrency, interactive use) and least when it is already full.
- **Prefix caching versus memory.** A warm prefix cache reduces prefill work but occupies blocks that active sequences need; under heavy load it may be evicted anyway.
- **Headroom versus cost.** Every percentage point of headroom is idle hardware; every point removed moves you toward the knee. Sixty to seventy percent of measured goodput capacity is a common operating posture for interactive services, and batch pools can run hotter.
- **Hosted versus managed open-weight versus self-hosted.** Each step toward self-hosting buys control, residency, and lower cost at high utilization, and costs operational burden, capacity risk, and slower access to new models. The managed open-weight middle buys model choice without the GPUs.
- **Engine choice.** Peak performance and hardware specialization against portability, model coverage, and operational familiarity. Benchmark, pin, and plan to re-benchmark on upgrade.

## Evaluation and testing

Three kinds of tests belong to a serving layer, and they run at different times.

**Unit tests of the arithmetic**, in `test_ch34.py`, pin the formulas to known values: 128 KiB per token for the illustrative shape, exactly 1 GiB at 8k and 4 GiB at 32k, a fourfold concurrency drop between them, and the replica estimate's binding constraint flipping from decode to KV when contexts and request durations lengthen. These tests are cheap and catch the unit errors (GB versus GiB, bytes versus elements, query heads versus KV heads) that produce wildly wrong plans.

**Offline tests of the load generator** run the real HTTP and SSE code against the in-process fake and assert the shape the protocol predicts: TTFT strictly below E2E, TPOT close to the configured per-token delay, p95 TTFT rising sharply once concurrency exceeds the fake's capacity, errors counted rather than raised, and the operating-point finder selecting the last level inside capacity. `test_walkthrough_shape_has_a_knee_at_capacity` runs the walkthrough's setup and checks its shape (a real capacity limit, TTFT rising sharply above it, throughput leveling off, flat TPOT, an operating point of 8); `test_cancelling_a_sweep_stops_it` checks that an outer deadline stops a sweep instead of being swallowed. If you extend the generator (for example to record engine metrics between levels), extend the fake first so the behavior is tested before it meets a GPU.

**Benchmark runs against real hardware** are evaluations, not tests, and should be stored as artifacts with the full configuration record from the protocol section. Compare runs by goodput at the operating point under a fixed SLO. Pair every performance run with the quality suite from Chapters 24 and 25 on the same deployment; a performance gain that comes with a quality regression is a failed run. Finally, record the operating point per replica in the capacity model and re-measure whenever the model, engine, quantization, or prompt distribution changes materially; the capacity envelope is empirical and it expires.

The full test file is on disk at `book/projects/examples/ch34/test_ch34.py`. The `aie_core` tests run offline against the fake server and a mock transport; the `integration` marker remains registered for tests you add against a real engine, which are skipped unless `--run-integration` is passed.

## Before you ship

- [ ] The self-hosting decision is written down per traffic slice, with cost per useful token computed at measured average utilization (including nights and weekends), not at peak throughput.
- [ ] Two SLOs exist, p95 TTFT and p95 completion, with per-replica alerts on each, plus leading-indicator alerts on queue depth, KV-cache utilization, and preemptions.
- [ ] The KV budget was computed from the model's configuration file (attention type, KV heads, sliding windows, MoE total parameters) and matches, within a few percent, the KV capacity the engine reports at startup.
- [ ] The engine's maximum model length is set to the p99 sequence plus margin, `ServingTarget.max_context_tokens` holds that same value, and a contract test sends a request one token over it and gets a clean rejection, not truncation.
- [ ] A load test with production-sampled prompt and output lengths, warm-up discarded, swept concurrency until p95 TTFT left the floor, and its full configuration record (weights, quantization, engine version, scheduler settings, hardware) is stored with the results.
- [ ] The replica count comes from load-tested goodput at a 60 to 70 percent target utilization, plus at least one failover replica.
- [ ] The sliced release evaluation (structured output, tool calls, long context, language) passed on the exact quantization, engine version, and adapter being deployed.
- [ ] Prefix-cache hit rate is on a dashboard, and CI asserts that the system prompt's first N tokens are unchanged across cache-compatible prompt versions.
- [ ] Readiness performs a real tiny completion; rolling upgrades drain with a timeout and promote one replica first.
- [ ] A cancellation test closes streams early and sees the engine's running-sequence count return to baseline within a bounded time.
- [ ] Every span carries model, quantization, adapter, engine version, and replica; the engine requires authentication even on the internal network; cross-tenant prefix-cache sharing has been reviewed.

## Exercises

**Start here:** K3, K7, E2, P3, D2 (about 4 hours). The rest go deeper.

### Knowledge questions

**K1.** Define queue time, TTFT, TPOT, end-to-end latency, throughput, and goodput. For an interactive chat product and for a nightly batch classification job, name the one metric each cares about most and explain why the other metrics matter less for that product.

**K2.** Explain in two or three sentences, without kernel detail, why the decode phase is limited by memory bandwidth and why that makes batching during decode nearly free in throughput terms.

**K3.** State the KV-cache formula and name each variable. For a model with 48 layers, 8 KV heads, head dimension 128, and a BF16 cache, compute the cache per token and per 16k-token sequence. How does the result change with an FP8 cache?

**K4.** Contrast static batching and continuous batching. What property of LLM requests makes static batching waste capacity, and what scheduling problem does continuous batching introduce for interactive latency?

**K5.** Why is "measure task quality, not perplexity alone" the right rule for evaluating a quantized model? Name three task slices that commonly regress while aggregate scores look stable.

**K6.** State Little's Law and give one example of using it in each direction: deriving in-flight requests from traffic, and deriving a sustainable arrival rate from a fixed concurrency budget.

**K7.** An illustrative mixture-of-experts model has 48B total and 8B active parameters, 32 layers, 8 KV heads, and head dimension 128. (a) Compute its weight memory at BF16 and at FP8, and, with the chapter's 10 percent headroom and 2 GiB runtime allowance on an 80 GiB device, the KV budget left in each case. (b) How much KV cache does one 8k-token sequence need, and how many fit? (c) Would you expect its TPOT at concurrency 1 and at concurrency 32 to resemble a dense 8B model or a dense 48B model, and why? (d) If 24 of its 32 layers used sliding-window attention with a 4,096-token window, what would one 32k-token sequence cost in KV memory?

### Engineering questions

**E1.** Northwind's logistics tenant requires that incident reports never leave the corporate network, while the retail tenant has no such constraint. Design the routing so both tenants use the same application code and the same gateway. Which `aie_core` components are involved, what does the `ServingTarget` list look like, and what must appear in every trace?

**E2.** A team proposes raising the engine's maximum model length from 16k to 64k so that one rare long-document feature works. Using the illustrative 8B shape on an 80 GiB device, quantify the effect on maximum concurrency and propose two alternatives that serve the feature without paying that cost for every request.

**E3.** Your prompt registry adds a `request_id` to the first line of every system prompt for debugging. Explain the effect on prefix caching, estimate the TTFT impact for a 2,000-token system prompt relative to a 300-token user message, and propose a layout that keeps the debugging value without the cost.

**E4.** A load test shows p95 TTFT crossing the 2 second SLO at concurrency 12, with TPOT flat and KV utilization at 60 percent. Queue depth is rising. Which of the three constraints in `replicas_needed` is binding, what does that imply about the fix, and why would quantizing the KV cache not help here?

### Practical exercises

**P1.** (about 2 hours) Extend `loadtest.py` to record the engine's Prometheus-style metrics (queue depth, KV utilization, running sequences) at the end of each concurrency level, add the columns to `LevelSummary` and the report, and extend `fake_server.py` to expose a `/metrics` endpoint so the feature is tested offline.

**P2.** (about 90 min) Write `plot_sweep.py` that reads a list of `LevelSummary` objects (serialize them to JSON from `sweep`) and produces the two protocol plots: p50 and p95 TTFT against requests per second, and output tokens per second against p95 E2E. Mark the SLO lines and the operating point.

**P3.** (about 2 hours) Build an admission-control function on top of `kv_cache.tokens_that_fit` that, given free KV bytes reported by the engine and a request's input tokens plus `max_tokens`, decides between admit, queue (with a bounded wait), and reject with a retryable error. Test it with a sequence of mixed-length requests against a fixed budget and assert that it never admits more tokens than fit.

**P4.** (about half a day, needs a local GPU or a capable CPU) Run the load generator against a real local OpenAI-compatible server of your choice, once with the default configuration and once with a quantized variant of the same model. Record the full configuration from the protocol section, find both operating points, and run the same small structured-output evaluation against both. Write a one-page comparison that reports goodput at the SLO and the quality deltas by slice.

### Debugging exercises

**D1.** After an engine upgrade, p50 TTFT for the RAG endpoint doubled across all replicas while TPOT, error rate, GPU utilization, and traffic were unchanged. The prompt registry shows no changes in the last week. Traces show input token counts identical to last week's. Diagnose the most likely cause and name the single engine metric that would confirm it.

**D2.** A dashboard shows output tokens per second at an all-time high and GPU utilization at 85 percent, yet support tickets about "the assistant hangs before answering" tripled this afternoon. KV utilization is pinned at 99 percent and the engine's preemption counter is climbing. Explain what is happening, why the throughput number is misleading, and what two changes, one immediate and one structural, address it.

**D3.** Overnight, a self-hosted replica's KV utilization climbed steadily to 95 percent and stayed there with almost no client traffic; the gateway shows a handful of open connections, the engine shows forty running sequences. In the morning every request queues. Identify the fault, the layer most likely responsible, and the test that should have caught it.

## Key takeaways

- Self-hosting is a hybrid decision made per traffic slice on five factors: cost at sustained volume, data residency, latency control, model availability, and the operational burden of owning a stateful GPU service. Both sides use the same `aie_core` client and gateway; the router chooses between targets with different capabilities, and gateway fallbacks hold only interchangeable replicas.
- Define metrics precisely. TTFT is where load appears, TPOT stays mostly flat under saturation, and goodput under SLO is the only capacity number worth advertising.
- Prefill is compute-bound and sets TTFT; decode is memory-bandwidth-bound, sets TPOT, and makes batching nearly free. Traffic shape, not model size alone, determines what users feel.
- KV cache per sequence is 2 × layers × KV heads × head dim × tokens × bytes. For the illustrative 8B shape that is 1 GiB at 8k and 4 GiB at 32k, so a fourfold context increase divides concurrency by four on the same hardware. Admission control must count tokens, not requests. Read the config, not the headline parameter count: MoE weight memory follows total parameters while small-batch speed follows active ones, and sliding-window and latent attention shrink the cache below the formula.
- Continuous batching and paged KV memory are why modern engines reach high utilization; the knobs you tune are maximum batched tokens and chunked prefill, against your SLO.
- Prefix caching is controlled by prompt layout: stable content first, variable content last, nothing unique in the shared prefix.
- Quantization's main gift is memory that becomes batch size; its cost is task-specific and visible only in a sliced quality suite. Speculative decoding helps interactive, low-concurrency traffic and depends on acceptance rate.
- Little's Law turns traffic into in-flight requests and KV demand; the queueing knee says why headroom is not waste. Size replicas from the load-tested operating point, divide by a target utilization, and add failover.
- A reproducible benchmark warms up, uses realistic prompt and output distributions, sweeps concurrency in a closed loop, records TTFT, TPOT, E2E, errors, and engine metrics, and chooses the operating point where p95s meet the SLO.
- Treat every engine, quantization, or scheduler change as a model release: record versions in traces, rerun the quality suite, roll out one replica at a time, and verify that cancellation frees KV memory.

## Further reading

- *Efficiently Scaling Transformer Inference* (Pope et al., 2023): the prefill versus decode split, memory bandwidth limits, and KV-cache costs worked out carefully; the source of most of this chapter's arithmetic.
- *Efficient Memory Management for Large Language Model Serving with PagedAttention* (Kwon et al., 2023): the vLLM paper; why paged KV memory makes continuous batching practical and how fragmentation wastes capacity.
- *Orca: A Distributed Serving System for Transformer-Based Generative Models* (Yu et al., 2022): iteration-level scheduling, the idea behind continuous batching.
- *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving* (Zhong et al., 2024): goodput under an SLO as the capacity metric, and when splitting prefill and decode pays.
- *Fast Inference from Transformers via Speculative Decoding* (Leviathan, Kalman, and Matias, 2023): the acceptance rule that makes speculation lossless and the arithmetic of its speedup.
- *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (Ainslie et al., 2023): the attention variant that shrinks `kv_heads` and therefore the KV cache in most current open models.
