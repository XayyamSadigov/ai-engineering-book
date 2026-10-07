# Chapter 30 — Performance and Cost Engineering

An AI request costs and takes time in proportion to its tokens, its calls and its failures, so performance and cost are designed in, not tuned afterwards. This chapter shows where the time and money in a request go and gives you budgets, scoped caches, a cost model and per-tenant spend controls to keep both within target.

**You will be able to:**
- Break a request's latency into stages and set a latency budget whose stages add up to the TTFT and completion objectives.
- Design cache keys for six layers (prompt prefix, embedding, retrieval, tool result, response, semantic) that cannot leak across tenants or serve stale answers, and lint them in CI.
- Calculate cost per successful task, including retries, failures, human review, amortized infrastructure and the break-even point of provider prompt caching.
- Enforce per-tenant spend limits with reservations, deduplicated alerts and a degrade mode, and attribute every span's spend to a tenant and feature.
- Run independent steps in parallel under a deadline, and choose between streaming, micro-batching and batch APIs.
- Apply optimizations in an order that takes the cheap, safe, large wins first.

**Prerequisites:** Chapter 3 (the `aie_core` gateway, `Usage`, `PricingTable`), Chapter 5 (prompt layout and prefix caching practice), Chapter 7 (routing and cascades). | **Code:** `book/projects/examples/ch30/` (run: `.venv/bin/python -m pytest book/projects/examples/ch30 -q` from the book root) | **Builds:** latency budgets and tracker, scoped caches with a key linter, `CostModel` and chargeback, `SpendGuard`, and a deadline-bound fan-out helper.

## Why this matters

A Northwind Assist answer that takes eleven seconds has failed even when correct; the employee has already asked a colleague. An answer costing four cents is fine until it runs 6,000 times a day across two tenants and finance asks why logistics pays for retail's experiments. Performance and cost are requirements with numbers, and a system meets them only if it is designed to.

AI systems make this harder than ordinary services in three ways. First, cost and latency scale with content rather than request count. A conventional endpoint costs roughly the same whether the request body says "hello" or quotes a contract; a model call costs in proportion to the tokens in and out, and its latency grows with both. A product manager's request to "include a bit more context" is a capacity change. Second, the expensive component is probabilistic. A request can fail validation and need a retry, an agent can take twelve steps instead of three, and a cheap model can be wrong often enough that its savings disappear into human review. So the honest unit is not "a call" but "a task that succeeded", and the gap between those two units is where most budget surprises live. Third, the obvious optimizations have correctness hazards that ordinary caching does not. A response cache keyed on the question alone will happily serve an HR-only answer to a warehouse clerk. A semantic cache, which reuses an answer for any sufficiently similar question, will with a loose threshold answer a question nobody asked.

This chapter gives you the anatomy, the budgets, the caches with their keys, the cost model, and the controls. It also gives you an order in which to apply them, because the most common waste in this area is engineers building a semantic cache for a workload whose real problem is an unbounded output length.

## Mental model

> **Mental model:** Latency is a sum along the critical path; cost is a product per successful task.

Latency adds up. Each stage on the critical path (the chain of steps that must happen one after another before the user sees something) contributes its time, and the only ways to shorten the total are to make a stage faster, take it off the critical path by running it in parallel or after the response, or remove it. The user feels two sums: the time to first token, which covers every stage before the first streamed word, and the time to completion. A latency budget is a statement of how much of each sum each stage may use.

Cost multiplies. The cost of one successful task is roughly the token cost per call, times the calls per attempt, times one plus the retry rate, plus fixed fees per attempt, all divided by the fraction of attempts that succeed. Because the terms multiply, a modest change in any factor moves the total: halving context halves the largest term, a 20-step agent loop multiplies everything by 20, and a success rate falling from 0.9 to 0.6 raises the cost of every success by half without changing a single price.

Underneath sit two book-wide models: context is a limited resource, so trimmed tokens are the cheapest saving in both cost and latency (Chapter 5); and production AI is a systems-engineering problem, solved here with budgets, deadlines, caches and attribution.

## Core concepts

### Latency anatomy

Before you can budget latency you need to know where it goes. A Northwind Assist answer passes through these stages, and each has a different cause and a different fix:

- **Network and edge.** TLS, load balancer, the client's link: tens of milliseconds you mostly do not control.
- **Admission and queueing.** Rate-limit waits, concurrency semaphores in the gateway, and queueing inside a provider or your own serving engine. Under load this is the stage that explodes, because queue delay grows sharply as utilization approaches capacity (Chapter 34 shows the curve). The gateway records it as `queue_ms` on its span.
- **Auth and policy.** Token validation and tenant resolution; small and constant.
- **Retrieval.** Query embedding, lexical and vector search, ACL filtering, fusion. Sensitive to index size, filter selectivity and cold caches.
- **Reranking.** A cross-encoder over the candidates, typically the most expensive retrieval step per request, proportional to the number of candidates scored.
- **Prefill and time to first token (TTFT).** The model processes the whole prompt before emitting anything. Prefill cost grows with input length, so a 9,000-token prompt has a noticeably later first token than a 4,500-token one. Provider prompt caching, the provider's reuse of its prefill work for a prompt beginning it has recently processed, shortens prefill for a repeated prefix (see Caching layers below).
- **Decode and time per output token (TPOT).** Generation is sequential: one token per step per sequence. Completion time after the first token is roughly the output length times TPOT. This is why output length is the single largest latency lever for long answers.
- **Tools.** Each tool call is a network round trip to some other system, plus a further model call to read the result. In an agent loop the model, tool, model, tool rhythm puts every step on the critical path.
- **Validation and persistence.** Schema and citation checks, guardrails, the database write; fast unless a guardrail is a model call.

A useful first-order equation for a single-call answer is:

```text
completion ≈ network + queue + pre-model stages + TTFT + output_tokens × TPOT + post-model stages
TTFT_seen_by_user ≈ network + queue + pre-model stages + model TTFT
```

Worked example, all numbers illustrative: with TPOT of 15 ms, a 300-token answer decodes in 4.5 seconds and a 600-token answer in 9 seconds. Against Northwind's 8-second p95 completion target, the 600-token answer has failed before retrieval has spent a millisecond. Prefill and decode mechanics, and why decode is memory-bandwidth bound, belong to Chapter 34; what matters here is that output tokens cost time linearly and you control their number with `max_tokens` and with the answer format you ask for.

Users feel the tail: p95 and p99, the latencies that the slowest 5 percent and 1 percent of requests exceed. Tails have specific causes: a cold cache, a reranker batch that waited, a slow provider minute, a retry. Measure each stage as a distribution, never a mean.

### Latency budgets

A latency budget allocates an end-to-end objective across stages. No single stage sees the whole request: if retrieval takes six seconds at p95 against an eight-second target, no downstream model choice can compensate, and a budget surfaces that in design review rather than in an incident. It gives each stage owner a number and the code a timeout.

Build a budget in four moves. Start from the objectives: Northwind's are p95 TTFT under 2 seconds and p95 completion under 8 seconds. Mark which stages sit before the first token, because they must fit inside the TTFT target as well as the total. Hold back a reserve, typically 5 to 10 percent, for network jitter and the one retry you are willing to pay for. Then split the rest by measured need. The chapter's demo gives Northwind auth 100 ms, retrieval 600, reranking 300, generation 6,200, persistence 300 and validation 100, plus a 400 ms (5 percent) reserve: 8,000 ms in total, with the three stages before the first token using 1,000 ms of the 2,000 ms TTFT target. Chapter 28's starting plan gave retrieval 1.5 s and reranking 0.8 s; measurements are what should move numbers like these.

Deriving budgets from measurements has a subtlety. Summing per-stage p95s overstates the end-to-end p95, since stages rarely all hit their tails on the same request. That makes the sum a safe, conservative check: if the stage p95s fit, the end to end will. If they do not fit, the answer is never to hand out budgets the stages cannot meet. Make a stage faster, move it off the critical path, or renegotiate the target. `LatencyBudget.from_measurements` refuses an infeasible split for exactly this reason.

Enforce the budget twice. At run time, each stage gets `min(its budget, time remaining)` as its timeout, and the remaining time propagates downstream so a slow stage fails into retry or a degraded mode instead of hanging (Chapter 3 for the gateway's deadline, Chapter 29 for carrying it across services). Offline, audit it from traces: group spans by request, compute per-stage percentiles, count violations, and look first at the stage with the most. Measure end to end as wall clock, since parallel stages overlap.

### Token budgets

Tokens are the unit both latency and cost are made of, so budgeting them is the most direct control you have. Input tokens drive prefill latency and input cost. Output tokens drive decode latency and output cost, and output tokens are typically priced several times higher than input tokens (the illustrative prices in this book use a 4:1 ratio).

Reasoning models add hidden "thinking" tokens before the visible answer. Chapter 3 covers how they are billed and reported: as output, inside `max_tokens`. For budgeting, two consequences matter. They are decode time spent before the first visible token, so they land in TTFT, not after it. And their number varies by question, which widens the cost and latency distributions rather than shifting their means, so budget against p95, not the average. Treat the reasoning-effort setting some providers expose as a lever like `max_tokens`, set per task by evaluation (Chapter 7).

There are three levels of token budget. Per section within one prompt: system instructions, tool definitions, history, evidence, and an output reserve each get a cap, and the context builder of Chapter 5 enforces it so a long document cannot silently crowd out the instructions. Per request: `max_tokens` caps output, and the prompt asks for an answer format sized to the need (three sentences and citations, not an essay). Per task: an agent or chain making several calls gets a cumulative cap, because per-request limits do not stop a loop of forty modest calls. `TaskTokenBudget` in this chapter's code charges each step's usage and refuses a step the remaining budget cannot cover, so the loop stops with a partial result rather than an overrun.

The fleet arithmetic is what makes token discipline a business concern. Northwind Assist answers about 6,000 questions a day (Chapter 35, Case 1). One thousand extra input tokens per answer is 6 million tokens a day, 180 million a month, or about 360 USD a month at the illustrative input price, for a change that might look like "add two more chunks". At a busier 3 requests per second around the clock (259,200 requests a day), the same thousand tokens is 259 million tokens a day, about 520 USD a day at the same illustrative price. Every per-request choice is multiplied by volume; make the multiplication explicit before the change ships.

More context is also not free in quality. Additional retrieved chunks add distractors, push high-value evidence away from where the model attends best, and leave less room for the answer. Trimming context frequently improves the success rate, which lowers cost per successful task twice.

### Caching layers and their correctness keys

A cache returns a value computed for one input as the value for another input it considers equal, so its correctness lives entirely in the key. The rule: the key includes every factor that changes the correct value, and nothing that does not. Missing a factor leaks or serves stale data. Including a per-request factor (a timestamp, a request id) makes every lookup a miss and drives the hit rate, the fraction of lookups answered from the cache, to zero.

An AI request path has six cacheable layers, and they differ sharply in value and risk:

| Layer | Caches | Key must include | Main hazard |
|---|---|---|---|
| Prompt prefix | provider's prefill computation | identical token prefix (implicit) | volatile content early in the prompt destroys hits |
| Embedding | text to vector | normalized text, embedding-space fingerprint | vectors from a different model, dimension or prefix |
| Retrieval | query to ranked evidence ids | normalized query, tenant, ACL scope, index version, retriever config | cross-group leakage; stale results after reindex |
| Tool result | read-only tool call to result | tool, arguments, tenant, tool version, TTL | stale status data; side-effecting tools cached by mistake |
| Response | full request to completion | request hash, model, tenant, ACL scope, prompt version, index version | personalization and permissions not in the prompt |
| Semantic | similar question to answer | embedding, tenant, ACL scope, versions, TTL | answering a different question; poisoning |

**Prompt prefix caching** is the provider's or engine's reuse of prefill for a prefix it has recently seen. Your code does not manage it; your prompt layout earns it. Stable content goes first and byte-identical: system prompt, tool definitions, shared reference material; then tenant context; then the volatile question and fresh evidence (Chapters 5 and 34). Measure it with `Usage.cached_input_tokens`, which the gateway records on every span.

The economics depend on the provider's terms, and this is the arithmetic Chapter 5 defers to here. Write the prefix's normal input price as 1. Some providers charge a write multiplier `w` the first time a prefix enters the cache and a read multiplier `r` on every later hit within its lifetime (the TTL); providers that cache implicitly have `w = 1`, so any hit is pure saving. With illustrative values `w = 1.25` and `r = 0.1`:

- **Per prefix.** One write followed by `n` reads costs `w + n × r` instead of `1 + n`. Caching wins when `n > (w − 1) / (1 − r)`, here 0.28, so a single reuse pays: two requests cost 1.25 + 0.1 = 1.35 instead of 2. A provider tier with a longer TTL and a higher write premium, say `w = 2`, needs two reuses.
- **Per request, from traffic.** Whether the next request finds the prefix depends on how often that exact prefix arrives. If it arrives at rate λ and entries live for T after their last use, the chance a request finds a live entry is about `p = 1 − e^(−λT)` for random arrivals. The expected prefix cost per request is `(1 − p) × w + p × r`, which beats 1 when `p > (w − 1) / (w − r)`, here about 0.22, or λT above about 0.25. With a 5-minute TTL, a prefix seen once every 20 minutes is the break-even; once a minute (λT = 5, p ≈ 0.99) it costs about 0.11, an 89 percent saving on the prefix; once an hour (p ≈ 0.08) it costs about 1.16, more than not caching at all.
- **On the bill.** Saving = prefix tokens × input price × (1 − expected prefix cost per request). The prefix is usually a minority of the input, so the bill moves less than the cache ratio suggests. In the ladder below, 800 cached tokens out of 4,500 save 12 percent per success. That figure assumes every request reads the prefix with no write premium; at a 95 percent hit rate with `w = 1.25` the saving is about 6 percent smaller.

Two design rules follow. Every distinct prefix divides the traffic, so one system prompt per tenant or per experiment arm can push low-traffic prefixes below break-even (Chapter 5); a feature whose prefixes sit below it should leave explicit caching off. And measure `p` per prefix, as the share of requests whose `cached_input_tokens` covers the prefix, rather than assuming it. Do not distort the instruction hierarchy to chase hits: if a constraint must sit near the end of the prompt to be followed, keep it there.

**Embedding caching** is the safest layer and saves the most during ingestion. A vector depends on more than model name and text: the model version, output dimensions, any instruction prefix (some models expect "query:" before queries) and the text preparation all change it. `aie_core.CachedEmbeddings` already keys on a fingerprint of the embedding space (provider, model, dimensions, instruction, text-preparation version) plus the raw text; Chapter 8 explains the fingerprint. Following Chapter 8's `NamespacedStore`, this chapter's `EmbeddingCache` adds text normalization on top of the same space fingerprint, so "Refund  policy" and "refund policy" share an entry and a space change simply stops old entries matching.

**Retrieval caching** saves the most in interactive chat, where repeated questions run the same search and rerank. Cache evidence ids, not text, so a document deleted or re-permissioned since caching is checked again at hydration. The key carries the tenant and the ACL scope (a hash of the caller's sorted groups), so employees with different groups never share an entry, and the index version, bumped on every reindex, so a reindex retires all old entries without a scan.

**Tool-result caching** applies only to read-only, idempotent tools such as `get_service_status` or `lookup_employee`, with a TTL matched to how fast the underlying data changes. A status check cached for an hour during an incident is worse than no cache. A side-effecting tool such as `send_reply` must never be cached; a cache hit would silently skip the side effect.

**Response caching** stores the whole completion. Chapter 3's gateway cache keys on everything in the request (model, messages, tools, schema, sampling) and caches only at temperature zero by default. Its only view of what makes an answer specific from outside the request is the optional `metadata["cache_scope"]` string, with `require_cache_scope=True` to refuse unscoped requests; that is enough to partition by tenant, but it does not structure the ACL scope or carry the prompt and index versions, and nothing checks that each component is present. `ScopedResponseCache` adds those as named key components the linter can verify and refuses to cache truncated or filtered completions. Hit rates are low on free-form chat and high on classification, routing and extraction of repeated inputs.

**Semantic caching** reuses an answer for a question that is similar rather than identical, judged by embedding similarity above a threshold. It can cut cost and latency sharply for FAQ-style traffic, and it has the worst failure mode of any layer: a confident, well-formatted answer to a question the user did not ask. "Can I carry over vacation days" and "can I cash out vacation days" are close in embedding space and have opposite answers.

Four rules make it usable. Check eligibility before similarity: an entry from another tenant, another ACL scope, another prompt or index version, or past its TTL is never a candidate, however similar. Refuse to store personalized answers ("your laptop return ships Friday") and ungrounded answers, which would be wrong or unsupported for the next asker. Record the source documents each answer cites, so a document change invalidates exactly the answers built on it. And tune the threshold on labeled pairs of questions, measuring the false-hit rate (pairs above threshold whose correct answers differ), not on intuition. Log near misses, lookups whose best similarity fell just under the threshold, because they are the data you tune with.

Caches are also an attack surface: an answer produced by a planted, injected document is replayed by a response or semantic cache to everyone in scope until it expires (Chapter 26). Store only validated answers and keep tenant and source purges fast.

A cache-key linter turns these rules into a test: each cache class declares its key components, and the linter flags missing scope or version components as errors and per-request components as warnings. In CI, a refactor that drops the tenant from a key fails the build.

### Streaming for perceived latency

Streaming sends tokens as they are generated, over Server-Sent Events or a WebSocket. It does not reduce compute or completion time; it moves what the user perceives from completion to TTFT, which for a 5-second answer is the difference between "slow" and "fine".

Streaming changes what you budget. The stages before the first token now carry the user experience, so they get the tight budget, and decode time can be longer as long as the stream keeps moving. Send something useful before the first model token: the request id, the prompt and index versions, and the citations as soon as retrieval returns. That is what the Chapter 28 request path does, and users read the source titles while the answer starts.

Streaming collides with validation. A schema check, a citation check or an output guardrail needs the complete text, but the user has already seen it. There are three workable designs. Validate per sentence or per paragraph, holding back a small buffer, for checks that are local. Stream, then retract or annotate if the final validation fails, which is acceptable for low-risk content with a clear correction in the UI. Or do not stream the user-visible output at all for structured or high-risk results, and stream progress events instead. Choose per feature; the guardrail implementations in Chapter 27 discuss where each check can run.

Streaming also enables cancellation: propagate a closed tab or stop button to the model call, or the abandoned stream generates and bills its full output for nobody.

### Parallelization

Independent steps should not wait for each other. In a RAG request, lexical search, vector search and a read-only status check can all start the moment the query is known; the critical path then costs the slowest of them, not their sum. In a workflow, sibling branches run concurrently (Chapter 17). Many models can also emit several tool calls in one turn, and the application should execute them concurrently rather than in sequence.

Fan-out needs three disciplines. Give the group a deadline and each step `min(its own timeout, time remaining)`, cancelling stragglers so they neither hold the answer nor keep billing. Separate required steps (no vector results is fatal) from optional ones (no rerank means answering in fused order). And bound concurrency: ten branches per request at 50 requests per second is 500 concurrent calls against shared rate limits.

Parallelism amplifies tails. If each of N independent branches finishes within its p95 95 percent of the time, all of them do so only 0.95^N of the time: with three branches, that is about 86 percent, so about 14 percent of requests wait on at least one branch's tail, nearly three times the 5 percent a single branch causes. More branches mean a slower p95 for the group. Keep fan-out wide only where each branch earns its place, and set per-branch timeouts that cut the tail.

Speculative prefetch goes one step further: start a call before you know you need it. While the model plans an incident answer, prefetch `get_service_status` for the service named in the question; if the model asks for it with the same arguments, the result is already there. Speculation trades money for latency. Wasted prefetches are real calls, so count them, and restrict prefetch to read-only, idempotent tools by construction. A speculative `send_reply` is a bug, not an optimization. Hedged requests (sending a duplicate to a second replica after a delay and taking the first answer) are a related tail-cutting technique with the same cost trade-off; Chapter 29 covers them with the other reliability patterns.

### Batching

Batching trades latency for throughput and price, and three different mechanisms share the name. **Micro-batching** in your own code coalesces concurrent single-item requests, such as query embeddings from many users, into one batched call, flushing when a batch fills or the first item has waited a few milliseconds. The wait is added to every request's latency, so on interactive paths keep it to single-digit milliseconds. **Provider batch APIs** accept a file of requests, process them within hours, and charge substantially less (Chapter 3); use them for everything nobody is waiting on: nightly ticket classification, re-embedding a corpus, generating evaluation data, judge runs. **Server-side continuous batching** is what a serving engine does with concurrent requests on a GPU, and it is why self-hosted throughput depends on concurrency (Chapter 34).

The decision rule is about who is waiting. If a person is waiting, do not batch beyond a few milliseconds. If a process is waiting with a deadline of hours, batch everything and take the discount. Ingestion pipelines should batch embedding calls by default (Chapter 15); a corpus embedded one text per request costs the same tokens and many times the wall clock and request overhead.

### The cost model per successful task

Cost per model call is the number providers bill, and on its own it misleads. The unit that compares architectures honestly is cost per successful task, because it charges each design for its retries, its failures and its human review.

A request's cost decomposes as:

```text
cost_per_attempt = model_calls × (1 + retry_rate) × cost_per_call(input, cached_input, output)
                 + embeddings + reranking + tool fees + amortized infrastructure
                 + human_review_rate × cost_per_review
cost_per_successful_task = cost_per_attempt / success_rate
```

`model_calls` counts every call an attempt makes, including chain stages, agent steps, judges and guardrails. `retry_rate` counts extra calls from transient errors and schema repairs, each re-paying the whole input. Infrastructure is the amortized share of what you run yourself: GPUs at real utilization (Chapter 34), the vector database, storage and egress. Human review is often the largest term when it exists: a 2 percent review rate at an illustrative 3 USD of loaded staff time per review adds 0.06 USD per attempt, more than twelve times the optimized model cost in the ladder below. The division by the success rate is what makes failures visible. If one attempt in five fails, every success carries 1.25 attempts of cost.

The model makes otherwise counterintuitive comparisons obvious. An agent using a model priced at one tenth of the capable model, but taking 20 steps with 9,000 input tokens each and succeeding 75 percent of the time, costs more per successful task than one call to the capable model that succeeds 90 percent of the time. At the illustrative prices, with 6,000 input and 600 output tokens for the single call, and 300 output tokens and a 10 percent retry rate per agent step, the single call costs about 0.019 USD per success and the agent about 0.060 USD. The test suite asserts exactly this case.

The chapter's worked example applies the model to a Northwind Assist answer, starting from a naive first version and applying one optimization at a time so each step's saving is visible. All prices and rates are illustrative, and the success rates are what an evaluation run would measure for each variant, not assumptions about the techniques:

```text
step                                USD/attempt  USD/success   vs v0  USD/month
v0 naive                                0.02462      0.02863      0%       5154
1 trim context + cap output             0.01236      0.01404    -51%       2527
2 stable prefix cached                  0.01086      0.01234    -57%       2221
3 + semantic cache, 8% hits             0.00999      0.01126    -61%       2027
4 + route 60% to small model            0.00484      0.00560    -80%       1008
```

The monthly column is cost per success times 6,000 questions a day for 30 days. The naive version sends 9,000 input tokens (twenty chunks and the full history) and lets answers run to 600 tokens. Trimming to the Chapter 35 budget of 4,500 input and 300 output tokens, and adding a reranker so eight good chunks replace twenty mediocre ones, halves the cost even though the reranker adds a fee. The success rate also rises slightly because there are fewer distractors. Caching the 800-token stable prefix saves another 12 percent of what remained (6 points against v0). A semantic cache with an 8 percent hit rate saves less than the prefix cache did, with far more correctness risk. Routing easy questions to a small model then halves what remains, because its success rate on those questions holds up in evaluation.

Read the ladder as the saving each step adds on top of the previous ones, not as the recommended sequence. The optimization order at the end of this section puts model choice before trimming, and that is about picking the smallest model that meets the bar for each step, a one-time decision made by evaluation. Step 4 here is a different thing: per-request routing, which needs a router, an evaluation per traffic slice and a measured misroute cost (Chapter 7), so it comes once the cheaper steps are in place. Each step's percentage also depends on what came before, because each one shrinks the base the next one acts on. What the ladder does show is the cost of a wrong order: had the team started with the semantic cache, it would have taken on the riskiest layer for an 8 percent saving while the 51 percent from trimming the prompt and capping the output went unclaimed.

### Retrieval, embedding, and index costs

After the model bill has been trimmed, the retrieval side is often the next largest line, and it behaves differently because most of it is fixed rather than per token. Three terms matter.

**Embeddings.** Query embeddings are small: a 20-token question embedded 6,000 times a day is about 3.6 million tokens a month, cents at an illustrative 0.02 USD per million tokens. Corpus embeddings are paid at ingestion and again at every re-embedding. A 50-million-token corpus costs about 1 USD to embed at that price, so the API bill for a re-embed is rarely the problem. Its real cost is the overlap: two index versions held side by side while the new one passes evaluation (Chapter 28, exercise E2), the build time on the database, and the engineering time. The embedding cache (Chapter 8's space fingerprint) makes re-chunking cheap only when the embedding model is unchanged.

**Reranking.** A cross-encoder scores every candidate, so its cost is candidates times tokens per candidate: 50 candidates of 300 tokens is 15,000 tokens of reranker input per request, paid as a per-search fee to a hosted reranker or as GPU time for a self-hosted one. That makes the candidate count a cost lever with a quality price. Measure recall at each depth (Chapter 14) and cut the candidate set to the smallest depth that keeps the gold evidence, and put the retrieval cache in front of the reranker, where a hit saves the most.

**The index.** A vector index that must answer in milliseconds lives in memory, so its cost is the instance or managed tier sized for vectors plus graph (Chapter 9 gives the sizing arithmetic). This is a fixed monthly cost, and the cost model carries it as `infra_usd`, amortized per attempt: monthly cost divided by monthly attempts. At an illustrative 600 USD a month and 180,000 answers, that is about 0.0033 USD per answer. Set against the optimized ladder above, where an attempt costs about 0.0048 USD, the index alone adds about 70 percent on top of everything the ladder counts. Two conclusions follow. Amortized infrastructure grows in relative weight as model optimizations succeed, so re-run the cost model after each step instead of assuming the model still dominates. And fixed costs shrink per task only with volume, so a low-traffic tenant on a dedicated index can cost more per answer than a busy one on a shared index with filters (Chapter 15).

### Routing for cost

Model routing sends each request to the cheapest model that handles it well enough. Chapter 7 owns routers and cascades; here is what the cost model adds. A traffic mix is priced as total spend over total successes, not as an average of each route's cost per success, so a cheap route with a low success rate drags the whole mix. `CostModel.mix` computes it that way. A cascade, which tries the small model first and escalates on low confidence, pays twice for every escalated request, so its savings depend on the escalation rate as much as on prices. Misroutes have a cost that is not in the token bill: a hard question answered badly by the small model is a silent error that a user acts on. Chapter 7's main recommendation applies here: tune routing and cascade thresholds on end-to-end utility that includes the cost of silent errors, not on router accuracy. Keep the division of labor straight: same-model retries live in the gateway, and capability-changing fallbacks live in the router.

### Cost monitoring, budgets, alerts and chargeback

You cannot manage spend you cannot attribute. The gateway records tokens and `cost_usd` on every `llm.complete` span, but that span does not know which tenant, request or feature it served. Attribution has to be stamped on at the boundary. In this chapter's code, `bind(tenant=..., request_id=..., feature=...)` sets context variables at the start of a request, and `AttributingTracer` copies them onto every span exported inside that context, including the gateway's. Context variables follow asyncio tasks automatically; work handed to a thread pool must carry them explicitly. Track the unattributed share of spend as a metric. It should be near zero, and when it is not, a background job or a new feature is spending without a tenant.

With attribution in place, four controls follow.

**Dashboards and anomaly alerts.** Hourly spend per tenant, feature, route and model against a trailing baseline, with the usual regression causes visible: route-mix shift, rising escalation rate, falling prompt-cache ratio, growing agent steps per task, retry storms.

**Spend limits.** A per-tenant daily limit with three enforcement modes. `observe` only alerts, which is right while you learn real usage. `degrade` switches to a cheaper model with a smaller output cap past a soft limit and blocks at the hard limit, which keeps the assistant useful for the rest of the day. `enforce` blocks at the hard limit, for tenants with contractual caps.

Enforcement must survive concurrency: if fifty requests each check "is there budget left" before any of them commits, all fifty pass. The guard therefore reserves each request's worst-case cost (all input uncached, the full `max_tokens` used) before the call and replaces the reservation with the actual cost afterwards. A model missing from the price table is refused rather than priced at zero, which would let it slip past every limit. Releasing a reservation when a call fails assumes the failure was not billed; a timeout after generation started may well have been, so reconcile against the provider's usage report periodically.

**Alerts.** Fire at fixed fractions of the limit (50, 80 and 100 percent), once per threshold per tenant per day, and outside any lock so a slow pager does not stall requests. An alert that fires on every request after the limit is crossed gets muted, and then it is useless.

**Chargeback.** Aggregate spend per tenant from traces: model spend from `llm.complete` spans, other spend (embeddings, reranking, tool fees) from spans that carry `cost_usd`, and successes from task spans, so each tenant sees its cost per successful task. Cache hits need one rule. A hit was not billed, but the price of the call it replaced is the cache's value. The `aie_core` gateway records a hit with `cost_usd = 0` and the replaced price as `avoided_cost_usd`, and `chargeback` adds that to avoided spend, never to spend, and reports it next to actual spend. Shared fixed costs such as the platform, idle failover capacity and the vector database are allocated by each tenant's share of direct spend, or by another rule finance agrees to; write the rule down, so that when a tenant disputes its share, the allocation can be checked against it.

### Back-of-envelope estimates

Chapter 35 owns the capacity formulas and the sizing module (`book/projects/examples/ch35/back_of_envelope.py`): peak requests per second, tokens per second, Little's Law for in-flight requests, daily tokens, and cost per day. Chapter 34 owns replica and KV-cache sizing. This chapter adds three estimates you will make weekly.

The cost of a change: tokens added per request times requests per day times price. Six thousand answers a day, plus 1,000 input tokens each, at an illustrative 2 USD per million, is 12 USD a day. `CostModel.what_if` does this for any field.

The latency of an output cap: output tokens saved times TPOT. Cutting 300 tokens at 15 ms each saves 4.5 seconds at p50, and somewhat more at p95, because TPOT degrades under load.

The value of a cache: hit rate times cost per miss times volume, minus the cost of the cache and of its false hits. A semantic cache with an 8 percent hit rate on 6,000 daily answers saves about 480 model calls a day; if 1 percent of its hits are wrong and each wrong answer costs a support ticket, price that in before calling the cache a win.

One serving sanity check is worth memorizing: 20 requests per second at 4 seconds each is 80 in flight by Little's Law, and at 1.5 GiB of KV cache each that is 120 GiB of accelerator memory before weights. Chapter 34 works through this calculation and its load-test follow-up; capacity planning starts from traffic traces, not single-request benchmarks.

### An optimization order

Applied in the wrong order, optimizations waste weeks and add risk for little gain. This order puts the cheap, safe, large wins first:

1. **Measure first.** Per-stage latency distributions, tokens per call by prompt section, calls per task, success rate, and cost per successful task, broken down by tenant and feature. Without these, every later step is a guess. Evaluate before optimizing, so you can see when an optimization costs quality.
2. **Right-size the model.** Choose the smallest model that meets the quality bar per step, by evaluation (Chapter 7). This is usually the largest single factor in the price. Per-request routing between models is a later refinement, once a router can be evaluated against the trimmed workload.
3. **Trim context and output.** Enforce token budgets per section, rerank to fewer and better chunks, compact history, cap `max_tokens`, and ask for concise formats. This cuts cost and latency together and often improves quality.
4. **Cache.** In order of safety: prompt-prefix layout, embeddings, retrieval, read-only tool results, exact responses where outputs are deterministic, and semantic caching last, only where traffic is repetitive and the false-hit rate is measured.
5. **Parallelize, stream and batch.** Take independent steps off the critical path, stream user-visible output, micro-batch interactive embeddings, and move offline work to batch APIs.
6. **Then infrastructure.** Self-hosting, quantization, speculative decoding, dedicated capacity (Chapter 34). These are powerful and expensive in engineering time, and they pay only after the steps above have shrunk the workload to its real size.

## How it works

Follow one Northwind Assist answer through the pieces. The API boundary resolves tenant and groups and opens `bind(tenant="retail", request_id=..., feature="chat")`, so every later span carries them. A `Deadline` is created from the budget: 8 seconds in total, 1 second of the 2-second TTFT target for the stages before the first token.

The `BudgetedClient` around the gateway estimates the worst-case cost and asks the `SpendGuard` for a reservation. Retail is under its soft limit, so the request proceeds; past it, in `degrade` mode, the request would be rewritten to the small model with a 512-token cap.

Retrieval runs as a fan-out under `deadline.timeout_for("retrieve")`. The query vector comes from the `EmbeddingCache`. The `RetrievalCache` misses, so lexical and vector search run concurrently, with the reranker as an optional step. The semantic cache, consulted with the same scope and versions, misses and logs a near miss. The context builder (Chapter 5) assembles the prompt with the stable prefix first; the gateway streams, and the provider reports 800 cached input tokens. Citations reached the client when retrieval returned; the first token arrives at 1.3 seconds. When the stream ends, the client commits the actual cost and the guard checks alert thresholds. The validated answer is stored in the semantic cache with its source ids. Offline, the `LatencyTracker` and `chargeback` read the same trace file for per-stage violations and per-tenant cost.

## Architecture

The first diagram shows the request path with its budget gates and caches. The caches are inside the trust boundary of one tenant's scope: every key below the gateway carries tenant and ACL scope.

```mermaid
flowchart TD
    U[Employee] --> API["API: auth, bind tenant and request_id"]
    API --> DL["Deadline from LatencyBudget"]
    DL --> SG{"SpendGuard reserve"}
    SG -- block --> E429["Budget exceeded response"]
    SG -- degrade --> DG["Rewrite to small model, cap max_tokens"]
    SG -- allow --> SC
    DG --> SC
    subgraph Scope["Keys scoped by tenant, ACL scope, versions"]
        SC{"Semantic cache eligible and similar?"}
        RC["Retrieval cache"]
        EC["Embedding cache, space fingerprint"]
        RESP["Scoped response cache"]
    end
    SC -- hit --> OUT[Stream cached answer]
    SC -- miss --> FO["Fan-out under deadline"]
    FO --> EC
    FO --> RC
    RC -- miss --> SEARCH["Lexical and vector in parallel, optional rerank"]
    SEARCH --> CB["Context builder, token budget, stable prefix first"]
    RC -- hit --> CB
    CB --> RESP
    RESP -- miss --> GW["ModelGateway: prefix cache, retries, cost on span"]
    GW --> STREAM[Stream tokens to client]
    STREAM --> COMMIT["SpendGuard commit actual cost, alerts"]
    COMMIT --> TR[("Trace sink: spans with tenant")]
    TR --> LT["LatencyTracker: p95 per stage, violations"]
    TR --> CHG["Chargeback per tenant"]
```

The second diagram shows the reservation protocol that keeps concurrent requests from jointly overshooting a limit, including the degrade path and the alert that fires after commit.

```mermaid
sequenceDiagram
    participant App
    participant BC as BudgetedClient
    participant G as SpendGuard
    participant GW as ModelGateway
    participant Pager
    App->>BC: complete(req, tenant=retail)
    BC->>BC: estimate worst case: input uncached plus max_tokens
    BC->>G: reserve(retail, estimate)
    alt projected over hard limit and mode is not observe
        G-->>BC: block
        G->>Pager: blocked alert, once per day
        BC-->>App: BudgetExceededError, not retryable
    else projected over soft limit in degrade mode
        G-->>BC: degrade with reservation
        BC->>G: release, then reserve at the cheaper estimate
        BC->>GW: complete(small model, capped max_tokens)
    else within budget
        G-->>BC: allow with reservation
        BC->>GW: complete(req)
    end
    GW-->>BC: completion with usage and cost_usd
    BC->>G: commit(reservation, actual cost)
    G->>Pager: threshold alert if 50, 80 or 100 percent first crossed today
    BC-->>App: completion
```

The third diagram shows the semantic cache's decision, which is the layer where the order of checks is the correctness property.

```mermaid
flowchart LR
    Q[Question] --> N[Normalize and embed]
    N --> F{"Same tenant, ACL scope, prompt and index version, unexpired?"}
    F -- no --> I["Not a candidate, count as ineligible"]
    F -- yes --> S{"Best similarity at or above threshold?"}
    S -- yes --> H[Serve cached answer, record hit]
    S -- no --> NM{"Within near-miss margin?"}
    NM -- yes --> L[Log pair for threshold review]
    NM -- no --> M[Miss]
    L --> M
    M --> GEN[Generate answer]
    GEN --> W{"Grounded and not personalized?"}
    W -- yes --> ST[Store with source ids]
    W -- no --> R[Refuse write]
    DOC[Document changed] --> INV[Invalidate entries citing it]
```

## Implementation

The code is in `book/projects/examples/ch30/`:

```text
book/projects/examples/ch30/
  README.md          run instructions and file guide
  attribution.py     AttributingTracer, bind(): request-scoped span attributes
  latency.py         LatencyBudget, StageBudget, Deadline, LatencyTracker, percentile
  caching.py         Scope, TTLCache, EmbeddingCache, RetrievalCache, ScopedResponseCache,
                     SemanticCache, lint_cache_key, lint_cache_classes
  cost.py            CostScenario, CostModel, chargeback, load_jsonl
  budgets.py         SpendPolicy, SpendGuard, BudgetedClient, TaskTokenBudget
  parallel.py        fan_out, Step, Prefetcher, MicroBatcher
  demo.py            prints the optimization ladder from this chapter
  test_ch30.py       offline tests
  test_hardening.py  edge cases: unpriced models, failed streams, cancellation, percentiles
```

Run it from the book root:

```bash
uv pip install --python .venv/bin/python -e book/projects/aie_core   # or: pip install -e book/projects/aie_core
.venv/bin/python -m pytest book/projects/examples/ch30 -q
cd book/projects/examples/ch30 && ../../../../.venv/bin/python demo.py
```

Nothing in the module reads environment variables directly. Prices come in as an `aie_core.PricingTable`, which a service loads from configuration with `PricingTable.from_json`; spend policies are plain dataclasses loaded from tenant configuration. The provider settings (`LLM_PROVIDER`, `LLM_MODEL`, `EMBEDDING_MODEL`) are those of `aie_core` and apply when you wrap a real gateway. The files are complete on disk; the listings below show the parts that carry the ideas.

### Attribution

The whole file is short, and everything else depends on it:

```python
# path: book/projects/examples/ch30/attribution.py
"""Attach request-scoped attributes (tenant, request_id, feature) to every span.

The gateway's `llm.complete` span knows the model, tokens and cost, and current `aie_core`
links it to its parent through trace and span ids, but it does not know which tenant, request
or feature it served. Latency tracking and chargeback both need
that join. `AttributingTracer` wraps any `Tracer` and stamps the attributes bound with
`bind()` onto each span at export time, using a context variable so concurrent asyncio
tasks keep their own values.
"""
from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Any, Iterator

from aie_core.observability import Span, Tracer

_BOUND: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar("ch30_bound_attrs", default={})


@contextmanager
def bind(**attributes: Any) -> Iterator[dict[str, Any]]:
    """Bind attributes for the current context; nested binds merge and restore on exit."""
    merged = {**_BOUND.get(), **attributes}
    token = _BOUND.set(merged)
    try:
        yield merged
    finally:
        _BOUND.reset(token)


def bound_attributes() -> dict[str, Any]:
    return dict(_BOUND.get())


class AttributingTracer(Tracer):
    """Delegating tracer: span attributes win over bound ones, so a span can override."""

    def __init__(self, inner: Tracer) -> None:
        self.inner = inner

    def export(self, span: Span) -> None:
        for key, value in _BOUND.get().items():
            span.attributes.setdefault(key, value)
        self.inner.export(span)


__all__ = ["AttributingTracer", "bind", "bound_attributes"]
```

### Latency budgets and the tracker

The budget validates itself on construction, so an infeasible budget cannot exist in the program. `from_measurements` derives budgets from measured p95s and refuses when they do not fit; `Deadline` hands out timeouts.

```python
# path: book/projects/examples/ch30/latency.py  (excerpt; full file on disk)
@dataclass(frozen=True)
class StageBudget:
    name: str
    budget_ms: float
    before_first_token: bool = False  # on the TTFT path (auth, retrieval, model prefill...)


@dataclass(frozen=True)
class LatencyBudget:
    total_ms: float
    stages: tuple[StageBudget, ...]
    ttft_ms: float | None = None
    reserve_ms: float = 0.0  # slack held back for network jitter and retries

    def __post_init__(self) -> None:
        names = [s.name for s in self.stages]
        if len(names) != len(set(names)):
            raise BudgetError(f"duplicate stage names: {names}")
        allocated = sum(s.budget_ms for s in self.stages) + self.reserve_ms
        if allocated > self.total_ms + 1e-6:
            raise BudgetError(f"stages plus reserve need {allocated:.0f} ms, total is {self.total_ms:.0f} ms")
        if self.ttft_ms is not None:
            ttft_path = sum(s.budget_ms for s in self.stages if s.before_first_token)
            if ttft_path > self.ttft_ms + 1e-6:
                raise BudgetError(f"first-token path needs {ttft_path:.0f} ms, TTFT target is {self.ttft_ms:.0f} ms")
```

```python
# path: book/projects/examples/ch30/latency.py  (excerpt; full file on disk)
    @classmethod
    def from_measurements(
        cls,
        total_ms: float,
        measured_p95_ms: Mapping[str, float],
        *,
        ttft_ms: float | None = None,
        before_first_token: Iterable[str] = (),
        reserve_fraction: float = 0.1,
    ) -> "LatencyBudget":
        """Derive budgets from measured stage p95s.

        If the measured p95s already fit, each stage gets its p95 plus a proportional share of the
        spare time. If they do not fit, raise: the fix is to make a stage faster or change the
        target, not to hand out budgets the stages cannot meet. Summing p95s overstates the
        end-to-end p95 (stages rarely all hit their tail on one request), so this errs on the safe side.
        """
        usable = total_ms * (1 - reserve_fraction)
        need = sum(measured_p95_ms.values())
        if need > usable:
            worst = max(measured_p95_ms, key=lambda k: measured_p95_ms[k])
            raise BudgetError(
                f"measured stage p95s sum to {need:.0f} ms but only {usable:.0f} ms are usable; "
                f"largest stage is {worst} at {measured_p95_ms[worst]:.0f} ms"
            )
        spare = usable - need
        ttft_set = set(before_first_token)
        stages = tuple(
            StageBudget(name, round(p95 + spare * p95 / need, 1) if need else 0.0, name in ttft_set)
            for name, p95 in measured_p95_ms.items()
        )
        reserve = total_ms - sum(s.budget_ms for s in stages)
        return cls(total_ms=total_ms, stages=stages, ttft_ms=ttft_ms, reserve_ms=reserve)


class Deadline:
    """An absolute deadline derived from the end-to-end budget; hands out per-stage timeouts."""

    def __init__(self, budget: LatencyBudget, *, clock: Callable[[], float] = time.monotonic, start: float | None = None) -> None:
        self.budget = budget
        self._clock = clock
        self.start = clock() if start is None else start
        self.expires_at = self.start + budget.total_ms / 1000

    def remaining_ms(self) -> float:
        return max(0.0, (self.expires_at - self._clock()) * 1000)

    def expired(self) -> bool:
        return self.remaining_ms() <= 0

    def timeout_for(self, stage: str) -> float:
        """Seconds to pass as a timeout to the stage: its budget, capped by what is left overall."""
        return min(self.budget[stage].budget_ms, self.remaining_ms()) / 1000
```

The tracker groups spans by request id, sums repeated runs of a stage (retries, several tool calls), and computes end to end as wall clock.

```python
# path: book/projects/examples/ch30/latency.py  (excerpt; full file on disk)
    def report(self) -> LatencyReport:
        per_stage: dict[str, list[float]] = defaultdict(list)
        e2e: list[float] = []
        violations: list[Violation] = []
        budgets = {s.name: s.budget_ms for s in self.budget.stages}
        for request_id, timings in self._timings.items():
            # A stage may run several times (retries, multiple tool calls): count its wall time once,
            # merging overlapping runs, so two parallel 500 ms calls are 500 ms, not 1000.
            stage_ms: dict[str, float] = {}
            by_stage: dict[str, list[StageTiming]] = defaultdict(list)
            for t in timings:
                by_stage[t.stage].append(t)
            for stage, runs in by_stage.items():
                total, cur_start, cur_end = 0.0, None, None
                for t in sorted(runs, key=lambda r: r.start_s):
                    if cur_end is None or t.start_s > cur_end:
                        if cur_end is not None:
                            total += cur_end - cur_start
                        cur_start, cur_end = t.start_s, t.end_s
                    else:
                        cur_end = max(cur_end, t.end_s)
                total += (cur_end - cur_start) if cur_end is not None else 0.0
                stage_ms[stage] = total * 1000
            for stage, ms in stage_ms.items():
                per_stage[stage].append(ms)
                if stage in budgets and ms > budgets[stage]:
                    violations.append(Violation(request_id, stage, ms, budgets[stage]))
            # End to end is wall clock, not the sum: parallel stages overlap.
            begin = min(t.start_s for t in timings)
            finish = max(t.end_s for t in timings)
            total = (finish - begin) * 1000
            e2e.append(total)
            if total > self.budget.total_ms:
                violations.append(Violation(request_id, "end_to_end", total, self.budget.total_ms))
            if self.budget.ttft_ms is not None and request_id in self._first_token:
                ttft = (self._first_token[request_id] - begin) * 1000
                if ttft > self.budget.ttft_ms:
                    violations.append(Violation(request_id, "ttft", ttft, self.budget.ttft_ms))
        p95 = {k: percentile(v, 95) for k, v in per_stage.items()}
        return LatencyReport(
            requests=len(self._timings),
            stage_p50_ms={k: percentile(v, 50) for k, v in per_stage.items()},
            stage_p95_ms=p95,
            stage_p99_ms={k: percentile(v, 99) for k, v in per_stage.items()},
            end_to_end_p95_ms=percentile(e2e, 95),
            sum_of_stage_p95_ms=sum(p95.values()),
            violations=violations,
        )
```

### Cache layers

The scope object and the embedding-space fingerprint are where most key mistakes are prevented:

```python
# path: book/projects/examples/ch30/caching.py  (excerpt; full file on disk)
@dataclass(frozen=True)
class Scope:
    """Who is asking. The ACL scope is a hash of the sorted groups so keys do not leak group names."""

    tenant: str
    groups: frozenset[str]

    @classmethod
    def of(cls, tenant: str, groups: Iterable[str]) -> "Scope":
        return cls(tenant, frozenset(groups))

    @property
    def acl_scope(self) -> str:
        return hashlib.sha256("\x00".join(sorted(self.groups)).encode("utf-8")).hexdigest()[:16]
```

```python
# path: book/projects/examples/ch30/caching.py  (excerpt; full file on disk)
    @property
    def space(self) -> str:
        """Fingerprint of everything besides the text that determines the vector."""
        material = {
            "model": self.model,
            "model_version": self.model_version,
            "dimensions": self.dimensions,
            "instruction_prefix": self.instruction_prefix,
            "text_prep": NORMALIZATION_VERSION,
        }
        return hashlib.sha256(json.dumps(material, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def key(self, text: str) -> str:
        return make_key("emb", {"space": self.space, "text": normalize_text(text)})
```

The retrieval key puts the tenant in the prefix so a tenant can be purged with one prefix delete:

```python
# path: book/projects/examples/ch30/caching.py  (excerpt; full file on disk)
    def key(self, query: str, scope: Scope, *, index_version: str, retriever_config: str) -> str:
        return make_key(
            f"ret:{scope.tenant}",  # tenant in the prefix enables per-tenant purge
            {
                "query": normalize_text(query),
                "tenant": scope.tenant,
                "acl_scope": scope.acl_scope,
                "index_version": index_version,
                "retriever_config": retriever_config,
            },
        )
```

The scoped response cache layers tenant, ACL scope and versions over `aie_core.cache_key`, and never stores truncated output:

```python
# path: book/projects/examples/ch30/caching.py  (excerpt; full file on disk)
    def complete(self, req: CompletionRequest, scope: Scope, *, prompt_version: str, index_version: str = "none") -> tuple[Completion, bool]:
        if req.temperature > self.max_temperature or req.metadata.get("cache") is False:
            return self.client.complete(req), False
        k = self.key(req, scope, prompt_version=prompt_version, index_version=index_version)
        hit = self.store.get(k)
        if hit is not None:
            self.hits += 1
            return hit, True
        self.misses += 1
        completion = self.client.complete(req)
        if completion.finish_reason in ("stop", "end_turn", "tool_calls", "tool_use"):
            self.store.set(k, completion, self.ttl_s)  # never cache truncated or filtered output
        return completion, False
```

The semantic cache checks eligibility before similarity, refuses personalized and ungrounded writes, and supports both version and dependency invalidation:

```python
# path: book/projects/examples/ch30/caching.py  (excerpt; full file on disk)
    def _eligible(self, e: SemanticEntry, scope: Scope, versions: Mapping[str, str], now: float) -> bool:
        return e.tenant == scope.tenant and e.acl_scope == scope.acl_scope and e.versions == dict(versions) and e.expires_at > now

    def lookup(self, question: str, scope: Scope, versions: Mapping[str, str]) -> SemanticHit | None:
        vec = self.embedder.embed_query(normalize_text(question))
        now = self._clock()
        best: tuple[float, SemanticEntry] | None = None
        best_ineligible = 0.0
        with self._lock:
            for e in self._entries:
                sim = cosine_similarity(vec, e.vector)
                if not self._eligible(e, scope, versions, now):
                    best_ineligible = max(best_ineligible, sim)
                    continue
                if best is None or sim > best[0]:
                    best = (sim, e)
        if best is not None and best[0] >= self.threshold:
            self.stats.hits += 1
            return SemanticHit(best[1].answer, best[0], best[1].entry_id, best[1].question)
        self.stats.misses += 1
        if best_ineligible >= self.threshold:
            self.stats.ineligible += 1
        if best is not None and best[0] >= self.threshold - self.near_miss_margin:
            self.stats.near_misses += 1
            self.stats.near_miss_log.append((question, best[1].question, round(best[0], 4)))
        return None

    def store(
        self,
        question: str,
        answer: str,
        scope: Scope,
        versions: Mapping[str, str],
        *,
        source_ids: Iterable[str] = (),
        personalized: bool = False,
        grounded: bool = True,
    ) -> str | None:
        """Store an answer. Refuses personalized or ungrounded answers: they are wrong for the next asker."""
        if personalized or not grounded:
            self.stats.refused_writes += 1
            return None
        entry = SemanticEntry(
            entry_id=uuid.uuid4().hex[:12],
            question=question,
            vector=self.embedder.embed_query(normalize_text(question)),
            answer=answer,
            tenant=scope.tenant,
            acl_scope=scope.acl_scope,
            versions=dict(versions),
            source_ids=frozenset(source_ids),
            expires_at=self._clock() + self.ttl_s,
        )
        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self.max_entries:
                self._entries.pop(0)
        return entry.entry_id
```

```python
# path: book/projects/examples/ch30/caching.py  (excerpt; full file on disk)
    def invalidate_sources(self, source_ids: Iterable[str]) -> int:
        """Dependency invalidation: drop every answer that cited a changed or deleted document."""
        changed = set(source_ids)
        with self._lock:
            before = len(self._entries)
            self._entries = [e for e in self._entries if not (e.source_ids & changed)]
            return before - len(self._entries)
```

The linter encodes the key rules as data:

```python
# path: book/projects/examples/ch30/caching.py  (excerpt; full file on disk)
REQUIRED_COMPONENTS: dict[str, frozenset[str]] = {
    "embedding": frozenset({"text", "space"}),  # space = model, version, dimensions, prefix, text prep (Ch 8)
    "retrieval": frozenset({"query", "tenant", "acl_scope", "index_version"}),
    "response": frozenset({"request", "model", "tenant", "acl_scope", "prompt_version"}),
    "semantic": frozenset({"tenant", "acl_scope", "ttl", "version"}),
    "tool": frozenset({"tool", "arguments", "tenant", "tool_version"}),
}
SCOPE_COMPONENTS = frozenset({"tenant", "acl_scope"})
PER_REQUEST_COMPONENTS = frozenset({"request_id", "timestamp", "trace_id", "session_nonce", "now"})


@dataclass(frozen=True)
class LintFinding:
    layer: str
    severity: str  # "error" leaks or serves stale data; "warning" kills the hit rate
    component: str
    message: str


def lint_cache_key(layer: str, components: Iterable[str]) -> list[LintFinding]:
    """Check a cache key definition against the components its layer needs."""
    if layer not in REQUIRED_COMPONENTS:
        raise ValueError(f"unknown cache layer {layer!r}; known: {sorted(REQUIRED_COMPONENTS)}")
    have = set(components)
    findings: list[LintFinding] = []
    for missing in sorted(REQUIRED_COMPONENTS[layer] - have):
        if missing in SCOPE_COMPONENTS:
            msg = f"{layer} key lacks {missing}: two callers with different access can share an entry"
        elif missing == "space":
            msg = f"{layer} key lacks the embedding-space fingerprint: a model, dimension, prefix or text-prep change serves stale vectors"
        elif missing in ("index_version", "prompt_version", "version", "tool_version", "model"):
            msg = f"{layer} key lacks {missing}: entries survive a change that alters the correct value"
        elif missing == "ttl":
            msg = f"{layer} entries have no TTL: answers over changing data never expire"
        else:
            msg = f"{layer} key lacks {missing}: different inputs can collide"
        findings.append(LintFinding(layer, "error", missing, msg))
    for noisy in sorted(have & PER_REQUEST_COMPONENTS):
        findings.append(LintFinding(layer, "warning", noisy, f"{layer} key includes {noisy}: every request is a miss"))
    return findings
```

### The cost model and chargeback

```python
# path: book/projects/examples/ch30/cost.py  (excerpt; full file on disk)
class CostModel:
    def __init__(self, pricing: PricingTable) -> None:
        self.pricing = pricing

    def model_call_usd(self, s: CostScenario) -> float:
        usage = Usage(input_tokens=s.input_tokens, output_tokens=s.output_tokens, cached_input_tokens=s.cached_input_tokens)
        return self.pricing.cost_usd(s.model, usage)

    def breakdown(self, s: CostScenario) -> CostBreakdown:
        calls = s.model_calls * (1 + s.retry_rate)
        components = {
            "llm": self.model_call_usd(s) * calls,
            "embedding": s.embedding_tokens * s.embedding_price_per_1m / 1_000_000,
            "rerank": s.rerank_calls * s.rerank_price_per_call,
            "tools": s.tool_fees_usd,
            "infra": s.infra_usd,
            "human_review": s.human_review_rate * s.human_review_usd,
        }
        per_attempt = sum(components.values())
        # Failed attempts are paid for too. If 1 in 5 attempts fails, each success carries 1.25 attempts.
        return CostBreakdown(s.name, components, per_attempt, per_attempt / s.success_rate)

    def compare(self, baseline: CostScenario, *alternatives: CostScenario) -> list[Comparison]:
        base = self.breakdown(baseline).per_successful_task
        out = []
        for s in (baseline, *alternatives):
            cost = self.breakdown(s).per_successful_task
            out.append(Comparison(s.name, cost, (cost - base) / base * 100 if base else 0.0))
        return sorted(out, key=lambda c: c.per_successful_task)

    def what_if(self, s: CostScenario, **changes: Any) -> CostBreakdown:
        """Re-price a scenario with some fields changed, e.g. what_if(s, input_tokens=s.input_tokens + 1000)."""
        return self.breakdown(s.model_copy(update={"name": f"{s.name}*", **changes}))

    def mix(self, name: str, parts: list[tuple[float, CostScenario]]) -> CostBreakdown:
        """Price a traffic mix, e.g. a router sending 70% of tasks to a small model.

        Cost per successful task of a mix is total spend over total successes, not the average of
        each route's cost per success: a cheap route with a low success rate drags the whole mix.
        """
        weight = sum(w for w, _ in parts)
        if weight <= 0:
            raise ValueError("weights must sum to a positive number")
        components: dict[str, float] = defaultdict(float)
        successes = 0.0
        for w, s in parts:
            b = self.breakdown(s)
            for k, v in b.components.items():
                components[k] += v * w / weight
            successes += s.success_rate * w / weight
        per_attempt = sum(components.values())
        return CostBreakdown(name, dict(components), per_attempt, per_attempt / successes)

    def monthly(self, s: CostScenario, successful_tasks_per_day: float, days: int = 30) -> MonthlyProjection:
        b = self.breakdown(s)
        successes = successful_tasks_per_day * days
        attempts = successes / s.success_rate
        return MonthlyProjection(
            scenario=s.name,
            successful_tasks=successes,
            attempts=attempts,
            total_usd=attempts * b.per_attempt,
            by_component={k: v * attempts for k, v in b.components.items()},
        )
```

Chargeback reads the dicts `JsonlTracer` writes. Note the cache-hit branch.

```python
# path: book/projects/examples/ch30/cost.py  (excerpt; full file on disk)
def chargeback(records: Iterable[Mapping[str, Any]], pricing: PricingTable | None = None, *, llm_span: str = "llm.complete", task_span: str = "task") -> ChargebackReport:
    # ...
    tenants: dict[str, TenantUsage] = {}

    def usage_for(tenant: str) -> TenantUsage:
        if tenant not in tenants:
            tenants[tenant] = TenantUsage(tenant)
        return tenants[tenant]

    for rec in records:
        attrs = rec.get("attributes", {})
        t = usage_for(str(attrs.get("tenant") or UNATTRIBUTED))
        feature = str(attrs.get("feature", "unknown"))
        name = rec.get("name")
        if name == llm_span:
            cost = float(attrs.get("cost_usd") or 0.0)
            usage = Usage(
                input_tokens=int(attrs.get("input_tokens", 0)),
                output_tokens=int(attrs.get("output_tokens", 0)),
                cached_input_tokens=int(attrs.get("cached_input_tokens", 0)),
            )
            if cost == 0.0 and pricing is not None and attrs.get("model"):
                cost = pricing.cost_usd(str(attrs["model"]), usage)  # older spans without cost_usd
            if attrs.get("cache_hit"):
                t.cache_hits += 1
                avoided = attrs.get("avoided_cost_usd")
                t.avoided_usd += float(avoided) if avoided else cost  # legacy spans: price in cost_usd
                continue
            if rec.get("status") == "error" and not usage.input_tokens:
                continue  # failed before the provider billed anything
            t.model_calls += 1
            t.spend_usd += cost
            t.input_tokens += usage.input_tokens
            t.cached_input_tokens += usage.cached_input_tokens
            t.output_tokens += usage.output_tokens
            t.by_feature[feature] += cost
        elif name == task_span:
            t.tasks += 1
            if attrs.get("success") is True:
                t.successes += 1
        elif attrs.get("cost_usd"):
            cost = float(attrs["cost_usd"])
            t.other_usd += cost
            t.by_feature[feature] += cost
    return ChargebackReport(tenants)
```

### Spend guard

Reservation and commit, with alerts fired outside the lock:

```python
# path: book/projects/examples/ch30/budgets.py  (excerpt; full file on disk)
    def reserve(self, tenant: str, estimate_usd: float) -> Decision:
        """Decide and, unless blocked, hold `estimate_usd` against today's limit."""
        policy = self.policy_for(tenant)
        day = utc_day(self._clock())
        alerts: list[Alert] = []
        with self._lock:
            ledger = self._ledger(tenant, day)
            projected = ledger.exposure + estimate_usd
            limit = policy.daily_limit_usd
            if policy.mode != "observe" and projected > limit:
                if "blocked" not in ledger.alerted:
                    ledger.alerted.add("blocked")
                    alerts.append(Alert(tenant, day, "blocked", 1.0, ledger.committed, limit))
                decision = Decision("block", None, ledger.exposure, limit, f"projected {projected:.4f} > limit {limit:.4f}")
            else:
                action: Action = "allow"
                if policy.mode == "degrade" and projected > limit * policy.soft_limit_fraction:
                    action = "degrade"
                res = Reservation(uuid.uuid4().hex[:12], tenant, day, estimate_usd)
                ledger.reserved[res.id] = estimate_usd
                decision = Decision(action, res, ledger.exposure, limit)
        for a in alerts:
            self.on_alert(a)
        return decision

    def commit(self, reservation: Reservation, actual_usd: float) -> None:
        """Replace the reservation with the actual cost and fire any threshold alerts it crosses."""
        policy = self.policy_for(reservation.tenant)
        alerts: list[Alert] = []
        with self._lock:
            if not reservation.open:
                return
            reservation.open = False
            # The reservation's day, not today's: a request that straddles midnight is charged where it started.
            ledger = self._ledger(reservation.tenant, reservation.day)
            ledger.reserved.pop(reservation.id, None)
            ledger.committed += actual_usd
            for threshold in policy.alert_thresholds:
                tag = f"t{threshold}"
                if ledger.committed >= threshold * policy.daily_limit_usd and tag not in ledger.alerted:
                    ledger.alerted.add(tag)
                    alerts.append(Alert(reservation.tenant, reservation.day, "threshold", threshold, ledger.committed, policy.daily_limit_usd))
        for a in alerts:
            self.on_alert(a)  # outside the lock: a slow pager must not stall every request
```

The client wrapper estimates, admits, degrades and commits:

```python
# path: book/projects/examples/ch30/budgets.py  (excerpt; full file on disk)
    def _priced(self, model: str) -> str:
        # An unknown model prices at $0 and would slip past every limit: refuse instead (fail closed).
        if self.pricing.lookup(model) is None:
            raise InvalidRequestError(f"no price for model {model!r}; refusing unmetered spend")
        return model

    def estimate_usd(self, req: CompletionRequest) -> float:
        """Worst case: every prompt token uncached, the full output budget used."""
        usage = Usage(input_tokens=count_message_tokens(req.messages, req.model), output_tokens=req.max_tokens)
        return self.pricing.cost_usd(self._priced(req.model or self.default_model), usage)

    def _admit(self, req: CompletionRequest) -> tuple[CompletionRequest, Reservation]:
        tenant = req.metadata.get("tenant")
        if not tenant:
            raise InvalidRequestError("request has no tenant in metadata; refusing unattributed spend")
        decision = self.guard.reserve(str(tenant), self.estimate_usd(req))
        if decision.action == "block" or decision.reservation is None:
            raise BudgetExceededError(f"daily budget exhausted for tenant {tenant}: {decision.reason}")
        reservation = decision.reservation
        if decision.action == "degrade" and self.degrade_model:
            degraded = req.model_copy(
                update={
                    "model": self.degrade_model,
                    "max_tokens": min(req.max_tokens, self.degrade_max_tokens),
                    "metadata": {**req.metadata, "budget_degraded": True},
                }
            )
            # Re-reserve at the cheaper estimate so the ledger does not hold the expensive one.
            self.guard.release(reservation)
            redo = self.guard.reserve(str(tenant), self.estimate_usd(degraded))
            if redo.reservation is None:
                raise BudgetExceededError(f"daily budget exhausted for tenant {tenant}: {redo.reason}")
            return degraded, redo.reservation
        return req, reservation

    def _actual_usd(self, completion: Completion, admitted_model: str) -> float:
        raw = completion.raw or {}
        if raw.get("cache_hit"):
            return 0.0
        if "cost_usd" in raw:
            return float(raw["cost_usd"])
        # A router may return a model id the table does not price; the call already happened, so
        # charge it at the admitted model's price rather than failing and leaking the reservation.
        model = completion.model if self.pricing.lookup(completion.model) else admitted_model
        return self.pricing.cost_usd(model, completion.usage)

    def complete(self, req: CompletionRequest) -> Completion:
        admitted, res = self._admit(req)
        try:
            completion = self.inner.complete(admitted)
        except BaseException:
            self.guard.release(res)  # assumes a failed call was not billed; see the chapter for when it is
            raise
        self.guard.commit(res, self._actual_usd(completion, admitted.model or self.default_model))
        return completion
```

### Fan-out under a deadline

```python
# path: book/projects/examples/ch30/parallel.py  (excerpt; full file on disk)
async def fan_out(steps: Mapping[str, Step], *, deadline_s: float, max_concurrency: int | None = None) -> FanOutResult:
    """Run `steps` concurrently; return when all finish or the deadline passes.

    Raises `RequiredStepFailed` if a required step errors or times out. Optional failures are
    reported in the result so the caller can answer in a degraded mode (e.g. without reranking).
    """
    loop = asyncio.get_running_loop()
    start = loop.time()
    end = start + deadline_s
    sem = asyncio.Semaphore(max_concurrency) if max_concurrency else None
    out = FanOutResult()

    async def run(name: str, step: Step) -> Any:
        async def call() -> Any:
            t0 = loop.time()
            try:
                remaining = max(0.0, end - loop.time())
                budget = remaining if step.timeout_s is None else min(step.timeout_s, remaining)
                return await asyncio.wait_for(step.fn(), timeout=budget)
            finally:
                out.step_ms[name] = (loop.time() - t0) * 1000

        if sem is None:
            return await call()
        async with sem:
            return await call()

    tasks = {name: asyncio.create_task(run(name, step), name=name) for name, step in steps.items()}
    required = {tasks[n] for n, s in steps.items() if s.required}
    pending: set[asyncio.Task[Any]] = set(tasks.values())
    try:
        while pending:
            done, pending = await asyncio.wait(pending, timeout=max(0.0, end - loop.time()),
                                               return_when=asyncio.FIRST_EXCEPTION)
            if not done:
                break                                          # the deadline passed
            if any(t in required and not t.cancelled() and t.exception() is not None for t in done):
                break                                          # a required step failed: stop the rest now
    finally:
        # Runs on deadline, on a required failure, and when the caller itself is cancelled (client
        # disconnect): a stopped HTTP call stops costing, and nothing outlives the request.
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
    for name, task in tasks.items():
        if task in pending or task.cancelled():
            out.timed_out.append(name)
            continue
        exc = task.exception()
        if isinstance(exc, asyncio.TimeoutError):
            out.timed_out.append(name)  # the step's own timeout fired before the global deadline
        elif exc is not None:
            out.errors[name] = exc
        else:
            out.results[name] = task.result()
    out.elapsed_ms = (loop.time() - start) * 1000
    for name, step in steps.items():
        if step.required and name not in out.results:
            raise RequiredStepFailed(name, out.errors.get(name))
    return out
```

The prefetcher refuses to speculate on anything not registered as read-only and counts the prefetches it throws away:

```python
# path: book/projects/examples/ch30/parallel.py  (excerpt; full file on disk)
class Prefetcher:
    """Speculative execution of read-only calls, keyed by (tool, normalized arguments).

    Only register tools that are read-only and idempotent: a speculative `send_reply` is a bug.
    """

    def __init__(self, read_only_tools: set[str]) -> None:
        self.read_only_tools = set(read_only_tools)
        self._inflight: dict[Hashable, asyncio.Task[Any]] = {}
        self.stats = PrefetchStats()

    @staticmethod
    def key(tool: str, args: Mapping[str, Any]) -> Hashable:
        return (tool, tuple(sorted((k, repr(v)) for k, v in args.items())))

    def start(self, tool: str, args: Mapping[str, Any], fn: Callable[[], Awaitable[Any]]) -> None:
        if tool not in self.read_only_tools:
            raise ValueError(f"refusing to prefetch {tool!r}: not registered as read-only")
        k = self.key(tool, args)
        if k not in self._inflight:
            self._inflight[k] = asyncio.create_task(fn())
            self.stats.started += 1

    async def take(self, tool: str, args: Mapping[str, Any], fn: Callable[[], Awaitable[Any]]) -> Any:
        """Return the prefetched result if one matches, else call `fn` now."""
        task = self._inflight.pop(self.key(tool, args), None)
        if task is not None:
            self.stats.used += 1
            return await task
        self.stats.direct += 1
        return await fn()

    async def cancel_unused(self) -> int:
        tasks = list(self._inflight.values())
        self._inflight.clear()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.stats.wasted += len(tasks)
        return len(tasks)
```

## Code walkthrough

**Attribution.** `AttributingTracer.export` uses `setdefault`, so a span's own attribute wins over a bound one. Export runs when a span closes, so a span must close inside the `bind` block; a test checks that the gateway's own `llm.complete` span picks up the tenant.

**Budgets and deadlines.** An infeasible `LatencyBudget` cannot be constructed. This chapter's `Deadline` is deliberately small: it turns a budget into per-stage timeouts inside one process. Chapter 29's `reliability.Deadline` is the production one, with a header to carry the remaining time across services, child deadlines and cancellation; a service using both creates the Chapter 29 deadline at the edge and asks this budget only for each stage's share. `Deadline.timeout_for` returns seconds, ready for `CompletionRequest.timeout_s` or `asyncio.wait_for`, and returns zero once time is gone so callers fail fast. `LatencyReport` carries both the wall-clock end-to-end p95 and the sum of stage p95s: when they are close, the path is serial and every stage's tail reaches the user. `worst_stage` ignores the synthetic `end_to_end` and `ttft` violations, which say that something was slow but not what.

**Keys.** `make_key` hashes a sorted JSON object under a namespace, so component order never changes a key and a tenant prefix can be purged. `Scope.acl_scope` hashes sorted groups, so group order does not matter and group names stay out of keys. The embedding cache's `space` fingerprint changes with model version, dimensions, prefix or text preparation; the tests change each and assert a miss.

**Semantic cache.** `lookup` filters by eligibility before comparing similarity, and also tracks the best similarity among ineligible entries, a measure of what the scope check prevented. Linear scan is fine for thousands of entries; beyond that, keep the vectors in your vector store with tenant and scope as filter columns (Chapter 9).

**Cost and chargeback.** `mix` sums weighted spend and weighted successes separately before dividing. `what_if` makes "what do 1,000 more tokens cost" a one-liner. `chargeback` counts `cache_hit` spans as avoided cost and puts tenantless spans in `_unattributed`; a test runs it without a pricing table to prove the hit's value comes from the span itself.

**Guard and fan-out.** `reserve` computes exposure as committed plus open reservations, so five concurrent 0.3 USD requests against a 1 USD limit yield three allows and two blocks. `commit` charges the reservation's day, so a request straddling midnight lands where it started. A stream stopped after output started commits the usage seen or, failing that, the reserved estimate: an unknown cost is overcounted rather than lost. A stream that fails before any output (a 429, say) releases its reservation, so an outage does not leave phantom spend that blocks the tenant for the rest of the day. `fan_out` stops waiting as soon as a required step fails, cancels and awaits the rest before raising `RequiredStepFailed`, and does the same when the caller itself is cancelled (a client disconnect), so nothing keeps running, or billing, after the request ends.

## Production considerations

**Latency.** Put stage budgets into code as timeouts. Alert on per-stage p95 against budget so the page names the stage. Rising gateway `queue_ms` with flat model latency means a rate limit or concurrency cap, which calls for capacity or admission control (Chapter 29), not a faster model. Measure TTFT at the client too.

**Cost.** Record usage and cost on every span, attribute every span, and alert on the unattributed share. Keep prices in versioned configuration so history can be recomputed at the prices then in force. Compare cost per successful task in canaries, not cost per call. Move every offline job to a batch API.

**Security.** Every key below the gateway includes tenant and ACL scope, with the linter in CI. Caches store only validated, grounded, non-personalized answers, and tenants and sources can be purged fast for poisoning incidents and deletion requests. Spend limits turn a prompt-injected loop or abusive client into a bounded bill (Chapter 26).

**Operations.** Index version, prompt version and embedding-space fingerprint are deployment artifacts bumped by the pipeline, because cache correctness depends on them. With more than one replica, the spend ledger must live in a shared store (Redis or PostgreSQL); the in-memory ledger here would let each replica spend the whole limit. Reconcile the ledger daily against the provider's usage export. Review semantic-cache near misses and false-hit samples on a schedule.

## Common mistakes

- **Caching before trimming.** Building a semantic cache for a workload whose largest cost is 9,000-token prompts and uncapped answers. Trim first; the cache then has less to save and less risk to carry.
- **Cost per call as the metric.** Choosing a model because its calls are cheaper while its success rate, retries or review rate make each success more expensive.
- **Unbounded output.** No `max_tokens` and no answer-format instruction, so output length, the dominant latency term, is whatever the model feels like.
- **Counting cache hits as spend.** Summing `cost_usd` over all gateway spans, including hits that were never billed.
- **Fan-out without cancellation.** Letting optional branches run after the deadline, so they keep consuming rate limit and money for answers nobody will read.
- **Batching interactive paths.** A 200 ms micro-batch window added to every user's TTFT.

## Failure modes

**Cross-scope cache leak.** An HR-only answer appears for a non-HR user. In telemetry: a cache hit span whose request's groups do not cover the cited document's ACL. Root cause: a key without ACL scope, or ACL enforced outside the prompt while the response cache keyed only on the prompt. Such a key passes every single-tenant test. Test: the linter and a two-scope miss test.

**Stale answer after reindex.** A policy changed an hour ago and answers still quote the old text. In telemetry: hits whose cached `index_version` differs from the current one, or answers citing chunk versions that no longer exist. Root cause: no index version in the key, or a long TTL with no invalidation. Test: version-bump and source-invalidation tests.

**Semantic false hit.** Users get a fluent answer to a neighboring question. In telemetry: thumbs-down and follow-up rephrasings concentrated on semantic-cache hits; hits with similarity just above threshold. Root cause: a threshold tuned for hit rate rather than false-hit rate. Test: false-hit rate on labeled question pairs as a release gate.

**Prefix cache collapse.** Cost per answer rises about 14 percent overnight with no traffic change. In telemetry: `cached_input_tokens / input_tokens` drops after a prompt deploy. Root cause: a volatile value (a timestamp, a request id, the user's name) moved into the prefix, or tool definitions reordered. Test: Chapter 5's prefix-stability check in CI.

**Retry storm.** A provider slowdown triggers timeouts, timeouts trigger retries, and retries add load and cost. In telemetry: `attempt` greater than one on a rising share of spans, cost per request rising while successes fall. Root cause: retries without a budget, or timeouts shorter than normal p99. Test: fault injection against the retry budget (Chapter 29).

**Runaway agent loop.** One task makes forty calls. In telemetry: steps per task distribution with a long tail, single tasks with outsized cost in chargeback. Root cause: no task-level token budget or step cap. Test: a fake model that never finishes, stopped by `TaskTokenBudget`.

**Budget overshoot.** A tenant's spend for the day exceeds its limit by several times. In telemetry: many requests admitted within a short window, each seeing the same remaining budget. Root cause: check without reservation, or one in-memory ledger per replica. Test: concurrent and multi-replica reservation tests.

**Tail amplification from fan-out.** p95 latency rises after adding a branch whose own p95 is fine. In telemetry: the request's end to end equals the slowest branch's duration, and that branch varies between requests. Root cause: more parallel branches without per-branch timeouts. Test: a randomly slow branch and an assertion on group p95.

## Tradeoffs

**Freshness versus hit rate.** Longer TTLs raise hit rates and widen the window of stale answers; invalidation by source id makes long TTLs safe at the cost of tracking citations.

**Semantic versus exact caching.** Exact caching never serves a wrong answer and rarely hits on free text; semantic caching hits far more and sometimes answers the wrong question. Use it only for repetitive, impersonal traffic with a measured false-hit rate.

**Speculation versus spend.** Prefetching and hedging cut tails by paying for calls that may be discarded. Worth it for read-only calls on the TTFT path with a high use rate.

**Streaming versus validation.** Streaming improves perceived latency and makes full-output validation retroactive. Stream low-risk text; buffer high-risk or structured output.

**Degrade versus block.** Degrading keeps service up at lower quality; blocking protects the budget exactly. Internal tools usually prefer degrade; contractual caps require enforce.

**Batch latency versus throughput.** Bigger batches lower unit cost and raise every item's latency: milliseconds for interactive paths, hours and the discount for offline ones.

**Self-hosting versus per-token pricing.** Self-hosting wins at sustained high utilization once engineer time and idle failover are counted (Chapter 34); the `infra_usd` term makes the two comparable.

## Evaluation and testing

Unit tests cover arithmetic and invariants. The chapter's suite checks that budgets reject infeasible splits, the tracker measures wall clock over overlapping spans, every cache layer misses when any scope or version component changes, the semantic cache refuses ineligible candidates and personalized writes, the linter flags a naive key, the cost model matches hand-computed numbers including the cheap-agent case, chargeback separates tenants and avoided cost, reservations prevent concurrent overshoot, alerts fire once per threshold, and fan-out costs the maximum, not the sum. `test_hardening.py` adds the edge cases: an unpriced model is refused, a stream that fails before output costs nothing, a failed required step and a cancelled caller stop every sibling, parallel calls to one stage count their wall time once, and the nearest-rank percentile is exact at p99.9. Timing tests use fake clocks or generous bounds so they do not flake.

Every optimization also needs a quality check, because most can reduce quality. Trimming context, switching models, adding a semantic cache or capping output each go through the offline evaluation set (Chapters 24 and 25), and the measured success rate is what enters the cost model. A change that saves 10 percent per call but drops success from 0.90 to 0.75 raises cost per successful task by about 8 percent (0.9 × 0.90 / 0.75 = 1.08), and the extra failures land on users.

Performance needs realistic load: queueing, rate limits, cache hit rates and tails only appear under concurrency with a realistic prompt-length mix. Use Chapter 34's load-test protocol against staging with production-like caches, report per-stage p95 from traces, and compare the canary's cost per successful task and stage latencies against control before widening a rollout (Chapter 32).

## Before you ship

- [ ] A `LatencyBudget` with p95 TTFT and completion targets exists in code, constructs without error, and each stage's timeout comes from `Deadline.timeout_for` or Chapter 29's deadline.
- [ ] Dashboards show per-stage p50, p95 and p99 from traces, with an alert on each stage's p95 against its budget, and TTFT is also measured at the client.
- [ ] Every model call has an explicit `max_tokens` sized for the answer format (and for reasoning tokens on reasoning models), and agent or chain tasks run under a `TaskTokenBudget` or step cap.
- [ ] The cache-key linter runs in CI over every cache class, with zero errors; a two-scope test proves each cache misses when tenant or ACL scope differs.
- [ ] Index version, prompt version and embedding-space fingerprint are bumped by the deploy pipeline, and a test proves a version bump retires old cache entries.
- [ ] The semantic cache, if enabled, has a threshold chosen on labeled question pairs, a false-hit rate gate in the release pipeline, source-id invalidation wired to document updates, and refuses personalized and ungrounded writes.
- [ ] The prompt-cache ratio (`cached_input_tokens / input_tokens`) per prompt version is on a dashboard, and a prefix-stability check runs in CI (Chapter 5).
- [ ] Every span carries tenant, request id and feature; the unattributed share of spend is a metric with an alert near zero.
- [ ] Each tenant has a `SpendPolicy` with a mode, a daily limit and alert thresholds; the ledger lives in a shared store when more than one replica runs, and a concurrent-reservation test passes against it.
- [ ] Prices live in versioned configuration, models missing from the price table are refused, and the spend ledger is reconciled daily against the provider's usage export.
- [ ] Fan-out groups have a deadline, per-branch timeouts and a concurrency bound, and a test proves a cancelled caller stops every branch.
- [ ] Every optimization was compared on the offline evaluation set, and the canary reports cost per successful task, not cost per call.

## Exercises

**Start here:** K3, K4, E1, P3, D1 (about 4 hours). The rest go deeper.

### Knowledge questions

K1. Explain why the end-to-end p95 latency of a request is usually lower than the sum of its stage p95s, and why the sum is still useful in budgeting.

K2. A team says provider prompt caching will cut their cost because "the model will remember the previous conversation". What is wrong with this description, and what actually determines a cache hit?

K3. List the components a retrieval-cache key needs for a multi-tenant assistant with document ACLs, and state what goes wrong if each one is missing.

K4. Why is cost per successful task a better unit than cost per model call when comparing a single strong-model call with a multi-step agent on a cheaper model?

K5. Name the three things called batching in an AI system and the situation in which each one is appropriate.

K6. Why does a spend guard need reservations rather than a check of remaining budget before each call?

### Engineering questions

E1. Northwind wants to add a guardrail model call after generation that checks every answer for policy violations. It takes 600 ms at p95. Using the chapter's budget (8 s total, 2 s TTFT, streaming answers), where can it go, and what design choices does each placement force?

E2. The logistics tenant has a 9 percent semantic-cache hit rate, and retail has 2 percent. Retail's product owner asks to lower the similarity threshold for retail only. What data would you collect before deciding, and what would make you say no?

E3. A finance stakeholder asks for chargeback that includes the shared vector database, the platform team's on-call cost and an idle failover GPU. Propose an allocation rule, explain what behavior it encourages in tenants, and name one rule you would avoid.

E4. You run three replicas of the API behind a load balancer and the in-memory `SpendGuard`. Describe the failure this causes and design the shared-ledger replacement, including what happens when the ledger store is unavailable.

E5. Northwind is considering one system prompt per tenant, each about 3,000 tokens, instead of one shared prompt. The provider charges an illustrative 1.25 times the input price to write a prefix into its cache, 0.1 times to read it, and entries live 5 minutes after their last use. The retail tenant sends about 40 requests an hour and a small tenant about 3 an hour. Estimate the expected prefix cost per request for each tenant with and without the split, and recommend a layout.

### Practical exercises

P1. (about 90 min) Extend `LatencyTracker` to report, per stage, the share of end-to-end violations in which that stage itself exceeded its budget. Add a test with synthetic spans where retrieval causes most violations.

P2. (about 2 hours) Implement a `ToolResultCache` for read-only tools with per-tool TTLs, a tool-version component, and a refusal to cache any tool not in a read-only registry. Declare its `key_components` and make it pass `lint_cache_key("tool", ...)`.

P3. (about 2 hours) Build a threshold-tuning script for `SemanticCache`: given labeled question pairs (same answer or not), compute hit rate and false-hit rate for thresholds from 0.70 to 0.99 using `FakeEmbeddings(vocabulary=...)`, and pick the lowest threshold whose false-hit rate is at or below a target.

P4. (about 2 hours) Add an hourly spend anomaly detector to `cost.py`: from trace JSONL, compute spend per tenant per hour and flag hours more than three times the trailing 7-day median for the same hour of the week. Test it with a synthetic spike.

### Debugging exercises

D1. After a prompt release, cost per answer rose about 14 percent while traffic, model and average input tokens were unchanged. Traces show `cached_input_tokens` per call fell from about 800 to near zero. The diff of the release shows the system prompt now starts with "You are Northwind Assist. Today is {date} {time}." and tool definitions are emitted from a Python `set`. Diagnose the cause and the fix, and name the telemetry that confirms the fix.

D2. A warehouse supervisor in the logistics tenant reports seeing an answer that quotes salary bands, which only HR should see. The retrieval cache hit rate is 35 percent. The retrieval cache key is built from `normalize_text(query)`, `tenant` and `index_version`. ACL filtering happens in the SQL query on a miss. Explain how the leak happened, which spans show it, and what change and test prevent recurrence.

D3. Retail's daily limit is 50 USD in `enforce` mode, yet yesterday's committed spend was 210 USD. The guard logged no blocked alert until 14:05, and traces show 4,000 requests admitted between 13:58 and 14:05 from a batch summarization job. The service runs four replicas. Identify the contributing causes and the fixes.

## Key takeaways

- Latency adds along the critical path; budget each stage, mark the ones before the first token, keep a reserve, and enforce budgets as deadlines that propagate into every call.
- Measure latency as per-stage distributions from traces, and end to end as wall clock; the stage with the most budget violations is where to look first.
- Tokens are the unit of both latency and cost. Output tokens dominate completion time; trimming context and capping output is usually the largest, safest optimization.
- A cache is only as correct as its key: include every factor that changes the answer (tenant, ACL scope, index, prompt and embedding-space versions) and nothing per-request, and lint keys in CI.
- Semantic caching has the worst failure mode of any layer; check eligibility before similarity, refuse personalized and ungrounded writes, invalidate by source, and tune the threshold on false-hit rate.
- Streaming moves perceived latency to TTFT; fan-out moves latency to the slowest required branch, at the price of tail amplification.
- Compare designs by cost per successful task, which charges each design for its calls, retries, failures and human review.
- Attribute every span to a tenant and feature, count cache hits as avoided cost rather than spend, and enforce per-tenant limits with reservations, deduplicated alerts and a degrade mode.
- Optimize in order: measure, right-size the model, trim context and output, cache, parallelize and batch, and only then change infrastructure.

## Further reading

- *The Tail at Scale* (Dean and Barroso, 2013): why fan-out amplifies tail latency and how hedged and tied requests cut it; the background for this chapter's parallelization section.
- *A Proof for the Queuing Formula: L = λW* (Little, 1961): the law behind every in-flight and capacity estimate in the back-of-envelope section.
- *Site Reliability Engineering* (Beyer et al., eds., 2016): service-level objectives and error budgets, the frame that latency budgets apply stage by stage.
- *Efficiently Scaling Transformer Inference* (Pope et al., 2023): why prefill and decode cost what they do, which explains the TTFT and TPOT terms in the latency equation.
- *SGLang: Efficient Execution of Structured Language Model Programs* (Zheng et al., 2024): prefix sharing in a serving engine, the self-hosted side of the prompt-cache arithmetic.
