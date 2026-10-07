# Appendix — Interview Preparation

An AI engineering interview tests whether you can reason from invariants: probabilistic output,
context limits, retrieval recall, authorization boundaries, queueing, latency, evaluation, failure
recovery. Vendor names age in months; the invariants do not. This appendix gives you a structure for
any answer, question banks with outlines keyed to the chapter where the material lives, the arithmetic
that makes answers credible, the behavioral signals interviewers score, a drill method, and the weak
answers to stop giving. Use it once your recall notes from the roadmap exist; the outlines are
skeletons, your notes and project code are the flesh.

## 1. Answer in layers

Every concept question has the same six layers. Say them in this order, compress or expand to fit
the time, and stop when the interviewer redirects.

1. **Definition.** One sentence. What it is, in words a backend engineer understands.
2. **Mechanism.** How it works: the data flow, the equation, or the state machine. Two to four sentences.
3. **Trade-off.** What it costs and what it buys, on named dimensions: latency, cost, quality, complexity, security surface.
4. **Failure mode.** One concrete way it breaks and what the telemetry looks like when it does.
5. **Measurement.** The metric and the dataset you would use to know it is working.
6. **Production example.** Where you used it or would use it, with one number.

Example, "What is reranking?" (Chapter 12): *A second-stage scorer that reorders first-stage
candidates with a model that sees query and passage together.* / *Dense and BM25 score query and
document independently, fast but coarse. A cross-encoder reads the pair and scores relevance; apply it
to the top 50, keep the top 5.* / *It raises precision at the top, which is what the context window
sees, for one extra model pass per query, usually the largest item in the retrieval latency budget.* /
*If first-stage recall is poor it cannot recover the missing evidence; the stage-isolation report
blames retrieval, not rerank.* / *nDCG at 5 and recall at 5 on a gold set, before and after.* / *In
Northwind Assist a local cross-encoder over 50 candidates raised recall at 5 from the mid-sixties to
the mid-eighties on the HR slice, illustratively, for about 300 ms at p95.*

That is about 130 words and 90 seconds. Most answers should be.

## 2. Question banks

Each entry gives the question, the outline of a strong answer, and the chapter where the material
lives. Outlines name what must be present; they are not scripts.

### 2.1 Foundations and model behavior (Chapters 1-2, 7)

1. **Why is LLM output nondeterministic, and what do you do about it?** Sampling from a token
   distribution shaped by temperature and top-p; even greedy decoding varies with batching and
   numerics. Design for distributions: validation, retries, evaluation over many cases. (Ch 2)
2. **Why does generation slow down as the conversation grows?** Prefill is parallel; decode is one
   token at a time and reads the whole KV cache per step, so per-token cost grows with context.
   Memory bandwidth, not compute, bounds decode. (Ch 2)
3. **What is the KV cache and why does an application engineer care?** Stored keys and values for
   earlier tokens so attention does not recompute them; the memory cost of long context and the
   mechanism prefix caching exploits. Give the bytes-per-token formula. (Ch 2, 34)
4. **What is "lost in the middle" and how do you design around it?** Models attend better to the
   beginning and end of long contexts. Order evidence by importance, put instructions at the ends, keep
   context small, measure attribution. (Ch 2, 5)
5. **Why do models fail at counting and arithmetic?** Tokenization splits numbers unpredictably and
   the model predicts tokens, not computes. Route arithmetic to a tool; never trust a model total. (Ch 2)
6. **Temperature zero: is the output deterministic?** Mostly but not guaranteed; batching, hardware,
   and provider changes alter results. Tests assert on structure and semantics, not exact strings. (Ch 2)
7. **What does the context window actually limit?** Input plus output tokens; the output reservation
   must be subtracted from the budget. Quality degrades before the hard limit. (Ch 2, 5)
8. **Small or large model?** Evaluate on your gold set; small models win on latency and cost for
   narrow, well-specified tasks; cascade with confidence routing when error cost is asymmetric. (Ch 7)
9. **What is a reasoning model and when is it worth the cost?** Test-time compute: intermediate
   tokens before the answer. Worth it on multi-step problems with verifiable answers; wasteful on
   extraction and classification. Measure quality per dollar. (Ch 2, 7)
10. **Walk me up the decision ladder.** Prompt, retrieval, tools, fine-tuning, agent; each rung adds
    nondeterminism, cost, and operational surface. Climb only when the lower rung is measured
    insufficient. (Ch 1)

### 2.2 LLM APIs and reliability (Chapters 3, 29)

1. **Design a provider-neutral LLM client.** A protocol with complete and stream, typed messages and
   tool specs, a usage object, an error taxonomy with a retryable flag, adapters per provider, and a
   fake for tests. (Ch 3)
2. **Which errors do you retry?** Rate limits with the server's retry-after, 5xx, timeouts, malformed
   responses once with a repair prompt. Never 4xx validation errors or content filters. Exponential
   backoff with jitter and a cap. (Ch 3, 29)
3. **How do you propagate deadlines through a chain of calls?** One request deadline; each stage
   gets remaining time minus a margin; a stage that cannot fit returns a degraded result rather than
   exceeding the budget. (Ch 3, 29)
4. **Streaming with structured output?** Stream text deltas for display but validate only the
   completed JSON; partial JSON parsers are for display, not decisions. (Ch 3, 13)
5. **Rate limits: client side or server side?** Both. Token bucket on requests and tokens per minute
   in the gateway, concurrency semaphores, admission control that sheds load before the provider
   returns 429s. (Ch 3, 29)
6. **What does a fallback chain change besides the model name?** Context length, tool support,
   structured-output quality, price, latency, and behavior under the same prompt. A fallback needs its
   own evaluation and its own prompt version. (Ch 3, 7)
7. **Where does caching help and where does it lie?** Exact-match response cache for repeated requests;
   provider prompt caching for a stable prefix; semantic caches return stale or wrong answers unless
   the key includes tenant, permissions, and prompt version. (Ch 3, 15, 30)
8. **How do you account cost per request?** Usage tokens times an illustrative pricing table in the
   gateway, attributed to tenant and feature in the trace, including retries and fallbacks. (Ch 3, 30)

### 2.3 Prompt and context engineering (Chapters 4-5)

1. **What is a prompt, as an engineering artifact?** A versioned contract between code and model:
   role, task, constraints, evidence slot, output schema, failure behavior; in a registry, tested with
   golden cases. (Ch 4)
2. **What is the instruction hierarchy?** System over developer over user over data. Models follow it
   imperfectly; label data as data and enforce the hierarchy in code where it matters. (Ch 4, 26)
3. **When do few-shot examples hurt?** They cost tokens on every call, anchor the model to their
   format and distribution, and leak into outputs; select per request and measure against zero-shot. (Ch 4)
4. **How do you test a prompt?** Golden cases with deterministic structural assertions, a judge
   rubric for semantic quality, regression in CI on every version change, slices by input type. (Ch 4, 24)
5. **Context is a budget, not a bucket. In code?** Reserve output tokens, then stable prefix, then
   state, then evidence by priority; truncate from the lowest priority; log what was dropped. (Ch 5)
6. **How do you compact a long conversation?** Old turns become structured state with fields that
   must never be lossy (identifiers, decisions, constraints) plus prose; recent turns stay verbatim;
   the compaction prompt is versioned. (Ch 5, 21)
7. **Why does prompt layout affect cost?** Prefix caching keys on identical leading tokens; a dynamic
   timestamp at the top defeats it. Stable content first, volatile content last. (Ch 5, 30)
8. **When is prompting not enough?** When the task needs knowledge the model lacks, actions, stable
   behavior across thousands of inputs, or scale economics; each points to the next rung. (Ch 4, 1)

### 2.4 Embeddings and retrieval (Chapters 8-9, 11-12)

1. **What is an embedding and how is one trained?** A vector whose geometry encodes similarity,
   trained contrastively so matching pairs are close and non-matching pairs far. (Ch 8)
2. **Cosine or dot product?** Equivalent after normalization; on unnormalized vectors dot product
   mixes magnitude into relevance. Normalize at index time. (Ch 8)
3. **Query and passage asymmetry?** Short questions and long passages occupy different regions;
   instruction-tuned models handle it; embed queries as queries. (Ch 8)
4. **When do you not need a vector database?** Up to roughly a million vectors, exact NumPy search
   is milliseconds with no recall loss; pgvector covers the next order of magnitude with filters and
   transactions. Dedicated stores are for scale and operations, not quality. (Ch 9)
5. **Explain HNSW in one minute.** A layered proximity graph; search descends from sparse top layers
   to the dense bottom; ef and M trade recall against latency and memory; build time and deletions are
   the operational costs. (Ch 9)
6. **Why is metadata filtering hard with ANN?** Post-filtering empties the top-k; pre-filtering
   disables the index; good systems filter inside the traversal or partition by filter value.
   Permissions are a filter, so this is a security question too. (Ch 9, 15)
7. **How do you pick a chunk size?** By evaluation across sizes on a gold set; document-aware
   boundaries beat fixed sizes; parent-child decouples the match unit from the context unit. (Ch 11, 14)
8. **BM25 versus dense: when does each win?** BM25 on exact terms, identifiers, rare words, code;
   dense on paraphrase and intent. Hybrid with reciprocal rank fusion is the default; give the RRF
   formula. (Ch 12)
9. **What does a reranker fix and what can it not fix?** Precision at the top; it cannot recover
   evidence the first stage missed. Size the candidate funnel accordingly. (Ch 12)
10. **When does HyDE help and when does it hurt?** Helps when the query is terse and the model knows
    the domain vocabulary; hurts when the model does not know the answer and hallucinates a confident
    hypothetical that retrieves confident wrong evidence. (Ch 12)

### 2.5 RAG in production (Chapters 10, 13-15)

1. **Why RAG instead of fine-tuning or a long context?** Fresh facts, citations, permissions, cost;
   fine-tuning teaches behavior, not facts; long context costs tokens per call and loses
   attribution. (Ch 10, 33)
2. **A RAG answer is wrong. How do you debug it?** Was gold evidence retrieved, did it survive
   reranking and packing, did generation use it; the stage-isolation report says which. (Ch 14)
3. **How do you enforce permissions in RAG?** ACL filter at retrieval with identity from the request;
   never after generation; permission scope in cache keys; cross-tenant leakage tests in CI. (Ch 15)
4. **What is the grounded answer contract?** Evidence is data, not instructions; cite or abstain; a
   structured answer with claims, citation IDs, confidence, and missing-evidence field; a validator
   checks IDs exist and spans support claims. (Ch 13)
5. **Document updates and deletions?** Versioned documents, content hashes for incremental indexing,
   deletion propagated to index, caches, and derived summaries; a freshness SLO from source change to
   retrievable. (Ch 15)
6. **How do you cache in RAG without serving wrong answers?** Embedding cache keyed on text and model
   version; retrieval cache on query, filters, tenant, index version; response cache only with prompt
   version and permission scope in the key; TTLs tied to freshness. (Ch 15, 30)
7. **How do you reduce hallucination in generation?** Smaller, cleaner evidence; quote-then-answer;
   claim-level verification; abstention when evidence is thin; self-consistency only where its cost is
   justified. (Ch 13)
8. **What is the latency budget of a RAG request?** The p95 target split across auth, embedding,
   retrieval, rerank, packing, TTFT, decode; rerank and TTFT dominate; parallelize retrieval; stream. (Ch 15, 30)

### 2.6 Tools, agents, and MCP (Chapters 16-23, 38)

1. **What is a tool to a model?** A JSON Schema the model fills in; code validates, authorizes,
   executes, and returns the result. The model never executes anything. (Ch 16)
2. **How do you classify side effects?** Read, reversible write, irreversible write, external
   communication; each class gets its own policy, from free through logged and idempotent to approval
   bound to arguments. (Ch 16)
3. **Why are idempotency keys essential for agents?** Retries, replays, and loops re-issue calls; a
   key derived from intent and arguments makes the second execution a no-op. (Ch 16, 29)
4. **What is approval "bound to arguments"?** The approval covers a hash of the exact tool and
   arguments; any change invalidates it; prevents approve-then-swap. (Ch 16)
5. **Workflow or agent?** Known path: workflow, deterministic and cheap. Path depends on
   observations: agent. Present the decision table; default to the workflow. (Ch 17)
6. **Define the agent loop precisely.** Model, tools, typed state, bounded loop: plan, act, observe,
   update state, check termination; every step appended to an event log. (Ch 19)
7. **Name six termination conditions.** Success validated against a Definition of Done, max steps,
   token budget, time budget, repeated state, no progress, approval required. (Ch 19)
8. **How do you debug an agent?** Replay the event log; inspect state per step; diff the trajectory
   against a known-good run; trajectory assertions in tests. (Ch 19, 25, 31)
9. **Compare planner-executor with ReAct.** Planner-executor front-loads structure, cheaper per step
   but brittle under surprises; ReAct adapts per step but wanders and costs more; replanning is the
   middle. (Ch 20)
10. **When is multi-agent justified?** Parallelism, specialization, permission isolation, independent
    verification. Compare against a single-agent baseline with budgets at parent and child. (Ch 22)
11. **What does MCP solve and not solve?** Standard discovery and invocation of tools, resources, and
    prompts across hosts; not authorization, trust of tool descriptions, or egress control. Host,
    client, server roles; stdio and HTTP transports. (Ch 18)
12. **How do you make an agent durable across a restart?** Make an append-only event log the source of
    truth and rebuild state by folding it (checkpoints are only a cache of that fold), make side
    effects idempotent under at-least-once delivery, treat approvals as first-class states, and fence
    workers with leases. (Ch 38)

### 2.7 Evaluation (Chapters 14, 24-25)

1. **How would you choose between two models for our product?** Representative gold set, defined
   metrics, both run offline, bootstrap confidence intervals, per-case deltas and slices, then canary.
   Not a leaderboard. (Ch 24, 7)
2. **What goes into a golden dataset?** Representative and adversarial inputs, expected outputs or
   rubrics, required evidence for RAG, permission context, slice tags; a frozen holdout separate from
   the development set. (Ch 24)
3. **When is an LLM judge appropriate?** Semantic properties with a clear rubric; single-dimension
   scores; pairwise with position randomization; calibrated against a human-labeled sample with
   agreement reported. (Ch 24)
4. **Name three judge biases.** Position, verbosity, self-preference; mitigate with randomization,
   length normalization, a different judge model, and calibration. (Ch 24)
5. **How many cases do you need?** Enough that the confidence interval is narrower than the decision;
   fifty cases give a wide interval, hundreds for a five-point difference. (Ch 24)
6. **How do you evaluate an agent?** Task completion, tool correctness, step efficiency, safety
   violations, cost per success, trajectory assertions on replayed event logs. (Ch 25)
7. **How does evaluation enter CI?** Unit tests with fakes per commit, eval job per prompt or model
   change, thresholds as a release gate, artifacts kept per run. (Ch 25)
8. **What is online evaluation and what goes wrong?** Feedback, corrections, canary comparisons
   joined to traces; feedback is sparse and biased; use it to sample cases into the offline set. (Ch 25, 31)

### 2.8 Security (Chapters 26-27)

1. **What is prompt injection, really?** Untrusted text the model treats as instructions. An
   authorization problem: limit what the model can do rather than asking it nicely. (Ch 26)
2. **Direct versus indirect injection.** Direct comes from the user; indirect arrives through
   retrieved documents, tool results, web pages, or memory, and is the dangerous one in RAG and
   agents. (Ch 26)
3. **How can a model exfiltrate data?** A markdown image or link whose URL carries secrets; a tool
   call with an external destination; a memory write another user reads. Egress allowlists and output
   guards. (Ch 26, 27)
4. **Why is "the system prompt says not to" not a control?** Models follow instructions
   probabilistically; a control is deterministic code: tool policy, ACL filters, sandboxes, approval. (Ch 26, 27)
5. **Design the defense layers for a tool-using agent.** Input classifiers as weak signal, data and
   instruction separation, tool allowlists and argument constraints, least-privilege credentials per
   tool, sandboxing, approval for consequential actions, output guards, audit log. (Ch 27)
6. **Fail open or fail closed?** Closed for actions and permission filters; open with logging for
   low-risk read paths where availability matters; decide per control and write it down. (Ch 27)
7. **How do you handle PII?** Detect and redact before the model where the task allows; redact in
   logs and traces; tenant-scoped retention; test the redactor. (Ch 27)
8. **What is memory or index poisoning?** An attacker writes content that is later retrieved as
   trusted context; defend with provenance, write policies, source trust levels, retrieved text as
   untrusted. (Ch 26, 21)

### 2.9 Production, observability, and cost (Chapters 28-32, 34)

1. **Draw the reference architecture.** Frontend with streaming and approvals; API; model gateway;
   prompt, retrieval, and tool layers; orchestration; relational, vector, object stores; queues and
   workers; caches; observability; evaluation pipeline. Mark trust boundaries. (Ch 28)
2. **What are the SLOs for an AI service?** Availability, p95 TTFT and completion, quality metrics
   from the eval gate, cost per task, freshness; each with an alert and a degraded mode. (Ch 29)
3. **Where do you put a circuit breaker?** Around each provider adapter and each external tool;
   half-open probes; fallback on open. (Ch 29)
4. **How do you handle a provider outage?** Fallback model with its own eval, degraded mode
   (retrieval only, cached answers), queue non-interactive work, shed low-priority load. (Ch 29)
5. **Why does p99 latency collapse near saturation?** Queueing: as utilization approaches one, wait
   time grows without bound; continuous batching raises throughput but lengthens per-token time.
   Operate below saturation with admission control. (Ch 29, 34)
6. **How does AI observability differ from HTTP observability?** Semantic failures need the exact
   prompt, evidence, and tool results: spans per stage carrying prompt version, model, tokens, cost,
   cache hits, evidence IDs, policy results, eval scores. (Ch 31)
7. **Quality dropped yesterday. What do you do?** Triage by stage from traces, diff prompt, model,
   index, and data versions, replay failing cases, check the gate's last run, roll back the changed
   component. (Ch 31)
8. **What is cost per successful task?** Spend on tokens, embeddings, reranking, tools, retries,
   fallbacks, and human review divided by successes, not requests. (Ch 30)
9. **Where do you cache?** Prompt prefix, embeddings, retrieval, responses, semantic; each with a
   correctness key and a TTL tied to the change rate underneath. (Ch 30)
10. **Hosted API or self-host?** Self-host when data residency, sustained-volume cost, latency
    control, or customization requires it and you can run capacity planning and on-call; price it at
    measured utilization, not peak throughput. (Ch 7 for the decision, Ch 34 for the serving math)

### 2.10 System design prompts (Chapters 35-36)

Use the ten steps in order: requirements, architecture, models, retrieval, tools, memory, security,
evaluation, scaling, failure modes. Interviewers test whether requirements come before vendors,
permissions before autonomy, and evaluation before scale. One line per step below.

**A. Enterprise knowledge assistant** (Ch 35, Case 1). Requirements: employee Q and A over internal
docs, every claim cited from the current version, ACLs exact, abstain on thin or conflicting evidence,
p95 TTFT 2 s, read-only. Architecture: identity gateway, query service, permission-aware hybrid
retrieval, reranker, context builder, grounded generator, citation validator, streaming; indexing
asynchronous. Models: small model for query rewrite, local cross-encoder for rerank, capable model for
the answer, embedding model pinned to the index version. Retrieval: ACL pre-filter inside both
searches, BM25 50 plus dense 50, RRF, rerank to 8, parent-child chunks. Tools: none; refuse the
agent. Memory: session history only. Security: ACL before candidates exist, cache keys with tenant,
group set, index and prompt version, retrieved text as untrusted. Evaluation: gold set with required
sources and permission context, recall at 50 and at 8 before any prompt change, groundedness,
citation precision, abstention, leakage tests. Scaling: about 6,000 questions a day, 1 rps at peak,
6 in flight, about 68 USD a day at illustrative prices, 6 GB of vectors. Failure modes: recall loss
after re-embedding, ACL leak after a group rename, reranker timeout, stale cached answer; degraded
ladder from skip-rewrite to search results only.

**B. Customer support copilot with voice** (Ch 35, Case 2). Requirements: draft replies for human
agents in chat within 2 s, account lookups bound to the conversation's customer, ticket creation only
after confirmation of the exact fields, voice turns starting in about 1 s with barge-in. Architecture:
intent classifier, read-only account tool, knowledge retrieval, draft generator, agent UI; in voice,
VAD, streaming STT, orchestrator, streaming TTS, human handoff. Models: small classifier with
calibrated confidence, capable drafting model with prefix caching, streaming speech models. Retrieval:
public KB and agent macros with ACLs, smaller funnel than Case 1. Tools: `lookup_account`,
`search_tickets`, `create_ticket` with confirmation and idempotency key, `send_reply` human-only in
chat, `handoff_to_human`. Memory: conversation-scoped state; call state persisted outside the worker;
customer facts stay in the CRM. Security: customer id bound by the tool layer, never by the model;
sensitive intents routed to humans by rules as well as the model; PII redacted before logging.
Evaluation: draft acceptance, hallucinated account facts as critical failures, wrong-action rate,
barge-in success, slices by accent and codec. Scaling: 5 rps of chat at peak where prefix caching
cuts cost by about 38 percent; 150 concurrent calls where speech is two thirds of the voice bill.
Failure modes: a write triggered from a partial transcript, duplicate ticket after a timeout, eager
endpointing, worker restart mid-call.

**C. Document processing system** (Ch 35, Case 3). Requirements: extract fixed schemas from invoices
and contracts with evidence locations per field, critical fields right or flagged, a two-hour morning
batch window, region-pinned deployment. Architecture: queue, classify, native text or layout OCR,
structure-aware split, schema extraction, deterministic validation, review queue, idempotent ERP
write. Models: small model with schema mode first, capable model on low confidence, then review.
Retrieval: sections of the document itself and the supplier master by tax id; no vector index.
Tools: none for the model; connectors invoked by code after validation. Memory: per-document stage
state and the corrections store. Security: documents are untrusted; bank-detail changes always go to
review; field values never logged. Evaluation: field-level exact and normalized match weighted by
criticality, evidence-location correctness, held out by supplier. Scaling: model cost around 15 USD a
day against about 1,200 USD of review time, so the review rate is the optimization target. Failure
modes: silently wrong totals, OCR on scanned tables, fraudulent bank details, duplicate ERP postings,
backlog past the window.

**D. Coding assistant** (Ch 35, Case 4; Ch 38). Requirements: fast autocomplete, and an agent mode
that produces a patch passing tests and linters, inside scope, reviewed before merge. Architecture:
autocomplete as a prompt over a fast model; agent mode as a bounded loop with narrow tools in a
sandbox and a Definition of Done checked by the runtime. Models: small self-hostable code model for
autocomplete, capable model for planning and editing. Retrieval: symbol lookup and lexical search to
build a capped working set; no embedding index to launch. Tools: `search_code`, `read_file`,
`apply_patch`, `run_tests`, `run_linter`, `show_diff`, `install_package` with approval; no general
shell. Memory: durable task state; per-repository notes written only from verified outcomes.
Security: no network, repository-only filesystem, forbidden paths enforced in code, test results read
from exit codes, not from the model. Evaluation: historical tasks with their tests, unnecessary-edit
rate, clarification-or-stop cases. Scaling: about 0.54 USD per task with an 80 percent prefix-cache
hit rate versus 1.62 without it, so prefix stability is a budget requirement. Failure modes: plausible
code with no test run, edits outside scope, no-progress loops, an invalidated cache prefix.

**E. Research agent with citations** (Ch 36, Case A). Requirements: briefs with claim-level
citations from web and internal sources, conflicts surfaced, internal documents only within the
user's ACL, minutes not hours, predictable cost per report. Architecture: planner with a question
tree, bounded research loop, sandboxed fetcher, extractor, deduplicated evidence store, synthesizer
outside the loop, claim-citation verifier. Models: strongest for planner and synthesizer, small for
extraction, calibrated mid-size verifier. Retrieval: web search and permission-aware internal hybrid
retrieval, credibility scores, two-independent-source sufficiency rule. Tools: two read-only searches;
the fetcher is harness code, not a model tool. Memory: question tree and evidence store per report;
no cross-user research memory. Security: fetched text as data, SSRF blocklist and egress allowlist,
extracted passages validated as substrings of the source. Evaluation: coverage of gold key facts,
citation support rate, citation precision, conflict surfacing, budget-exhaustion rate. Scaling: about
0.60 USD per report with a hard cap near 1.50, three minutes per report; fan out to at most four
workers and only for independent sub-questions. Failure modes: rephrasing loops, unsupported synthesis,
citation to the wrong passage, cost blowup, ACL leak through the fetch cache.

**F. Analytics assistant over a warehouse** (Ch 36, Case B). Requirements: numbers that match the
organization's metric definitions, SQL shown, strictly read-only, tenant rows isolated, personal-data
columns never returned, p95 under 10 s. Architecture: a semantic-layer path that compiles structured
intent into SQL for known metrics, a long-tail path that retrieves schema and generates candidates,
one SQL guard and one read-only executor for both. Models: small model for intent extraction, stronger
model sampled three times for long-tail SQL. Retrieval: metric catalog and schema metadata by lexical
and metadata search, joins expanded along foreign keys. Tools: one, `run_sql`, called by the harness
after the guard. Memory: recent questions and their approved SQL for follow-ups. Security: parser-based
guard (single SELECT, table allowlist, blocked columns and functions, LIMIT), then a database role with
column grants, row-level security, timeouts, and a plan check. Evaluation: execution accuracy by result
equivalence on a frozen snapshot, never SQL string similarity; abstention on ambiguous questions.
Scaling: about 0.024 USD of model cost per question blended; raising the semantic-layer share is the
lever. Failure modes: valid SQL with the wrong metric, injection through column comments, unbounded
scans, stale cached answers after a load.

**G. Enterprise workflow automation** (Ch 36, Case C). Requirements: vendor onboarding with required
documents and approvals, no duplicate vendors, bank details matching a verified document, an audit
trail that reconstructs every decision. Architecture: a deterministic state machine with checkpoints
and approval pause states; model calls only inside extraction, approver summary, and
request-for-information steps. Models: structured extraction with vision for scans, small models for
summaries; no model decides. Retrieval: exact lookups against the vendor master, sanctions list, and
policy table. Tools: called by the engine, never by a model; ERP create with an idempotency key and
check-before-create, nightly reconciliation. Memory: the workflow state is the memory. Security:
uploaded documents untrusted, evidence checked against the document, signed approval events,
append-only audit log. Evaluation: critical-field accuracy, exception rate per step, cycle time,
historical replay when the policy table changes. Scaling: model cost under a dime per vendor, analyst
time down from about 600 to 160 hours a month. Failure modes: wrong bank account, duplicate vendor,
stuck approvals, ERP outage mid-create. The interview point is saying why this is not an agent.

**H. Internal LLM serving platform** (Ch 36, Case D) and **I. Evaluation platform** (Ch 36, Case E)
are the infrastructure prompts. For serving, lead with KV memory per token and admission by token
budget rather than request count, separate interactive and batch pools, scale on queue wait, and
treat every build change as a model release. For evaluation, lead with the data model: immutable,
versioned cases, suites, runs, metrics, and judges; replay with divergence detection; gates with
confidence intervals and hard security floors.

## 3. Numbers that make answers credible

A number with its derivation shows you have operated the thing. All figures are illustrative; the
formulas are what you memorize.

**Tokens and text.** English prose runs about four characters per token; code and non-Latin scripts
cost one and a half to three times more. A 5,000-word policy is roughly 6,500 tokens.

**Decode time.** Completion latency = TTFT + output tokens × time per output token (TPOT). Three
hundred output tokens at 20 ms each is 6 s of decode; at 50 ms, 15 s. People read at roughly 5 tokens
per second, so streaming faster than that feels instant.

**KV cache per token.** Bytes per token = 2 (keys and values) × layers × KV heads × head dimension ×
bytes per value. An illustrative 32-layer model with 8 KV heads of dimension 128 in 16-bit: 2 × 32 × 8
× 128 × 2 = 131,072 bytes, about 128 KB per token, so an 8,000-token context holds about 1 GB per
sequence. That is why grouped-query attention exists and why concurrency is memory-bound.

**Little's Law.** Concurrency = arrival rate × residence time. Twenty requests per second with a 6 s
mean completion means 120 in flight, each holding its KV cache. Halve latency and you halve memory for
the same load.

**Retrieval.** Recall at k = relevant in the top k / all relevant; hit rate = fraction of queries with
at least one relevant in the top k; MRR = mean of 1 / rank of the first relevant. Reciprocal rank
fusion scores a document as the sum over lists of 1 / (60 + rank). A funnel of 50 candidates reranked
to 5 is a common operating point: first stage buys recall at 50, reranker buys precision at 5.

**Cost per successful task.** (Input tokens × input price + output tokens × output price) × (1 + retry
and fallback rate) + embeddings + reranking + tools + human review share, divided by success rate. One
unit per answer at 90 percent success is 1.11 per success; at 60 percent, 1.67. Quality improvements
often pay for themselves this way.

**Latency budget for a 2 s p95 TTFT.** Auth 50 ms, query embedding 100, hybrid retrieval 150, rerank
300, packing 50, model TTFT 800, buffer 550. Rerank and TTFT are the two you negotiate with.

**Evaluation sample size.** A proportion estimated on 50 cases has a 95 percent interval of roughly
plus or minus 13 points; on 400 cases, about plus or minus 5. Detecting a five-point difference between
two systems takes hundreds of paired cases. "I would not ship on 30 examples" is a strong sentence.

**Saturation.** Queueing delay grows roughly as utilization / (1 − utilization). At 80 percent
utilization waits are four times the service time; at 95 percent, nineteen times. Operate below the
knee and shed load above it.

## 4. Behavioral signals interviewers score

**Quantified trade-offs.** Not "RAG is better" but "RAG adds a few hundred milliseconds of retrieval
and an indexing pipeline to operate, and buys fresh facts, citations, and permission filtering; I
choose it when knowledge changes faster than model releases or when citations are required." Every
comparison names dimensions and gives at least one number.

**Architectural restraint.** You reach for the simplest rung that meets the requirement and say what
measurement would justify the next one. "This does not need an agent" is a strong sentence when the
reasoning is explicit.

**Evidence discipline.** You distinguish what you measured, what you read, and what a vendor claimed.
Asked which model or framework is better, you name the deciding dimensions and propose the experiment,
not a winner.

**Model proposes, code authorizes.** Permissions, validation, budgets, and side effects live in code
outside the model. Candidates who put them in the prompt fail the security round.

**Stage thinking.** You decompose a failure into stages (retrieval, ranking, packing, generation,
grounding; or plan, act, observe) and name the metric for each. "The answer was wrong" becomes "recall
at 5 was fine, the reranker demoted the gold chunk, here is the span that shows it."

**Honest limits.** You say what you have not run in production and what you would measure first.

## 5. The drill method

For each chapter, three passes, aloud, timed.

1. **Ninety seconds.** The six layers, compressed: definition, mechanism in two sentences, one
   trade-off, one failure, one metric, one example with a number. Record yourself; cut filler.
2. **Five minutes.** The same topic with one artifact on a whiteboard: an equation (RRF, KV bytes,
   Little's Law), a data-flow diagram, or a state machine. Interviewers remember diagrams.
3. **Failure scenario.** "When does this break?" The trigger, how it shows in telemetry, how
   evaluation detects it, the control that prevents it. If you cannot do this pass, you memorized the
   topic.

Then one design problem per week on a whiteboard, ten steps, no product names until the architecture
is complete. Use the Try-it-first boxes in Chapters 35 and 36: 45 minutes on your own design, then
the five-minute whiteboard version aloud, then the case's follow-up questions and rubric. Your recall notes feed pass one, your project code pass two, the chapter's failure-modes
section and debugging exercises pass three. In the last four weeks of the study plan: two chapters a
day on passes one and three, one whiteboard a day, one full design case each weekend.

## 6. Common weak answers and how to fix them

| Weak answer | Why it fails | Stronger answer |
|---|---|---|
| "RAG is retrieval plus generation." | Definition only; no mechanism, cost, or failure. | Ingestion, chunking, hybrid retrieval, rerank, packing, grounded generation; chosen over fine-tuning for freshness and citations; fails on recall; measured by recall at k and groundedness separately. |
| "Inference is slow because the model is large." | Misses prefill versus decode. | Prefill is parallel and compute-bound; decode is sequential and memory-bandwidth-bound; TTFT and TPOT are different problems with different fixes. |
| "We tell the model in the system prompt not to do that." | Prompt wording is not a control. | Tool policy, ACL filters, sandboxes, egress allowlists, approval bound to arguments; the prompt helps, the code enforces. |
| "We'd use an agent so it can figure out what to do." | Autonomy without justification. | Start with a workflow where the path is known; graduate to an agent only where observations determine the path, with budgets and a Definition of Done. |
| "Model X is the best, the leaderboard says so." | No evidence discipline. | Name the dimensions; propose a gold-set evaluation on representative traffic with confidence intervals; then canary. |
| "We'd use a vector database." | Tool before requirement. | State corpus size, filters, permissions, update rate; exact search or pgvector until scale demands otherwise. |
| "We tested it and it works." | No dataset, no metric, no interval. | Gold set size, metrics, baseline, confidence interval, slices, the release gate threshold. |
| "The judge model scores the answers." | Uncalibrated judge. | Rubric, single dimension, position randomization, agreement with a human-labeled sample, known biases mitigated. |
| "Retries will handle it." | Unbounded retries amplify outages. | Retry only retryable errors, with backoff and jitter and a cap; circuit breaker; deadline propagation; fallback or degraded mode. |
| "We log the requests." | HTTP logs cannot explain semantic failures. | Spans per stage with prompt version, evidence IDs, tool results, tokens, cost, policy decisions; replay from the trace. |
| "Fine-tuning will teach it our data." | Fine-tuning teaches behavior, not facts. | Fine-tune for format, style, stable narrow behavior, or a smaller model; use retrieval for facts and permissions; compare on a frozen holdout. |

Every fix adds the same four things: a mechanism, a dimension, a measurement, and a control that lives
in code. When you catch yourself giving a one-line answer, add those four.
