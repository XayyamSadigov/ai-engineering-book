# Proposed Table of Contents and Chapter Briefs

Each brief lists: scope, must-cover items (from the competency map), source modules to mine, the
project attached (if any), and length target. Chapter files live in `book/chapters/` with the file
names given here. "Source" references are to module text files in the analysis folder
(`analysis/source-extracted-text.txt`, page numbers from the PDF).

Length targets: conceptual chapters 5,000-7,000 words; implementation chapters 7,000-10,000 words
including code. Every chapter ends with Exercises (no answers) and Key takeaways; solutions go to
`book/solutions/chNN-solutions.md`.

---

## PART I — AI Engineering Foundations

### Ch 1 — What AI Engineering Is (`01-what-ai-engineering-is.md`)
Scope: define the discipline as systems engineering around probabilistic models. The five-layer stack
(model, context, action, control, operations). The decision ladder prompt -> retrieval -> tools ->
fine-tuning -> agent and the principle of architectural restraint. Request lineage (what you must be
able to answer about any production request). The ten mental models. Evidence discipline for vendor
claims. Introduce the running example (Northwind Assist) and the roadmap of the book.
Source: M0, Appendix A playbook, M17 evidence hierarchy, Primer "how to use math". Length: 5-6k.

### Ch 2 — How LLMs Work, for Engineers (`02-how-llms-work-for-engineers.md`)
Scope: internals only as far as they explain behavior. Tokens and tokenization (BPE, cost, multilingual
and code penalties, counting), embeddings inside the model vs retrieval embeddings, the transformer at
block-diagram level, attention as "every token can look at every earlier token" and its quadratic cost,
context window as working memory, prefill vs decode, KV cache (formula and one worked number), why
decode is sequential, sampling (logits -> softmax -> temperature -> top-k/top-p), greedy vs sampling,
stop conditions, structured/constrained generation, lost-in-the-middle, model types (SLM, general,
reasoning/test-time compute, decision-style models), capability limits (no ground truth, no state,
knowledge cutoff, arithmetic, counting, instruction competition). Include a tokenizer cost experiment
and a sampling experiment in code. Source: Primer, M3 (tokenization, embeddings, attention, logits),
M4, M5 (one paragraph), M6. Length: 7-8k.

### Ch 3 — Working with LLM APIs (`03-working-with-llm-apis.md`)
Scope: the shared `aie_core` library is built here. Messages and roles, system prompts, the request/
response shape across providers, provider abstraction (`LLMClient` protocol, OpenAI-compatible and
Anthropic adapters, `FakeLLM`), streaming (SSE, deltas, partial JSON), structured outputs (JSON schema
mode), tool calling protocol mechanics (tool specs, tool_calls, tool results), token counting and
usage, error taxonomy (4xx vs 5xx vs timeouts vs malformed), retries with exponential backoff and
jitter, timeouts and deadline propagation, rate limits (token bucket), concurrency (semaphores, async),
batching (batch APIs and client-side batching), caching (exact-match response cache, provider prompt
caching), model fallbacks, cost accounting. Full `ModelGateway` implementation with tests.
Source: M8 prompt caching, M4 streaming, M16 API design/reliability. Project: `projects/aie_core/`.
Length: 9-10k.

## PART II — LLM Application Development

### Ch 4 — Prompt Engineering as Engineering (`04-prompt-engineering.md`)
Scope: prompt as versioned interface contract; instruction hierarchy (system > developer > user >
data); prompt anatomy (role, task, constraints, evidence, output schema, failure behavior); few-shot
example selection and its costs; decomposition and chaining; reasoning patterns (plan-execute-verify,
structured intermediate artifacts, when to request reasoning); templates and dynamic prompts
(Jinja-like rendering with escaping); defensive prompting and data/instruction labeling (pointer to
Ch 26); prompt registry with versions; prompt testing (golden cases, deterministic assertions, judge
rubric) and regression in CI; when prompting is insufficient (knowledge, actions, stable behavior,
scale). Implementation: `PromptRegistry`, templates, a prompt regression test suite.
Source: M8. Length: 7-8k.

### Ch 5 — Context Engineering (`05-context-engineering.md`)
Scope: context window as an engineering budget; what belongs in context and what does not; the
context pipeline (identify -> fetch -> filter -> dedupe -> compress -> order -> label -> attribute);
budget allocation (reserve output, stable prefix, state, evidence); ordering and lost-in-the-middle;
compression and compaction (summaries vs structured state, what must never be lossy); conversation
state and history management; ephemeral vs persistent context; memory pointers (Ch 21); cache-friendly
layout (stable prefix first); measuring context effectiveness (attribution, ablation). Implementation:
`ContextBuilder` with token budget, priority ordering, compaction, source labeling; tests.
Source: M8 context engineering/compaction, M4 lost-in-the-middle. Length: 7-8k.

### Ch 6 — Structured Output and Extraction (`06-structured-output-and-extraction.md`)
Scope: why structured generation; schema design (flat vs nested, enums, optional fields, evidence
fields, confidence); provider JSON-schema modes vs tool-calling-as-schema vs constrained decoding vs
prompt+parse; validation with pydantic; repair strategies (re-ask with error, partial parse, fallback
model); deterministic post-processing (normalization, reference checks, business rules); extraction
pipelines (classify -> extract -> validate -> route to human); classification with calibrated
thresholds and abstention; entity extraction; batch extraction economics; evaluation per field
(pointer to Ch 25). **Project 1: LLM-powered structured extraction API** (FastAPI service over
`aie_core`, invoice/ticket extraction, validation, repair loop, human-review queue, tests, Dockerfile).
Source: M8 structured outputs, CS5 document extraction, M4 constrained decoding. Length: 9-10k.

### Ch 7 — Model Selection and Routing (`07-model-selection-and-routing.md`)
Scope: selection axes (capability, reasoning, latency, price, context, structured-output quality, tool
use, multimodality, reliability/availability, data residency); evaluation-driven selection procedure;
small vs large models; reasoning effort as a knob; multimodal inputs; routing (rules, classifiers,
embedding similarity, confidence cascades), routing error cost, fallbacks that change assumptions
(context length, tool support); vendor neutrality and model pinning; a `Router` implementation and
cascade evaluation. Source: M6, M16 routing, Recipe 13. Length: 6-7k.

## PART III — Embeddings and Retrieval

### Ch 8 — Embeddings (`08-embeddings.md`)
Scope: what embeddings are and how they are trained (contrastive, one paragraph); vector spaces and
geometry; similarity metrics (cosine, dot, Euclidean) and normalization; dimensionality and Matryoshka
truncation; chunk vs document embeddings; query vs passage asymmetry and instructions; model choice
(hosted vs local, multilingual, domain); embedding quality evaluation (retrieval benchmarks on your
data, nearest-neighbor sanity checks); versioning (re-embed on model change); batching and caching;
use cases beyond RAG: semantic dedup, clustering/topic discovery, classification via kNN/centroids,
intent routing, anomaly detection, recommendation. Implementation: `EmbeddingClient` adapters, a
`VectorMath` module, and worked examples for each use case; tests with `FakeEmbeddings`.
Source: M1 contrastive learning, M3 embeddings, M9 embeddings/similarity, Recipe 4. Length: 7-8k.

### Ch 9 — Vector Search and Vector Databases (`09-vector-search-and-vector-databases.md`)
Scope: exact search and when it is enough (NumPy up to ~1M vectors); ANN (HNSW, IVF, PQ) intuition and
tuning (recall vs latency vs memory); metadata filtering (pre/post filtering problems); namespaces,
collections, tenants; indexing strategies and rebuilds; PostgreSQL + pgvector (schema, HNSW index,
filters, hybrid with tsvector) vs dedicated vector DBs (comparison table, decision matrix); scaling and
operations (replication, reindexing, deletion, consistency); when a vector DB is unnecessary.
**Project 2: Semantic search system** (ingest a document set, embed, pgvector store with a NumPy
fallback, filtered search API, evaluation with recall@k, CLI + FastAPI). Source: M9 vector DBs/ANN/
HNSW/vectorless. Length: 8-9k.

## PART IV — Production RAG

### Ch 10 — RAG Fundamentals (`10-rag-fundamentals.md`)
Scope: why RAG exists (fresh knowledge, citations, permissions, cost vs long context vs fine-tuning);
the minimal pipeline (ingest -> chunk -> embed -> retrieve -> pack -> generate) built in ~150 lines over
`aie_core`; RAG as two systems; naive RAG failure modes catalogue (wrong chunk, missing evidence,
distractors, stale index, no abstention, hallucinated citations, permission leak); the stage model used
in Chapters 11-15; RAG vs long context; what "production" adds. Source: M0, M9 synthesis, Recipe 5/15.
Length: 5-6k.

### Ch 11 — Ingestion and Chunking (`11-ingestion-and-chunking.md`)
Scope: document model (id, version, source, ACL, metadata, structure); parsing PDFs (text layer vs
OCR, layout, tables), HTML, Markdown, Office, transcripts, code; cleaning and normalization; metadata
extraction; structured vs unstructured sources; deduplication; chunking strategies implemented:
fixed-size, recursive character/token, sentence/paragraph, document-aware (headings, code blocks,
tables as units), semantic (embedding-shift), parent-child; overlap; chunk-size trade-offs and how to
evaluate them; tables and code special handling; incremental ingestion (hashes, versions) pointer to
Ch 15. Implementation: parsers, `Chunker` strategies with tests. Source: M9 chunking, Deep Practice
ingestion. Length: 8-9k.

### Ch 12 — Retrieval Engineering (`12-retrieval-engineering.md`)
Scope: dense retrieval; lexical retrieval and BM25 (implemented from scratch, then Postgres tsvector);
hybrid retrieval with Reciprocal Rank Fusion and weighted fusion; metadata filters; multi-query
retrieval; query rewriting (conversation-aware), expansion, decomposition; HyDE (when it helps/hurts);
reranking with cross-encoders (local model) and LLM rerankers; contextual retrieval (chunk
contextualization); parent-document retrieval; retrieval fusion; candidate funnel sizing (k values);
latency budgeting across stages. Implementation: `BM25Index`, `HybridRetriever`, `Reranker`,
`QueryTransformer` with tests. Source: M9 lessons 2-8, RRF/HNSW notes. Length: 9-10k.

### Ch 13 — Grounded Generation and Citations (`13-grounded-generation-and-citations.md`)
Scope: evidence packing (dedupe, boundaries, ordering, budget); grounded answer contract (data vs
instructions, cite-or-abstain); structured answer schema (claims, citations, confidence, missing
evidence); citation validation (IDs exist, spans support claims); abstention and escalation policies;
handling conflicting or stale evidence; hallucination reduction techniques (quote-then-answer,
claim-level verification, self-consistency with cost); streaming grounded answers safely; answer
construction for UIs. Implementation: `EvidencePacker`, `GroundedGenerator`, `CitationValidator`.
Source: M9 generation/grounding contract, M13. Length: 6-7k.

### Ch 14 — RAG Evaluation (`14-rag-evaluation.md`)
Scope: building the retrieval gold set (required/acceptable sources, rubric, permission context, tags);
synthetic question generation from chunks and its biases; retrieval metrics (hit rate, recall@k,
precision@k, MRR, nDCG) implemented; context relevance; answer faithfulness and relevance judges with
calibration; citation precision/recall; abstention correctness; stage isolation report (which stage
lost the evidence); slice analysis; regression runs and comparing configurations. Implementation:
`rag_eval` module with CLI producing a report. Source: M9 metrics, Workshop G, M13. Length: 7-8k.

### Ch 15 — Production RAG (`15-production-rag.md`)
Scope: indexing pipelines (workers, queues, idempotency, versions); incremental updates and document
deletion (propagation to index, cache, derived data); permissions and ACL-aware retrieval (filter at
retrieval, never after generation); multi-tenancy (namespaces vs shared index with tenant filters);
caching layers (embedding, retrieval, semantic cache with correct keys and TTLs); observability of
each stage; cost and latency budgets; scalability (parallel retrieval, replica reads); freshness
SLOs; failure handling and degraded modes. **Project 3: Production RAG assistant** (full service:
ingestion worker, pgvector + BM25 hybrid, reranking, ACL, citations, evaluation suite, tracing, Docker
Compose, tests). Source: M9 Deep Practice, CS1, M16. Length: 9-10k.

## PART V — Tools and Workflows

### Ch 16 — Tool Calling (`16-tool-calling.md`)
Scope: what a tool is to a model; tool schema design (narrow, enums, descriptions, examples); tool
selection quality and description engineering; argument validation outside the model; side-effect
classes (read / reversible write / irreversible / external); permissions and least privilege;
idempotency keys and duplicate suppression; retries and error contracts (machine-readable errors);
timeouts and result truncation; sandboxing code execution and file/network scope; human approval
binding to concrete arguments; auditing. Implementation: `ToolRegistry`, `ToolPolicy`,
`IdempotencyStore`, `SandboxRunner`, tool-calling loop over `aie_core`. **Project 4: AI application
with tools** (support assistant that looks up accounts, drafts and sends replies with approval, creates
tickets; policy tests; injection test). Source: M10 function calling/permissions, Recipe 7, M14 tool
security. Length: 9-10k.

### Ch 17 — AI Workflows and Orchestration (`17-ai-workflows-and-orchestration.md`)
Scope: the spectrum deterministic workflow / probabilistic workflow / LLM-enhanced application / agent /
multi-agent; decision framework (table and flowchart); orchestration patterns (sequence, branch,
parallel fan-out/fan-in, map-reduce, retry, fallback, human approval); workflows as explicit state
machines/graphs; typed state between steps; checkpoints; error handling; a plain-Python workflow
engine (~200 lines) and the same workflow in graph form; cost and latency of chains; when to graduate
to an agent. Source: M0 decision ladder, M8 chaining, M10 orchestration, M11 graph engineering.
Length: 7-8k.

### Ch 18 — MCP and Tool Ecosystems (`18-mcp-and-tool-ecosystems.md`)
Scope: the integration problem MCP solves; host, client, server roles; tools, resources, prompts;
discovery vs authorization; transports (stdio, HTTP) and statelessness trends; capability negotiation;
versioning; security (untrusted tool descriptions, confused deputy, credential scoping, egress);
remote tool servers and gateways; MCP vs plain function calling vs plugins vs Agent Skills; building a
small MCP server and client in Python (official SDK, with fallback to a minimal JSON-RPC
implementation); architectural implications for enterprises. Freshness note on protocol revisions.
Source: M0 MCP, M10 MCP/skills, Workshop I. Length: 6-7k.

## PART VI — Agents

### Ch 19 — The Agent Loop (`19-the-agent-loop.md`)
Scope: what an agent actually is (model + tools + state + loop); the loop as a state machine; typed
state and event log (event sourcing); reasoning and planning inside the loop; observation handling and
tool-result truncation; termination conditions (success validated, max steps, budget, time, repeated
state, no progress, approval needed); budgets; error classes (transient, validation, semantic,
impossible); replay for debugging and evaluation; Definition of Done; when not to use an agent.
Implementation: `AgentRuntime` with events, budgets, policies, replay; tests with `FakeLLM`.
Source: M10 (loop, ReAct, stop conditions, Deep Practice), M11 loop/harness, Recipe 8/16. Length: 9-10k.

### Ch 20 — Agent Architectures (`20-agent-architectures.md`)
Scope: ReAct; router; planner-executor (and replanning); supervisor/worker; reflection; evaluator-
optimizer; parallel agents; sequential workflows; hierarchical agents; for each: structure, when to
use, advantages, failure modes, cost profile, evaluation; implementations over `AgentRuntime`.
**Project 5: Agentic workflow** (incident-research agent: plans, searches runbooks and logs via tools,
drafts a report, evaluator-optimizer loop with Definition of Done; trajectory tests). Source: M10
lessons 4-6, 12-13, M11. Length: 9-10k.

### Ch 21 — Memory Systems (`21-memory-systems.md`)
Scope: memory taxonomy (working, conversation, episodic, semantic, procedural, user profile); storage
choices; summarization and compaction (pointer to Ch 5); memory retrieval (recency, relevance, salience);
consolidation and deduplication; write policies, provenance, confidence; expiration and TTL; privacy,
deletion, tenant scope; memory poisoning; when memory becomes harmful; evaluation of memory.
Implementation: `ConversationMemory`, `EpisodicStore`, `SemanticMemory`, `UserProfileMemory` with
tests. Source: M10 memory (x3), M8 compaction. Length: 7-8k.

### Ch 22 — Multi-Agent Systems (`22-multi-agent-systems.md`)
Scope: when multiple agents are justified (parallelism, specialization, isolation, independent
verification, permission domains); patterns (supervisor-worker, debate/critique, map-reduce, blackboard,
pipeline); message contracts; shared state vs event logs; budgets at parent and child; trace
propagation; failure modes (runaway spawning, contradictory outputs, context loss, cost blow-up);
evaluation. **Project 6: Multi-agent system where justified** (parallel research workers + verifier +
supervisor with budgets; comparison against single-agent baseline). Source: M10 multi-agent/subagents/
supervisor. Length: 7-8k.

### Ch 23 — Frameworks (`23-frameworks.md`)
Scope: primitive-first mapping: for each framework concept show the underlying primitive built in
earlier chapters, then how the framework implements it. LangChain (runnables, prompts, output parsers,
retrievers), LangGraph (state graph, checkpoints, interrupts), LlamaIndex (indices, query engines,
node parsers), DSPy (signatures, modules, optimizers), provider agent SDKs and MCP SDKs; selection
criteria (transparency, persistence, streaming, tracing, testability, lock-in); how to keep domain
logic framework-independent; migration strategy. Source: M11 lessons 5-6, selection criteria.
Length: 6-7k.

## PART VII — Evaluation

### Ch 24 — Evaluation Fundamentals (`24-evaluation-fundamentals.md`)
Scope: "no evaluation, no engineering"; failure taxonomy first; dataset types (golden, synthetic,
production-sampled, adversarial, regression) and their construction; dev vs frozen holdout; deterministic
evaluation; model-based evaluation and LLM-as-judge (rubrics, single-dimension scoring, pairwise with
position randomization, biases, calibration against humans, agreement metrics); human evaluation
design; statistics (sample size, bootstrap CIs, significance, per-case deltas, slices); metrics
vocabulary (precision, recall, F1, groundedness, faithfulness, correctness, relevance, task completion,
tool correctness); classification metrics inheritance (thresholds, calibration). Implementation:
`evalkit` core (case schema, runner, judges, stats). Source: M13, M1 metrics/thresholds/calibration/
leakage. Length: 8-9k.

### Ch 25 — Evaluating AI Systems in Practice (`25-evaluating-ai-systems-in-practice.md`)
Scope: evaluating prompts, RAG (pointer to Ch 14), agents (trajectory assertions, tool correctness,
task completion, efficiency, safety violations, replay), extraction (field-level P/R, evidence
location), classification (macro-F1, per-class recall, calibration), summarization (coverage,
faithfulness, compression ratio), tool usage; synthetic data generation and its validation; CI/CD
integration (pytest markers, eval job, thresholds, release gate, artifacts); online evaluation (feedback,
corrections, canary comparisons); the evaluation platform design (CS7). Implementation: task-specific
evaluators, GitHub Actions / generic CI workflow, release gate script. Source: M13 Deep Practice,
Workshop K, CS6, CS7, Recipe 14. Length: 8-9k.

## PART VIII — Security and Guardrails

### Ch 26 — Threat Modeling and Prompt Injection (`26-threat-modeling-and-prompt-injection.md`)
Scope: assets, actors, trust boundaries for AI systems; threat catalogue: direct and indirect prompt
injection, jailbreaks, malicious documents, tool abuse, data exfiltration (including via URLs/markdown
images), secrets exposure, insecure output handling (XSS/SQL from model output), excessive agency,
memory poisoning, index poisoning, supply chain (prompts, tools, skills, MCP servers, models); worked
threat models for a RAG assistant and a tool-using agent; real-world incident patterns; red-team plan;
why prompt wording is not a control. Source: M14, M10 permission boundaries, Recipe 9. Length: 7-8k.

### Ch 27 — Guardrails and Safe Tooling (`27-guardrails-and-safe-tooling.md`)
Scope: layered controls; input guardrails (classifiers, allow/deny, injection heuristics as weak
signal); data/instruction separation in prompts; output guardrails (schema, sensitive-data detection,
URL/egress allowlists, insecure output encoding); tool policy engine (allowlists, argument constraints,
approval, rate limits); permission boundaries and tenant isolation (tests); PII detection and redaction
(before model, in logs); content moderation integration; secrets management; sandboxing; fail-closed
vs fail-open decisions; measuring guardrails (false positives, bypass rate). Implementation:
`guardrails` package with tests and an adversarial test suite for Project 3/4. Source: M14 guardrails/
sandboxing/privacy, Workshop L. Length: 7-8k.

## PART IX — Production AI Engineering

### Ch 28 — AI Application Architecture (`28-ai-application-architecture.md`)
Scope: reference architecture with Mermaid diagrams: frontend (streaming UI, approvals), backend API,
model gateway, prompt layer, retrieval layer, tool layer, orchestration, persistence (relational, vector,
object storage), queues and workers, caches, observability, evaluation pipelines; request path with
timeouts; async job model; data architecture and lineage (versions of prompts/models/indexes); transport
choices (HTTP, SSE, WebSocket); deployment topology (containers, Compose, Kubernetes-level sketch);
how the running example maps onto it. Source: M16 synthesis/Deep Practice, M12 lesson 17. Length: 6-7k.

### Ch 29 — Reliability and Scalability (`29-reliability-and-scalability.md`)
Scope: SLOs for AI services; retries with exponential backoff and jitter (what is retryable);
timeouts and deadline propagation; fallbacks (model, provider, degraded mode); circuit breakers;
bulkheads; handling malformed responses; provider outages; partial failures in chains and agents;
idempotency; concurrency control; queues and async processing; worker architecture; rate limits and
quotas per tenant; admission control and load shedding; backpressure; graceful degradation; chaos
tests. Implementation: `reliability` primitives (retry policy, circuit breaker, token bucket, worker
queue with Redis or in-memory), tests. Source: M16 reliability/async/backpressure. Length: 7-8k.

### Ch 30 — Performance and Cost Engineering (`30-performance-and-cost-engineering.md`)
Scope: latency anatomy (queue, prefill/TTFT, decode/TPOT, tools, retrieval, network); latency budgets
per stage; token budgets; caching layers (prompt prefix, embedding, retrieval, response, semantic) with
correctness keys; streaming for perceived latency; parallelization (fan-out retrieval, speculative
tool prefetch); batching; cost model per successful task (tokens, embeddings, reranking, tools, retries,
human review); model routing for cost; cost monitoring, budgets, alerts, chargeback per tenant;
back-of-envelope capacity and cost estimates. Implementation: `CostModel`, latency budget tracker,
caching decorators. Source: M16 cost model/back-of-envelope, Recipe 12/19, M12 metrics. Length: 7-8k.

### Ch 31 — Observability for AI Systems (`31-observability.md`)
Scope: why AI observability differs (semantic failures need the exact context); trace model (request ->
spans for router, retrieval, rerank, model call, tool call, validator, agent step); attributes (prompt
version, model, tokens, cost, cache hits, evidence IDs, policy results, eval scores); OpenTelemetry
wiring; logging prompts/responses with redaction and sampling; agent trajectory views; dashboards and
alerts (quality, latency, cost, error classes); joining offline eval results and online feedback to
traces; the debugging playbook when quality degrades (regression triage by stage, version diff, replay).
Implementation: `aie_core.observability` with OTel exporter and JSONL fallback; trace query examples.
Source: M13 observability, M16 observability, Recipe 15/16. Length: 7-8k.

### Ch 32 — Engineering Practices for AI Systems (`32-engineering-practices.md`)
Scope: clean architecture for AI apps (domain, application, adapters); modularity and boundaries;
provider abstraction and anti-corruption layers; testability (fakes, recorded fixtures, property tests
for parsers, contract tests for tools); versioning prompts, models, indexes, datasets, evaluators;
configuration management (settings, environments, secrets); feature flags and gradual rollout;
experiments (A/B, shadow, canary); CI/CD pipeline for AI changes (lint, unit, eval gate, canary);
documentation and ADRs; avoiding the "scripts folder" anti-pattern; code review checklist for AI code.
Source: M16 rollout/change management, M11 harness-as-code, M1 experiment design. Length: 6-7k.

### Ch 33 — Fine-Tuning for Engineers (`33-fine-tuning-for-engineers.md`)
Scope: what fine-tuning changes; when it is useful (stable behavior, format, narrow task, latency/cost
via smaller model, style) and when not (fresh facts, permissions, rapidly changing requirements);
fine-tuning vs prompting vs RAG vs tools; SFT mechanics; LoRA/QLoRA/PEFT at engineering level (parameter
count worked example); dataset construction (sourcing, cleaning, dedup, splits by entity/time, formats);
quality checks; evaluation protocol (baselines, frozen holdout, regression suite, slices); training
options (hosted fine-tuning APIs vs open-weights with PEFT); deployment (adapters, serving, versioning,
rollback); distillation; preference tuning in one section (RLHF/DPO conceptually). Implementation:
dataset builder and evaluation protocol code; a hosted-API fine-tuning walkthrough with provider-neutral
interface. Source: M7, CS6, Workshop E. Length: 6-7k.

### Ch 34 — Inference and Serving Essentials (`34-inference-and-serving-essentials.md`)
Scope: hosted APIs vs self-hosting decision; serving metrics (TTFT, TPOT, throughput, goodput,
p95/p99); prefill vs decode and memory-bandwidth-bound decode; KV cache sizing (formula, worked
numbers); continuous batching; prefix caching; quantization trade-offs; speculative decoding (one
section); parallelism (one paragraph); engines comparison table (vLLM, SGLang, TensorRT-LLM, llama.cpp/
GGUF) as options not endorsements; capacity planning with Little's Law and load testing protocol;
benchmark protocol; serving a local model behind the `aie_core` OpenAI-compatible adapter.
Source: M12, M5 (GQA paragraph), Recipe 11/12, Workshop J, CS4. Length: 6-7k.

## PART X — AI System Design

### Ch 35 — System Design Method and Cases I (`35-system-design-method-and-cases-1.md`)
Scope: the 10-step method (requirements, architecture, models, retrieval, tools, memory, security,
evaluation, scaling, failure modes) with templates; cases: enterprise knowledge assistant, customer
support copilot (chat + voice latency budget), document processing system, coding assistant. Each with
Mermaid architecture, numbers, and failure modes. Source: CS1, CS2, CS3, CS5, M16 Workshop N, M18.
Length: 8-9k.

### Ch 36 — System Design Cases II (`36-system-design-cases-2.md`)
Scope: research agent, analytics assistant (text-to-SQL with semantic layer, safety, verification),
enterprise workflow automation (approvals, idempotency, audit), plus an LLM serving platform sketch
and an evaluation platform (CS4, CS7). Same 10-step treatment. Source: CS4, CS7, M18 research
assistant example. Length: 8-9k.

## PART XI — Advanced Patterns

### Ch 37 — Advanced Retrieval Patterns (`37-advanced-retrieval-patterns.md`)
Scope: agentic RAG (iterative retrieval with budgets), GraphRAG (entity graphs, community summaries,
costs), vectorless and structured retrieval (SQL, metadata, full-text, hierarchical navigation), late
interaction (ColBERT) at concept level, multimodal documents (images, tables, charts), long-context vs
RAG hybrid strategies, recursive/long-input processing, retrieval over code. Source: M9 lessons 6,
11-13, M6 RLMs, M15 multimodal product. Length: 6-7k.

### Ch 38 — Durable and Long-Running Agents (`38-durable-and-long-running-agents.md`)
Scope: checkpointing and resumption; durable execution and idempotent side effects under at-least-
once; human-in-the-loop interrupts and approvals as first-class states; replay and counterfactual
testing; long-horizon context compaction for agents; coding agents and IDE agents as case studies of
harness engineering (sandbox, test feedback loop, diff review); computer-use agents; Agent Skills as
procedural knowledge packaging; voice and realtime agents (streaming, barge-in, latency budget).
Source: M11 (harness, checkpointing, coding agents, IDE agents), M10 skills/computer use, M16 voice
agent, CS2, CS3. Length: 7-8k.

## PART XII — Capstone

### Ch 39 — Capstone: Northwind Assist (`39-capstone.md`)
Scope: a production-grade AI system assembling the projects: API + minimal web UI with streaming and
approvals, model gateway, hybrid RAG with ACL and citations, tools with policy and approval, structured
output, conversation + profile memory, evaluation suite and release gate, observability (OTel), auth
(JWT + tenants), security tests, caching, Docker Compose deployment, CI/CD pipeline, cost monitoring.
Deliverables: architecture, code tree in `book/capstone/`, run book, acceptance tests, extension ideas.
Length: 8-10k + code.

## Appendices and companion files

- `00-learning-roadmap.md` — reading paths (full 16-week path; accelerated path; role-based paths), per-part
  prerequisites, project schedule.
- `glossary.md` — terms used in the book with chapter pointers.
- `references.md` — primary papers, specifications, and documentation (optional reading).
- `appendix-c-interview-preparation.md` — question banks and answer structures (from M18).
- `coverage-matrix.md` — competency x chapter x depth x exercise.
- `exercises/` — exercise sections per chapter; `solutions/` — answers, separate from exercises.
- `AI_ENGINEERING_BOOK.md` — the complete book in one file, assembled from the chapters.
