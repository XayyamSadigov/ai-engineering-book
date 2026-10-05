# AI Engineer Competency Map

This is the target the book is written against. Each competency lists what a strong AI engineer must be
able to *do* (not merely define). Depth levels used throughout the book and in the coverage matrix:

- **Introductory** - can explain the idea and recognize where it applies.
- **Intermediate** - can implement a working version and reason about trade-offs.
- **Production** - can implement, test, operate, secure, and debug it under real traffic.
- **Advanced** - can choose among non-obvious variants, design for scale/adversaries, and teach it.

| # | Competency area | Must be able to | Target depth |
|---|---|---|---|
| A | LLM internals for engineers | explain tokens, tokenization cost, embeddings, attention and the context window as a resource, prefill vs decode, KV cache, sampling (temperature/top-p), structured generation, capability limits; predict *why* a system behaves as it does | Intermediate |
| B | Working with LLM APIs | build a provider-independent client: messages/roles, system prompts, streaming, structured outputs (JSON schema), tool calling, token counting, timeouts, retries with backoff, rate-limit handling, concurrency control, batching, caching, fallbacks, error taxonomy | Production |
| C | Prompt engineering | write prompts as versioned, tested contracts: instruction hierarchy, structure, few-shot selection, decomposition, reasoning patterns, templates, dynamic prompts, defensive prompting; know when prompting is not enough | Production |
| D | Context engineering | budget, select, order, compress, and label context; manage conversation state; handle long-context limits and lost-in-the-middle; design ephemeral vs persistent context | Production |
| E | Embeddings | compute and compare embeddings, choose metrics, normalize, chunk vs document embeddings, choose models/dimensionality, evaluate embedding quality, use embeddings for clustering, dedup, classification, routing, anomaly detection | Production |
| F | Vector search and storage | choose between brute force, pgvector, and dedicated vector DBs; understand HNSW/IVF trade-offs, metadata filtering, namespaces/collections, indexing strategies, scaling; know when a vector DB is unnecessary | Production |
| G | RAG ingestion and chunking | parse PDFs/HTML/Office/tables/OCR, clean, normalize, attach metadata and identity; fixed/recursive/semantic/document-aware/parent-child chunking; pick chunk sizes by evaluation | Production |
| H | Retrieval | dense, BM25, hybrid with RRF, filters, multi-query, query rewriting/expansion, reranking with cross-encoders, contextual retrieval, HyDE, parent-document retrieval | Production |
| I | Grounded generation | evidence packing, citations mapped to source IDs, abstention on missing evidence, hallucination reduction, structured answer schemas, citation validation | Production |
| J | RAG evaluation | build gold sets with required sources; recall@k, precision@k, MRR, nDCG; context relevance; faithfulness; answer relevance; stage-wise failure isolation | Production |
| K | Production RAG | indexing pipelines, incremental updates and deletions, ACL-aware retrieval, multi-tenancy, caching with correct keys, observability, cost and latency budgets, scaling | Production |
| L | Tool calling | design schemas, validate arguments, classify side effects, idempotency keys, retries, permissions, sandboxing, human approval | Production |
| M | Workflows vs agents | distinguish deterministic workflow, probabilistic workflow, LLM-enhanced app, agent, multi-agent; apply a decision framework; implement orchestration graphs | Production |
| N | Agent loop | implement a bounded loop with typed state, termination conditions, budgets, event log, replay; reasoning/planning/observation handling | Production |
| O | Agent architectures | implement and compare ReAct, router, planner-executor, supervisor/worker, reflection, evaluator-optimizer, parallel fan-out, sequential, hierarchical; know their failure modes | Intermediate-Production |
| P | Memory | implement short-term, conversation, long-term semantic/episodic/profile memory with summarization, retrieval, consolidation, expiration, privacy; know when memory harms | Production |
| Q | Multi-agent systems | justify, design, and bound multi-agent systems; message contracts; shared state; cost accounting; debugging | Intermediate |
| R | MCP and tool ecosystems | explain host/client/server roles, tools/resources/prompts, discovery vs authorization, transport, security; build a small server and client | Intermediate |
| S | Evaluation | build golden and synthetic datasets, deterministic metrics, LLM-as-judge with calibration, pairwise, rubrics, human review; evaluate prompts/RAG/agents/extraction/classification/summarization/tool use; statistics; CI integration; release gates | Production-Advanced |
| T | Security and guardrails | threat-model prompt injection (direct/indirect), jailbreaks, malicious documents, tool abuse, exfiltration, secrets, insecure output handling, excessive agency; implement input/output/tool guardrails, PII handling, tenant isolation, moderation | Production |
| U | Structured output and extraction | schema design, constrained generation, validation, repair/retry strategies, deterministic post-processing, extraction and classification pipelines | Production |
| V | Model selection and routing | choose by capability, reasoning, latency, price, context, structured-output quality, tool use, multimodality, reliability; implement routing and cascades; use small models deliberately | Production |
| W | Fine-tuning | decide when (not) to fine-tune; SFT; LoRA/PEFT at engineering level; dataset construction; evaluation; deployment | Intermediate |
| X | AI application architecture | lay out frontend/backend/model gateway/prompt layer/retrieval/tools/orchestration/persistence/queues/caches/vector and relational stores/object storage/observability/eval pipelines | Production |
| Y | Reliability and scalability | retries, backoff, fallbacks, circuit breakers, malformed responses, provider outages, partial failures; queues, async workers, rate limits, load shedding | Production |
| Z | Performance and cost | latency budgets, token budgets, caching layers, streaming, parallelization; cost model per successful task; monitoring and routing for cost | Production |
| AA | Observability | traces/spans for prompts, responses, tools, retrieval, tokens, latency, cost, eval results, agent trajectories; debugging quality degradation | Production |
| AB | Engineering practices | clean architecture, modularity, provider abstraction, testability with fakes, versioning of prompts/models/indexes, configuration, feature flags, experiments, CI/CD | Production |
| AC | Frameworks | understand LangChain, LangGraph, LlamaIndex, DSPy, agent SDKs as implementations of primitives already learned; evaluate lock-in | Intermediate |
| AD | Inference and serving basics | TTFT/TPOT/goodput, KV-cache memory, continuous batching, quantization trade-offs, capacity planning for self-hosted models | Intermediate |
| AE | System design | run the design method on knowledge assistant, support copilot, document processing, coding assistant, research agent, analytics assistant, workflow automation | Advanced |
| AF | Professional judgment | evidence discipline for vendor claims, incremental rollout, architectural restraint, interview-grade explanations | Intermediate |

## Mental models reinforced throughout (section 27 of the brief)

1. LLM output is probabilistic; design for distributions, not single answers.
2. Context is a limited engineering resource; more context is not better context.
3. Retrieval quality usually dominates generation quality.
4. Evaluate before optimizing; a system without evaluation is a demo.
5. Agents add nondeterminism and cost; prefer deterministic workflows where the path is known.
6. Every external tool widens the security boundary; the model proposes, code authorizes.
7. Reliability is engineered around the model, not expected from it.
8. Model quality alone does not determine application quality.
9. Observe at the prompt/retrieval/tool level, not just HTTP.
10. Production AI is primarily a systems-engineering problem.
