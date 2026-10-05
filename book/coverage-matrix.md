# Coverage Matrix

This matrix maps every AI engineering skill in the book's competency map (`analysis/02-competency-map.md`)
to the chapter that teaches it, the depth reached, and the hands-on work that exercises it. It was
built after all chapters were written and reviewed, by checking chapter text and code on disk.

Depth levels:

- **Introductory**: you can explain it and recognize where it applies.
- **Intermediate**: you can implement a working version and reason about its trade-offs.
- **Production**: you can implement, test, operate, secure, and debug it under real traffic.
- **Advanced**: you can choose among non-obvious variants, design for scale and adversaries, and teach it.

"Practical" names the project, package, or exercise ids where you build the skill. P-exercises are
in each chapter's Exercises section; their solutions are in `solutions/`.

## Foundations

| AI Engineering Skill | Covered? | Chapter | Depth | Practical Exercise |
|---|---|---|---|---|
| What AI engineering is, five-layer stack, decision ladder | Yes | 1 | Intermediate | Ch 1 P1-P4; `examples/ch01` lineage record |
| Request lineage and evidence discipline for vendor claims | Yes | 1, 2 | Intermediate | Ch 1 P2, D1-D3 |
| Tokens, tokenization, token cost by language and format | Yes | 2 | Intermediate | `examples/ch02/tokenizer_experiment.py`; Ch 2 P1 |
| Embeddings inside the model vs retrieval embeddings | Yes | 2, 8 | Intermediate | Ch 2 K-exercises; Ch 8 |
| Transformer and attention at block-diagram level | Yes | 2 | Introductory | Ch 2 K1-K6 |
| Context window as working memory; quadratic cost intuition | Yes | 2, 5 | Intermediate | Ch 5 position sweep |
| Prefill vs decode, TTFT vs TPOT | Yes | 2, 34 | Intermediate | `examples/ch34/loadtest.py` |
| KV cache sizing and its effect on concurrency | Yes | 2, 34 | Intermediate | `examples/ch02/kv_cache_calc.py`, `examples/ch34/kv_cache.py` |
| Sampling: temperature, top-k, top-p, greedy, stop sequences | Yes | 2 | Intermediate | `examples/ch02/sampling.py` |
| Structured and constrained generation | Yes | 2, 6 | Production | Project 1 |
| Model types, reasoning effort, capability limits | Yes | 2, 7 | Intermediate | Ch 7 selection harness |

## Working with LLM APIs

| AI Engineering Skill | Covered? | Chapter | Depth | Practical Exercise |
|---|---|---|---|---|
| Messages, roles, system prompts, provider request shapes | Yes | 3 | Production | `aie_core.llm` |
| Provider abstraction and fakes for offline tests | Yes | 3, 32 | Production | `aie_core` FakeLLM; Ch 32 record/replay |
| Streaming over SSE, partial JSON, cancellation | Yes | 3, 13, 28 | Production | `aie_core` streaming; Ch 13 GroundedStreamer; Project 3 SSE |
| Structured outputs: JSON schema mode, tool-as-schema, repair loop | Yes | 3, 6 | Production | `aie_core.llm.structured`; Project 1 |
| Tool-calling protocol mechanics | Yes | 3, 16 | Production | `aie_core`; `toolkit.ToolLoop` |
| Error taxonomy, retries, exponential backoff with jitter | Yes | 3, 29 | Production | `aie_core.ModelGateway`; `reliability` |
| Timeouts and deadline propagation | Yes | 3, 29 | Production | `reliability.Deadline` |
| Rate limits, token buckets, concurrency control | Yes | 3, 29 | Production | `aie_core.RateLimiter`; `reliability.AdmissionController` |
| Batching (client-side and provider batch APIs) | Yes | 3, 6, 8 | Intermediate | Project 1 batch mode; Ch 8 batching |
| Response caching and provider prompt caching, tenant cache scope | Yes | 3, 30 | Production | `aie_core` ResponseCache with `cache_scope`; Ch 30 caching |
| Token counting and cost accounting | Yes | 3, 30 | Production | `aie_core.PricingTable`; Ch 30 CostModel |
| Model fallbacks and what breaks on fallback | Yes | 3, 7, 29 | Production | ModelGateway fallbacks; Ch 7 Router |

## Prompt and Context Engineering

| AI Engineering Skill | Covered? | Chapter | Depth | Practical Exercise |
|---|---|---|---|---|
| Prompt as a versioned interface contract | Yes | 4 | Production | `examples/ch04` PromptRegistry, lock file |
| Instruction hierarchy and prompt structure | Yes | 4 | Production | Ch 4 P1-P4 |
| Few-shot example selection | Yes | 4 | Intermediate | Ch 4 `select_examples` |
| Decomposition, chaining, reasoning patterns | Yes | 4, 17 | Intermediate | Ch 17 triage pipeline |
| Templates and dynamic prompts with safe variables | Yes | 4 | Production | Ch 4 PromptTemplate (sandboxed, untrusted blocks) |
| Defensive prompting and data/instruction separation | Yes | 4, 5, 26, 27 | Production | `guardrails.wrap_untrusted` |
| Prompt versioning, canary rollout, regression testing | Yes | 4, 32 | Production | Ch 4 regression harness and PromptRollout |
| When prompting is insufficient | Yes | 1, 4, 33 | Intermediate | Ch 1 decision ladder; Ch 33 decision table |
| What belongs in context and what does not | Yes | 5 | Production | `examples/ch05` ContextBuilder |
| Context selection, filtering, prioritization, ordering | Yes | 5 | Production | ContextBuilder placement policies |
| Context compression and compaction | Yes | 5, 21, 38 | Production | ConversationState; memorykit; Ch 38 CompactingLLM |
| Long-context limits and lost-in-the-middle | Yes | 2, 5, 37 | Production | Ch 5 position experiment; Ch 37 long-context cost |
| Conversation state, ephemeral vs persistent context | Yes | 5, 21 | Production | ConversationState with versioned store |
| Context window as a budget; cache-friendly layout | Yes | 5, 30 | Production | Ch 5 layout lint; Ch 30 latency budgets |

## Structured Output and Model Selection

| AI Engineering Skill | Covered? | Chapter | Depth | Practical Exercise |
|---|---|---|---|---|
| Schema design for extraction | Yes | 6 | Production | Project 1 domain schemas |
| Validation, repair strategies, deterministic post-processing | Yes | 6 | Production | Project 1 ExtractionService |
| Classification with thresholds and abstention | Yes | 6, 24, 25 | Production | Project 1 routing; evalkit threshold sweep |
| Entity extraction and field-level evaluation | Yes | 6, 25 | Production | Project 1 eval; Ch 25 ExtractionEvaluator |
| Model selection by capability, latency, price, context, tools, modality | Yes | 7 | Production | `examples/ch07` selection harness |
| Model routing, cascades, utility-based thresholds | Yes | 7 | Production | Ch 7 Router and cascade sweep |
| When smaller models are better | Yes | 7, 33 | Intermediate | Ch 7 cascade; Ch 33 distillation case |

## Embeddings and Vector Search

| AI Engineering Skill | Covered? | Chapter | Depth | Practical Exercise |
|---|---|---|---|---|
| Embeddings, vector spaces, similarity metrics, normalization | Yes | 8 | Production | `examples/ch08` embedlab |
| Dimensionality, Matryoshka truncation, model choice | Yes | 8 | Intermediate | Ch 8 truncation report |
| Chunk vs document embeddings | Yes | 8, 11 | Intermediate | Ch 8 comparison |
| Embedding quality evaluation | Yes | 8, 14 | Production | Ch 8 evaluate_retrieval |
| Embedding versioning and re-embedding | Yes | 8, 15 | Production | Ch 8 EmbeddingSpace fingerprint; `aie_core` space fingerprint |
| Uses beyond RAG: dedup, clustering, classification, routing, anomaly, recommendation | Yes | 8 | Production | Ch 8 usecases package |
| Exact search vs ANN (HNSW, IVF, PQ) and recall measurement | Yes | 9 | Production | Project 2 ann-check, IVF sweep |
| Metadata filtering, pre vs post filter, selectivity | Yes | 9, 12 | Production | Project 2 bench-filter |
| Namespaces, collections, tenants | Yes | 9, 15 | Production | Project 2 make_namespace; Project 3 |
| PostgreSQL + pgvector vs dedicated vector DB | Yes | 9 | Production | Project 2 PgVectorStore (real pgvector not verified in this build) |
| When a vector DB is unnecessary | Yes | 9, 37 | Intermediate | Ch 37 vectorless retrieval |

## RAG

| AI Engineering Skill | Covered? | Chapter | Depth | Practical Exercise |
|---|---|---|---|---|
| Minimal RAG and naive failure modes | Yes | 10 | Intermediate | `examples/ch10` minimal_rag and failure_modes |
| Parsing PDF, HTML, Markdown, JSONL; OCR and tables | Yes | 11 | Production | `ragkit.parsers` |
| Cleaning, normalization, metadata, deduplication | Yes | 11 | Production | `ragkit.normalize` |
| Fixed, recursive, sentence, document-aware, semantic, parent-child chunking | Yes | 11 | Production | `ragkit.chunking`; chunk-size evaluation |
| BM25 and lexical retrieval | Yes | 12 | Production | `ragkit.retrieval.BM25Index` |
| Dense retrieval | Yes | 12 | Production | `ragkit.retrieval.DenseRetriever` |
| Hybrid retrieval and RRF fusion | Yes | 12 | Production | HybridRetriever; compare_retrievers |
| Query rewriting, multi-query, decomposition, HyDE | Yes | 12 | Production | `ragkit.retrieval.query` |
| Reranking with cross-encoders and LLM rerankers | Yes | 12 | Production | `ragkit.retrieval.rerank` |
| Contextual retrieval and parent-document retrieval | Yes | 12 | Production | ContextualEnricher; ParentDocumentRetriever |
| Diversity (MMR) | Yes | 5, 8, 12 | Intermediate | MMRDiversifier |
| Evidence packing, grounding, citations, abstention | Yes | 13 | Production | `ragkit.generation` GroundedQA |
| Handling conflicting versions and hallucination reduction | Yes | 13, 15 | Production | Packer conflict notes; Project 3 authority layer |
| Retrieval metrics: hit@k, recall@k, precision@k, MRR, nDCG | Yes | 14 | Production | `ragkit.eval.rag_metrics` |
| Context relevance, faithfulness, answer relevance judges | Yes | 14 | Production | `ragkit.eval.rag_judges` |
| Stage isolation of RAG failures | Yes | 14 | Production | `ragkit.eval.stage_isolation` |
| Indexing pipelines, incremental updates, deletion | Yes | 15 | Production | Project 3 ingestion worker |
| ACL-aware retrieval and multi-tenancy | Yes | 15, 27 | Production | Project 3 ACL tests |
| RAG caching, observability, cost, latency, scaling | Yes | 15, 30, 31 | Production | Project 3 |
| Agentic RAG, GraphRAG, late interaction, multimodal documents | Yes | 37 | Advanced | `examples/ch37` |

## Tools, Workflows, MCP

| AI Engineering Skill | Covered? | Chapter | Depth | Practical Exercise |
|---|---|---|---|---|
| Tool schema design and tool selection | Yes | 16 | Production | `toolkit.ToolRegistry` |
| Argument validation, permissions, policy engine | Yes | 16, 27 | Production | `toolkit.PolicyEngine`; `guardrails.ToolPolicyCheck` |
| Idempotency keys and duplicate suppression | Yes | 16, 38 | Production | `toolkit.IdempotencyStore`; Ch 38 ReconcilingTool |
| Retries and machine-readable tool errors | Yes | 16 | Production | `toolkit.ToolExecutor` |
| Sandboxing code execution | Yes | 16, 26, 38 | Production | `toolkit.SandboxRunner` (process sandbox) |
| Human approval bound to exact arguments | Yes | 16, 38 | Production | `toolkit.ApprovalManager`; Project 4 |
| Deterministic vs probabilistic workflow vs LLM-enhanced app vs agent vs multi-agent | Yes | 17 | Production | Ch 17 decision framework, Northwind placement table |
| Orchestration patterns and state-machine workflows with checkpoints | Yes | 17 | Production | `examples/ch17` workflow engine |
| MCP host, client, server; tools, resources, prompts | Yes | 18 | Intermediate | `examples/ch18` JSON-RPC server and client |
| MCP discovery vs authorization, tool poisoning, remote servers | Yes | 18, 26 | Intermediate | Ch 18 poisoned-server test |

## Agents

| AI Engineering Skill | Covered? | Chapter | Depth | Practical Exercise |
|---|---|---|---|---|
| What an agent is; the loop as a state machine | Yes | 19 | Production | `agentkit.AgentRuntime` |
| Typed state, event sourcing, budgets, termination conditions | Yes | 19 | Production | agentkit events and Budget |
| Definition of Done and verification | Yes | 19, 20 | Production | agentkit DefinitionOfDone |
| Replay and counterfactual testing | Yes | 19, 25, 38 | Production | `agentkit.replay`; Ch 25 agentkit_replay_target |
| When not to use an agent | Yes | 1, 17, 19 | Production | Ch 17 P-exercises |
| ReAct, router, planner-executor, supervisor/worker | Yes | 20 | Production | `examples/ch20/patterns` |
| Reflection, evaluator-optimizer, parallel, sequential, hierarchical | Yes | 20 | Production | `examples/ch20/patterns`; Project 5 |
| Memory: short-term, conversation, long-term, semantic, episodic, profile | Yes | 21 | Production | `memorykit` |
| Memory summarization, retrieval, consolidation, expiration, privacy | Yes | 21 | Production | memorykit WritePolicy, tombstones |
| When memory becomes harmful; memory poisoning | Yes | 21, 26 | Production | memorykit poisoning test |
| Multi-agent justification, contracts, budgets, trace propagation | Yes | 22 | Production | Project 6 (with single-agent baseline) |
| Frameworks (LangChain, LangGraph, LlamaIndex, DSPy, agent SDKs) mapped to primitives | Yes | 23 | Intermediate | `examples/ch23` |
| Durable execution, checkpoints, long waits, crash reconciliation | Yes | 38 | Advanced | `examples/ch38` DurableRunner |
| Coding agents, computer-use agents, Agent Skills, voice agents | Yes | 35, 38 | Intermediate | Ch 38 coding harness and skills loader |

## Evaluation

| AI Engineering Skill | Covered? | Chapter | Depth | Practical Exercise |
|---|---|---|---|---|
| Failure taxonomy first | Yes | 24 | Production | Ch 24 P-exercises |
| Golden, synthetic, production-sampled, adversarial, regression datasets | Yes | 24, 25 | Production | `evalkit.Dataset`; Ch 25 SyntheticGenerator |
| Dev vs frozen holdout, leakage checks | Yes | 24 | Production | evalkit split and check_leakage |
| Deterministic metrics | Yes | 24 | Production | `evalkit.metrics` |
| LLM-as-a-judge, rubrics, pairwise, bias, calibration vs humans | Yes | 24 | Production | `evalkit.judges` |
| Human evaluation design | Yes | 24 | Intermediate | Ch 24 E-exercises |
| Statistics: bootstrap CIs, paired and cluster bootstrap, slices | Yes | 24 | Advanced | `evalkit.stats` |
| Evaluating prompts | Yes | 4, 25 | Production | Ch 4 regression harness |
| Evaluating RAG | Yes | 14 | Production | `ragkit.eval` |
| Evaluating agents: trajectories, tool correctness, task completion | Yes | 25 | Production | Ch 25 TrajectoryEvaluator over agentkit logs |
| Evaluating extraction, classification, summarization, tool usage | Yes | 25 | Production | `examples/ch25/taskevals` |
| CI/CD integration and release gates | Yes | 25, 32 | Production | Ch 25 release_gate.py and CI files; Ch 32 pipeline |
| Online evaluation, feedback joining, canary decisions | Yes | 25, 31, 32 | Production | Ch 25 CanaryMonitor; Ch 31 join_signals |

## Security and Guardrails

| AI Engineering Skill | Covered? | Chapter | Depth | Practical Exercise |
|---|---|---|---|---|
| Threat modeling for AI systems | Yes | 26 | Production | `examples/ch26/threat_model.py` |
| Direct and indirect prompt injection, malicious documents | Yes | 26, 27 | Production | `examples/ch26/attack_corpus.py`; guardrails red team |
| Jailbreaks and why system prompts are not a boundary | Yes | 26 | Intermediate | Ch 26 K/E exercises |
| Tool abuse, confused deputy, excessive agency | Yes | 16, 26 | Production | Project 4 injection test |
| Data exfiltration channels and egress control | Yes | 26, 27 | Production | `guardrails.UrlAllowlistCheck` |
| Secrets exposure | Yes | 26, 27 | Production | `guardrails.secrets` |
| Insecure output handling | Yes | 26, 27, 36 | Production | guardrails output checks; Ch 36 SQL guard |
| Permission boundaries and tenant isolation | Yes | 15, 27, 28 | Production | `guardrails.tenancy`; Project 3 isolation tests |
| PII detection and redaction | Yes | 27 | Production | `guardrails.pii` with vault |
| Content moderation | Yes | 26, 27 | Intermediate | `guardrails.moderation` |
| Incident response and red-team plan | Yes | 26 | Production | Ch 26 runbook |
| Measuring guardrails (false positives, bypass rate) | Yes | 27 | Production | `guardrails.eval.measure` |

## Production Engineering

| AI Engineering Skill | Covered? | Chapter | Depth | Practical Exercise |
|---|---|---|---|---|
| Reference AI application architecture | Yes | 28 | Production | `examples/ch28` API skeleton, schema, Compose |
| Async job model, queues, workers | Yes | 28, 29 | Production | `reliability.JobQueue`, Worker; Ch 28 bridge |
| SLOs and burn-rate alerting | Yes | 29, 31 | Production | `reliability.slo`; Ch 31 alert rules |
| Circuit breakers, bulkheads, hedged requests | Yes | 29 | Production | `reliability` |
| Admission control, load shedding, degraded modes | Yes | 29, 15 | Production | AdmissionController; DegradePolicy |
| Malformed responses and partial failures in chains | Yes | 29 | Production | complete_with_recovery; run_chain; chaos tests |
| Latency budgets and token budgets | Yes | 28, 30 | Production | Ch 30 LatencyBudget |
| Caching layers with correctness keys | Yes | 30, 15 | Production | Ch 30 caching and cache-key linter |
| Parallelization and streaming for perceived latency | Yes | 30 | Production | Ch 30 fan_out |
| Cost model per successful task, monitoring, chargeback, spend guards | Yes | 30 | Production | Ch 30 CostModel, SpendGuard |
| Observability: traces of prompts, responses, tools, retrieval, tokens, cost | Yes | 31 | Production | `examples/ch31` AITracer, OTel setup |
| Agent trajectory views and eval results in traces | Yes | 31 | Production | Ch 31 analysis tools |
| Debugging degraded output quality | Yes | 31 | Advanced | Ch 31 incident walkthrough |
| Clean architecture, modularity, provider abstraction | Yes | 32 | Production | `examples/ch32` layered app with fitness test |
| Testability: fakes, record/replay, property tests, contract tests | Yes | 32 | Production | Ch 32 tests |
| Versioning prompts, models, indexes, datasets; version manifest | Yes | 32 | Production | Ch 32 VersionManifest |
| Configuration management | Yes | 32 | Production | Ch 32 config validation |
| Feature flags, experiments, canary, shadow traffic | Yes | 32 | Production | Ch 32 flags and experiments |
| CI/CD for AI changes | Yes | 25, 32, 39 | Production | GitHub Actions and GitLab CI files |
| Fine-tuning decision, SFT, LoRA/PEFT, dataset construction, evaluation, deployment | Yes | 33 | Intermediate | `examples/ch33` |
| Self-hosting and serving: batching, quantization, capacity planning | Yes | 34 | Intermediate | `examples/ch34` load test and capacity tools |

## AI System Design

| AI Engineering Skill | Covered? | Chapter | Depth | Practical Exercise |
|---|---|---|---|---|
| Ten-step system design method with back-of-envelope estimates | Yes | 35 | Advanced | `examples/ch35/back_of_envelope.py` |
| Enterprise knowledge assistant | Yes | 35 | Advanced | Ch 35 P-exercises; Project 3 |
| Customer support copilot (chat and voice) | Yes | 35 | Advanced | Ch 35; Project 4 |
| Document processing system | Yes | 35 | Advanced | Ch 35; Project 1 |
| Coding assistant | Yes | 35, 38 | Advanced | Ch 38 coding harness |
| Research agent | Yes | 36 | Advanced | Ch 36; Project 6 |
| Analytics assistant (text-to-SQL) | Yes | 36 | Advanced | `examples/ch36/sql_guard.py` |
| Enterprise workflow automation | Yes | 36 | Advanced | Ch 36; Ch 17 workflow engine |
| LLM serving platform and evaluation platform | Yes | 36 | Advanced | Ch 36 |

## Capstone

| AI Engineering Skill | Covered? | Chapter | Depth | Practical Exercise |
|---|---|---|---|---|
| Integrating API/UI, LLM, retrieval, tools, structured output, memory, evaluation, observability, authentication, security, caching, deployment, CI/CD and cost monitoring in one system | Yes | 39 | Production | `capstone/northwind-assist`: composes `aie_core`, `ragkit`, `semsearch`, P1, P3, P4 (`toolkit`), `agentkit`, `memorykit`, `evalkit`, `guardrails`, `reliability` and the Ch 4, 5, 7, 25, 26, 30, 31, 32 examples; 67 offline tests; release gate run on the candidate (exit 0) and with the ACL filter disabled (exit 1); Ch 39 P1-P5 |
| Operating an AI system: runbook, dashboards, alerts, failure analysis, production readiness review | Yes | 39 | Production | capstone README runbook; `ops/alerts.yaml` tested on capstone traces; Ch 39 "Operating the system", two failure analyses, readiness table; D1-D3 |

## Known gaps and limits of this edition

- **Not verified against live infrastructure.** pgvector DDL and HNSW settings, Redis-backed queues,
  a live OpenTelemetry collector, and most Docker image builds were checked offline or with emulation only,
  because those services were unavailable in the build environment. The capstone image is the exception:
  it was built from a local base-image mirror, passed the release gate inside the container, and served
  a streamed answer; its Compose stack was not started because the pgvector and collector images were
  unavailable. Integration tests for them exist
  and are skipped by default.
- **Multimodal generation** (image and audio generation models) is out of scope; multimodal *inputs*
  are covered in Chapters 7 and 37, and voice agents in Chapters 35 and 38.
- **Training internals** (backpropagation, alignment algorithms such as PPO, DPO and GRPO) are
  deliberately reduced to one conceptual section in Chapter 33.
- **Inference engine internals** (kernels, disaggregated serving) are summarized as options in
  Chapter 34, not taught in depth.
