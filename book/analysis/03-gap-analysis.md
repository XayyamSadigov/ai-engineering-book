# Gap Analysis: Source vs Competency Map

Legend for source coverage: **Full** (concept + mechanism + production notes), **Partial** (concept only
or thin), **Absent**. "Resolution" names the chapter(s) of the new book that close the gap.

| Competency | Source coverage | Specific gaps | Resolution |
|---|---|---|---|
| A. LLM internals | Full (over-deep) | Depth is in derivations (RoPE, backprop, MoE) rather than consequences; no tokenizer cost experiment; capability-limit discussion scattered | Ch 2 rewrites internals as engineering consequences; derivations dropped; Ch 34 keeps serving math |
| B. LLM APIs | **Absent** | No messages/roles, SDK usage, streaming code, tool-call protocol mechanics, error taxonomy, retries/backoff, rate limits, concurrency, batching, caching, fallbacks, token counting | **New Ch 3** with the `aie_core` gateway library and tests |
| C. Prompt engineering | Partial | 5 lessons; no templates, instruction hierarchy, dynamic prompts, prompt versioning implementation, test harness code, defensive prompting examples | Ch 4 (full implementation of prompt registry + regression tests) |
| D. Context engineering | Partial | Pipeline listed, no implementation of budgeting/ordering/compaction; conversation state and ephemeral vs persistent not developed | Ch 5 (ContextBuilder with budget, ordering, compaction, tests) |
| E. Embeddings | Partial | No implementation; dimensionality/normalization/model choice thin; no uses beyond RAG | Ch 8 (embedding service, similarity metrics, clustering/dedup/classification/routing examples, evaluation) |
| F. Vector search/storage | Partial | HNSW intuition only; no pgvector; no filtering/namespaces/collections implementation; "when unnecessary" only as vectorless note | Ch 9 (NumPy brute force, pgvector, dedicated DB comparison, filtering, decision matrix) |
| G. Ingestion/chunking | Partial | Parsing/OCR/tables mentioned in two paragraphs; chunking strategies listed, none implemented; no metadata model | Ch 11 (parsers, normalization, document model, five chunkers with tests) |
| H. Retrieval | Full concept / no code | BM25, RRF, reranking, HyDE, multi-query described; no implementation; contextual retrieval and parent-document retrieval only hinted | Ch 12 (BM25 from scratch, hybrid with RRF, reranker, query transforms, parent-child, contextual retrieval) |
| I. Grounded generation | Partial | Grounding contract described; no citation validation implementation; no abstention logic | Ch 13 |
| J. RAG evaluation | Partial | Metrics named; one worked example; recipe for recall@k; no faithfulness/relevance judges; no stage isolation code | Ch 14 (eval dataset format, metrics module, faithfulness judge, stage isolation report) |
| K. Production RAG | Full concept / no code | Incremental updates, deletion, ACL, multi-tenancy, caching described; no pipeline implementation; no observability wiring | Ch 15 + Project 3 |
| L. Tool calling | Partial | Recipe 7 (toy); schema design, validation, idempotency, permissions, sandboxing described abstractly | Ch 16 (tool registry, policy engine, idempotency store, sandbox runner) + Project 4 |
| M. Workflows vs agents | Full concept | No decision framework table; no graph implementation; orchestration patterns listed without code | Ch 17 (decision framework, plain state-machine implementation) |
| N. Agent loop | Full concept / pseudocode | No typed state, event log, termination, replay implementation | Ch 19 (complete agent runtime with tests) |
| O. Agent architectures | Partial | ReAct/plan-execute/reflection/supervisor described; router, evaluator-optimizer, parallel, hierarchical not developed; no implementations; failure modes scattered | Ch 20 |
| P. Memory | Partial | Taxonomy strong; no implementation; summarization/consolidation/expiration/privacy as bullets | Ch 21 (memory store implementations) |
| Q. Multi-agent | Partial | Patterns listed; no message contracts, no cost accounting, no implementation | Ch 22 + Project 6 |
| R. MCP | Partial (dated) | Protocol-level explanation tied to a specific 2026 revision; no server/client architecture walkthrough; security questions listed | Ch 18 (architecture-first, revision-agnostic, small server + client) |
| S. Evaluation | Partial | Strong principles; one JSON case; no harness, judge, pairwise, statistics code; no CI integration; extraction/classification/summarization evaluation absent | Ch 24, Ch 25 (full harness, judges, bootstrap CIs, pytest/CI integration, per-task evaluators) |
| T. Security | Partial | Principles strong; no guardrail implementations, PII redaction, policy engine, tenant isolation tests, moderation integration | Ch 26, Ch 27 |
| U. Structured output | Partial (scattered) | No schema design guidance, no validation/repair loop, no extraction/classification pipeline | Ch 6 + Project 1 |
| V. Model selection | Full | Routing pseudocode only; no evaluation-driven selection procedure | Ch 7 (selection procedure, router implementation, cascade evaluation) |
| W. Fine-tuning | Full (over-deep) | Alignment algorithms exceed need; dataset construction and deployment thin; no code | Ch 33 (decision, SFT/LoRA engineering, dataset builder, eval protocol, deployment) |
| X. Application architecture | Partial | Request path and components described; no diagrams; no layering guidance | Ch 28 (reference architecture with Mermaid diagrams) |
| Y. Reliability/scalability | Full concept | No implementation of retries, circuit breaker, queue/worker, admission control | Ch 29 |
| Z. Performance/cost | Full concept | No cost model code; caching layers listed; no latency budget tooling | Ch 30 |
| AA. Observability | Partial | Trace schema described; no OpenTelemetry wiring; no debugging playbook with trace queries | Ch 31 |
| AB. Engineering practices | **Absent** | No clean architecture, provider abstraction, config, feature flags, experiments, CI/CD, testing with fakes | **New Ch 32** |
| AC. Frameworks | Partial | LangChain/LangGraph only; LlamaIndex, DSPy, agent SDKs absent; "primitive first" principle stated | Ch 23 |
| AD. Inference/serving | Full (over-deep) | Engine-specific detail; keep metrics, KV math, batching, quantization, capacity | Ch 34 (compressed) |
| AE. System design | Full (8 cases) | Analytics assistant and enterprise workflow automation absent; customer support copilot only as voice; research agent only as interview sketch | Ch 35, Ch 36 (seven required cases, uniform 10-step method) |
| AF. Judgment | Full | Spread across modules | Ch 1, Appendix: interview preparation |

## Out-of-scope material removed or compressed (brief section 23)

- Modules 1-2 (classical ML, backprop, normalization, PyTorch/TF training) -> only precision/recall,
  thresholds, calibration, leakage, drift survive, inside the evaluation chapters.
- Module 3 derivations (RoPE math, multi-head subspaces, Transformer-from-scratch workshops) -> Ch 2
  keeps attention intuition, tokenization, context cost; workshops dropped.
- Module 5 (MoE, GQA, sliding window, sinks, FlashAttention, DeepSeek) -> two paragraphs in Ch 2/Ch 34.
- Module 7 alignment algorithms (PPO/DPO/GRPO/RLHF mechanics) -> one section in Ch 33.
- Module 12 engine internals (Medusa, EAGLE, disaggregation, GGUF, SGLang, TensorRT-LLM) -> one
  comparison table in Ch 34.
- Module 15 (ViT, diffusion, GANs, VAEs) -> dropped; multimodal inputs covered in Ch 7 and Ch 36.
- Module 16 hardware (GPUs, CUDA, TPUs, LPUs, Android TF Lite, WebRTC) -> dropped except a paragraph on
  memory-bandwidth-bound decode in Ch 34 and transport choice (SSE vs WebSocket) in Ch 29.
- Module 17 (JEPA, world models, RSI) -> dropped; evidence discipline kept in Ch 1.
- Dated vendor claims (MCP 2026-07-28 specifics, DeepSeek V4.1, Jev/System One) -> not asserted;
  MCP is explained revision-agnostically with a freshness note.

## Prerequisite ordering fixes

- Source introduces RAG (M9) before explaining APIs, structured outputs, or embeddings implementation.
  New order: APIs -> prompting -> context -> structured output -> embeddings -> vector search -> RAG.
- Source introduces agents (M10) before tool calling has been implemented. New order: tool calling ->
  workflows -> MCP -> agent loop -> architectures -> memory -> multi-agent -> frameworks.
- Source places evaluation (M13) after agents and inference. New book introduces evaluation early as a
  habit (Ch 4 prompt tests, Ch 14 RAG evaluation) and then dedicates Part VII to it in depth.
- Source puts security last-but-two. New book threads security into tool calling (Ch 16) and RAG
  (Ch 15), then dedicates Part VIII.

## Added material (not in source)

LLM API client library; prompt registry and tests; ContextBuilder; extraction/classification pipelines;
embedding use cases beyond RAG; pgvector; parsers and chunkers; BM25/RRF/reranker code; citation
validator; evaluation harness with judges and statistics; tool registry/policy/idempotency/sandbox;
plain state-machine orchestrator; MCP server/client; agent runtime with event log and replay;
memory stores; multi-agent coordinator; guardrail implementations; reliability primitives (retry,
circuit breaker, worker queue, admission control); cost model; OpenTelemetry tracing; engineering
practices chapter; analytics assistant and workflow automation designs; six projects and a capstone with
code, tests, and run instructions; debugging exercises with separate solutions; coverage matrix.
