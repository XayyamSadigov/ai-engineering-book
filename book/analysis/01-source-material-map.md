# Source Material Map

Source: `AI_Engineering_Complete_Study_Book_v2_2026.pdf` (199 pages, ~80,000 words, 1,037 TOC entries,
generated 2026-10-03 from a python-docx/LibreOffice pipeline). Self-described as a "self-contained
learning companion based on the topic map of the public AI Engineering Course by Amit Shekhar /
Outcome School". Extracted text is kept at `analysis/source-extracted-text.txt`.

## 1. Structure of the source

Every module follows one rigid template:

```
Module N: Title
  Learning objectives
  Core lessons
    k. Lesson title
       (summary paragraph)
       Mechanics and context
       Engineering takeaways
       Study lens / Production test / Mini exercise / Mastery check   <- templated boilerplate
  Module synthesis notes
  Deep study (Deep Dive / Worked Example / Practical Lab)
  Advanced technical notes (Advanced Note / Deep Practice)
  Workshop X
  Exercises + Answer key (answers directly below questions)
```

Roughly 20% of the text is the four templated paragraphs ("Study lens", "Production test",
"Mini exercise", "Mastery check") repeated for all 153 lessons with the lesson name substituted in.
They carry almost no information and are not preserved.

## 2. Inventory by module

| # | Module | Pages | Words | Core content | Code present | Relevance to AI Engineering |
|---|--------|------:|------:|--------------|--------------|------------------------------|
| P | Mathematical and Engineering Primer | 3-5 | 1.1k | vectors, dot product, softmax, cross-entropy, KL, precision formats, Big-O, queueing intuition; Workshop A (cosine, softmax by hand, shape drill) | none | Partial: cosine/softmax/queueing/shape literacy useful; rest is ML math |
| 0 | Must Know | 6-11 | 2.7k | LLM, RAG, MCP, agent, fine-tuning, quantization; **five-layer stack** (model/context/action/control/operations); **decision order** prompt -> RAG -> tools -> FT -> agent; policy-assistant worked example | none | **High** - core mental models |
| 1 | ML Foundations | 12-21 | 4.5k | supervised/unsupervised, regression, features, precision/recall, L1/L2, regularization, RL, contrastive learning; leakage, imbalance, calibration, drift, threshold cost worked example; experiment design checklist | none | Partial: precision/recall/threshold/calibration/leakage/drift/experiment discipline are needed for evaluation; rest is classical ML |
| 2 | Deep Learning | 22-32 | 4.6k | bias, gradient descent, backprop, cross-entropy, dropout, BatchNorm/LayerNorm/RMSNorm, RNNs, PyTorch, TF Lite; training-loop recipe; Workshop B | PyTorch loop (15 lines) | Low: out of scope except "why cross-entropy = next-token prediction" |
| 3 | Transformer | 33-48 | 6.4k | generative AI, autoregression, **tokenization/BPE**, **embeddings**, RNN vs Transformer, encoder/decoder, self-attention, Q/K/V, multi-head, causal mask, RoPE, FFN, residuals, logits; NumPy attention; Workshops C/D | NumPy attention (12 lines), PyTorch block skeleton | Medium: tokenization, embeddings, attention intuition, context-length cost, KV-cache origin are needed; derivations are not |
| 4 | How LLMs Generate Text | 49-53 | 2.0k | temperature, top-k/top-p, streaming (TTFT/TPOT), lost-in-the-middle, repetition/stopping, constrained decoding | none | **High** |
| 5 | Modern LLM Architecture | 54-60 | 2.9k | MoE, GQA, sliding window, attention sinks, FlashAttention, DeepSeek V4/V4.1 note | none | Low: compress into "why KV cache / context cost behave as they do" |
| 6 | Types of Language Models | 61-66 | 2.4k | SLMs, reasoning models (test-time compute as a knob), recursive LMs, diffusion LMs, "System One"/Jev decision models; **model selection matrix**, routing and cascades | none | **High** for model selection (vendor-specific items dated) |
| 7 | Training, Fine-Tuning, Alignment | 67-78 | 5.1k | SFT, LoRA (parameter derivation), prefix tuning, distillation, continual learning, RLHF, InstructGPT, PPO, DPO, GRPO; **fine-tuning decision checklist**; adaptation experiment protocol; Workshop E | none | Medium: when/when-not, SFT, LoRA/QLoRA, data, eval are needed; PPO/DPO/GRPO internals are not |
| 8 | Prompt & Context Engineering | 79-84 | 2.3k | CoT, chaining, prompt caching, context engineering pipeline, compaction; prompt-as-contract, few-shot, structured outputs, versioning; regression harness lab | none | **High** but thin (5 lessons) |
| 9 | Vector Search & RAG | 85-99 | 6.2k | vector DBs, ANN/HNSW, semantic/hybrid search, rerankers, ColBERT, chunking, HyDE, embedding cache, semantic cache, agentic RAG, GraphRAG, vectorless RAG; **RAG is two systems**; RRF; metrics worked example; **debugging decision tree**; production architecture; Workshops F/G | pseudocode `answer()` (9 lines) | **High** |
| 10 | Agents | 100-114 | 6.0k | agent definition, function calling, loop, ReAct, plan-and-execute, reflection, memory taxonomy, MCP, Agent Skills, multi-agent, subagents, orchestration, supervisor, permission boundaries, computer use, stop conditions; **state-machine pseudocode**; approval/side effects; Workshops H/I | pseudocode loop (9 lines) | **High** |
| 11 | Agentic Engineering & Frameworks | 115-122 | 2.9k | harness, loop, graph engineering, Definition of Done, LangChain, LangGraph, coding agents, IDE agents; framework selection criteria; checkpointing/replay | none | **High** (frameworks shallow) |
| 12 | Inference Engineering | 123-139 | 6.7k | TTFT/TPOT/goodput, prefill/decode, disaggregation, KV cache (+calculator), compression, PagedAttention, continuous batching, speculative decoding, Medusa, EAGLE, quantization, GGUF, llama.cpp, vLLM, SGLang, TensorRT-LLM; capacity planning; Workshop J | none | Medium: metrics, KV cache, batching, quantization, capacity planning needed; engine internals compress |
| 13 | Evaluation & Observability | 140-146 | 2.8k | evaluation, LLM-as-judge (biases), agent evaluation, traces/spans; golden datasets, deterministic-first, replay, error taxonomy, **release gate**, confidence intervals, online signals; Workshop K (JSON case example) | JSON example (10 lines) | **High** but thin |
| 14 | Safety & Security | 147-152 | 2.5k | guardrails, prompt injection, watermarking; threat model, exfiltration, tool security, sandboxing, jailbreaks, RAG security, supply chain, privacy, red-team plan; Workshop L | none | **High** but thin |
| 15 | Multimodal & Generative | 153-159 | 2.5k | multimodal, ViT, image embeddings, diffusion, GANs, VAEs | none | Low: only "multimodal inputs in products" survives |
| 16 | Infrastructure, Deployment, System Design | 160-172 | 5.3k | GPUs, CUDA, TPUs, LPUs, cloud vs device, TF Lite, **LLM routing**, voice agent, **system design method**, HTTP/SSE/WebSocket, WebRTC; API design, async jobs, reliability, caching, data architecture, back-of-envelope; **Deep Practice: SLOs, request path, backpressure, routing, caching, failure modes, observability, cost model, rollout**; Workshop N checklist | none | **High** for the production half; hardware half is low |
| 17 | Frontier Ideas | 173-176 | 1.6k | JEPA, world models, recursive self-improvement, evidence hierarchy | none | Low: only "evidence discipline" survives |
| 18 | Interviews | 177-181 | 1.8k | question blocks per topic, system-design framework, worked examples | none | Medium: becomes an appendix |
| CS | End-to-End Case Studies | 182-186 | 2.1k | 8 cases: knowledge assistant, coding agent, voice agent, serving platform, document extraction, domain classifier, evaluation platform, edge assistant | none | **High** |
| A | Appendices | 187-198 | 3.9k | Production playbook (10 steps), glossary (~60 terms), 20 recipes, 5 capstone specs + acceptance tests, 12-week plan, mastery standard | recipes 1-13 (toy code, 5-15 lines each) | **High** as seeds |

## 3. What the source does well (preserve)

- Systems framing: "AI engineering is systems engineering around probabilistic models"; five-layer stack.
- The decision ladder: prompt -> retrieval -> tools -> fine-tuning -> agent loop, with "architectural restraint".
- "The model proposes; the harness disposes" - authorization lives outside the model.
- RAG as two separately evaluated systems; stage-by-stage debugging decision tree; recall-first then precision.
- Memory as a database-design problem with write policy, provenance, expiration.
- Evaluation as a release-engineering system: failure taxonomy first, deterministic-first, judge calibration,
  frozen holdout, release gate, replay.
- Security as authorization, not prompt wording; threat model lists; red-team checklist.
- Production: SLOs, deadline propagation, backpressure, cost per successful task, rollout with canaries.
- Concrete worked numbers: KV-cache formula, Little's Law, back-of-envelope token volumes, threshold cost
  example, retrieval metrics example, LoRA parameter count.
- Case studies and capstone acceptance tests with "it answers my demo questions is not acceptance".

## 4. Weaknesses of the source (fix)

1. **Almost no runnable code.** 20 recipes of 5-15 lines; no directory structures, dependencies, config,
   tests, or execution instructions. The RAG and agent "implementations" are pseudocode.
2. **No chapter on working with LLM APIs** (messages, roles, SDKs, streaming, structured outputs, tool
   calling mechanics, retries, rate limits, timeouts, batching, fallbacks). The book assumes the reader
   already calls models.
3. **Templated filler**: the four repeated lens paragraphs per lesson (~16k words) and duplicated
   explanations between "Core lessons", "Synthesis notes", "Deep dive", and "Deep practice" (the KV-cache
   formula appears four times; the agent loop three times; RAG pipeline five times).
4. **Depth inverted relative to the job.** Transformer internals, alignment algorithms, serving engines,
   GANs/VAEs/diffusion, GPUs/TPUs/CUDA, JEPA/world models get ~30k words; prompt/context engineering,
   evaluation, and security together get ~7.5k words.
5. **Missing topics**: embeddings beyond RAG, pgvector vs dedicated vector DB, parsing/OCR/tables in
   ingestion, parent-child and contextual retrieval details, extraction/classification pipelines,
   model fallbacks, concurrency/batching code, memory implementation, evaluation harness code,
   CI/CD integration, guardrail implementations, PII handling, clean architecture, provider abstraction,
   feature flags, LlamaIndex/DSPy/agent SDKs, analytics assistant and workflow automation designs.
6. **Exercises** are 3-5 short questions per module with answers immediately below; no debugging
   exercises, no implementation exercises with acceptance criteria.
7. **Vendor-dated items** presented as facts (MCP 2026-07-28 revision details, DeepSeek V4.1 Flash,
   "System One"/Jev). These cannot be verified from stable sources and are treated as dated claims.
8. **Diagrams**: the PDF references "Conceptual architecture" figures, but they are raster images that
   did not survive extraction; the text never describes them.

## 5. Duplication map (merge targets)

| Concept | Appears in | Merged into chapter |
|---|---|---|
| Decision ladder prompt/RAG/tools/FT/agent | M0 deep dive, M7 checklist, Recipe 18, Playbook A | Ch 1, Ch 17 |
| Agent loop / state machine | M0, M10 (x3), M11, Recipe 8 | Ch 19 |
| RAG pipeline stages | M0, M9 (x5), CS1, Recipe 5/15 | Ch 10, Ch 15 |
| KV-cache formula | M3, M5, M12 (x3), Recipe 11 | Ch 2, Ch 34 |
| Prompt injection defence | M8, M10, M14 (x4), Recipe 9 | Ch 26, Ch 27 |
| Evaluate retrieval before generation | M9 (x4), M13, CS1 | Ch 14 |
| Release gate | M13 (x3), Recipe 14, CS7 | Ch 25, Ch 32 |
| Routing and cascades | M6 (x3), M16 (x3), Recipe 13 | Ch 7 |
| Memory taxonomy | M10 (x3) | Ch 21 |
| Back-of-envelope capacity | M12, M16 (x3), Recipe 12 | Ch 30, Ch 34 |
