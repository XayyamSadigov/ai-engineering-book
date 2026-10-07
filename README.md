<h1 align="center">AI Engineering</h1>
<h3 align="center">From Software Engineer to Production AI Engineer</h3>

<p align="center">
  <a href="https://xayyamsadigov.github.io/ai-engineering-book/"><img alt="Read online" src="https://img.shields.io/badge/read-online-3f51b5?style=for-the-badge"></a>
  <img alt="39 chapters" src="https://img.shields.io/badge/chapters-39-ff7043?style=for-the-badge">
  <img alt="6 projects + capstone" src="https://img.shields.io/badge/projects-6%20%2B%20capstone-26a69a?style=for-the-badge">
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-555?style=for-the-badge">
</p>

**A free, complete textbook and field guide for building production applications on large language models — 39 chapters, a shared Python library, six projects, a production-grade capstone, exercises with separate solutions, and system design cases. Every pattern in the book comes with runnable, tested code.**

> This book is for you if you want to become:
>
> - AI Engineer / Applied AI Engineer
> - LLM Engineer / Gen AI Engineer
> - Agentic AI Engineer
> - AI Platform Engineer
> - AI Solutions Architect / Forward Deployed Engineer
> - Backend or platform engineer who ships AI features

**Read online:** <https://xayyamsadigov.github.io/ai-engineering-book/> · **Single file:** [AI_ENGINEERING_BOOK.md](book/AI_ENGINEERING_BOOK.md)

**Run the code in two commands** (needs [uv](https://docs.astral.sh/uv/)): `book/tools/setup_dev.sh` then `book/tools/verify_code.sh`. Everything runs offline; no API key needed. Details in [Running the Code](#running-the-code).

---

## Table of Contents

- [About This Book](#about-this-book)
- [Who This Book Is For](#who-this-book-is-for)
- [What You Will Build](#what-you-will-build)
- [How to Use This Book](#how-to-use-this-book)
- [Curriculum at a Glance](#curriculum-at-a-glance)
- [Part I: AI Engineering Foundations](#part-i-ai-engineering-foundations)
- [Part II: LLM Application Development](#part-ii-llm-application-development)
- [Part III: Embeddings and Retrieval](#part-iii-embeddings-and-retrieval)
- [Part IV: Production RAG](#part-iv-production-rag)
- [Part V: Tools and Workflows](#part-v-tools-and-workflows)
- [Part VI: Agents](#part-vi-agents)
- [Part VII: Evaluation](#part-vii-evaluation)
- [Part VIII: Security and Guardrails](#part-viii-security-and-guardrails)
- [Part IX: Production AI Engineering](#part-ix-production-ai-engineering)
- [Part X: AI System Design](#part-x-ai-system-design)
- [Part XI: Advanced Patterns](#part-xi-advanced-patterns)
- [Part XII: Capstone](#part-xii-capstone)
- [Projects and Libraries](#projects-and-libraries)
- [Appendices](#appendices)
- [Running the Code](#running-the-code)
- [FAQ](#faq)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## About This Book

**This book takes a strong software engineer to the point where they can design, implement, evaluate, secure, deploy, observe, and improve production AI applications built on LLMs.**

It is written as engineering, not as a catalog of tricks. Every chapter follows the same shape — *why this matters, mental model, core concepts, how it works, architecture, implementation, code walkthrough, production considerations, common mistakes, failure modes, exercises, key takeaways* — and every implementation is real code that runs offline in tests against fake model and embedding clients, so you can study it without an API key.

One running example ties the book together: **Northwind Assist**, an internal assistant for a fictional company with `retail` and `logistics` tenants. It makes its first model call in Chapter 3 and becomes a complete, deployable system in Chapter 39.

In simple words:

**AI Engineering = understanding model behavior + building reliable systems around it + operating them in production.**

## Who This Book Is For

- **Backend, full-stack, and platform engineers** comfortable with Python, HTTP APIs, SQL, testing, and containers, who are new to building on LLMs.
- **Engineers joining a RAG, search, or agents team** who need the production view, not just the demo.
- **SREs and platform engineers** who will run AI services and need SLOs, observability, and cost control.
- **Tech leads and architects** who must choose between prompting, RAG, tools, fine-tuning, and agents — and defend the choice.
- **Anyone preparing for AI engineering interviews** (see [the interview appendix](book/appendix-c-interview-preparation.md)).

No machine-learning background is assumed, and none is taught beyond what explains model behavior.

## What You Will Build

| | What | Where |
|---|---|---|
| 🧱 | `aie_core` — provider-neutral LLM client and gateway: retries, fallbacks, rate limits, caching, tracing, cost | [book/projects/aie_core](book/projects/aie_core) |
| 1️⃣ | **Project 1** — structured extraction API (FastAPI) | [p1-extraction-api](book/projects/p1-extraction-api) |
| 2️⃣ | **Project 2** — semantic search system (pgvector with NumPy fallback) | [p2-semantic-search](book/projects/p2-semantic-search) |
| 3️⃣ | **Project 3** — production RAG assistant with citations, abstention, and an eval gate | [p3-rag-assistant](book/projects/p3-rag-assistant) |
| 4️⃣ | **Project 4** — support assistant with governed tools and approvals | [p4-support-assistant](book/projects/p4-support-assistant) |
| 5️⃣ | **Project 5** — incident-research agent with planning, replanning, and replay | [p5-incident-agent](book/projects/p5-incident-agent) |
| 6️⃣ | **Project 6** — multi-agent research team, where justified | [p6-research-team](book/projects/p6-research-team) |
| 🏁 | **Capstone** — Northwind Assist: UI/API, RAG, tools, memory, eval, observability, auth, security, CI/CD, K8s | [capstone/northwind-assist](book/capstone/northwind-assist) |

## How to Use This Book

1. **Start with the [Learning Roadmap](book/00-learning-roadmap.md).** It has a full 16-week path, an accelerated 8-week path, and role-based paths (backend, RAG/search, agents, platform/SRE).
2. **Read in order.** Each part builds on the previous one; the shared library from Part I is imported everywhere.
3. **Run the code as you read.** Each chapter points to its package or example under `book/projects/`.
4. **Do the exercises before opening the solutions.** Exercises are in [`book/exercises/`](book/exercises), answers are kept separately in [`book/solutions/`](book/solutions).
5. **Finish with the capstone** and its acceptance checklist.

## Curriculum at a Glance

| Part | Chapters | What you build |
|---|---|---|
| I. AI Engineering Foundations | 1–3 | the shared `aie_core` library |
| II. LLM Application Development | 4–7 | prompt registry, context builder, **Project 1**, model router |
| III. Embeddings and Retrieval | 8–9 | embedding toolkit, **Project 2** |
| IV. Production RAG | 10–15 | parsers, chunkers, hybrid retrieval, grounded generation, RAG eval, **Project 3** |
| V. Tools and Workflows | 16–18 | tool registry, policy engine, sandbox, **Project 4**, workflow engine, MCP |
| VI. Agents | 19–23 | agent runtime with replay, **Project 5**, memory stores, **Project 6** |
| VII. Evaluation | 24–25 | `evalkit`, CI release gate |
| VIII. Security and Guardrails | 26–27 | threat models, adversarial corpus, `guardrails` |
| IX. Production AI Engineering | 28–34 | reference architecture, reliability, cost, observability, fine-tuning, serving |
| X. AI System Design | 35–36 | the 10-step method applied to nine systems |
| XI. Advanced Patterns | 37–38 | advanced retrieval, durable agents |
| XII. Capstone | 39 | **Northwind Assist** |

## Part I: AI Engineering Foundations

| # | Chapter | What you will learn | Practice |
|---|---|---|---|
| 1 | [What AI Engineering Is](book/chapters/01-what-ai-engineering-is.md) | The discipline as systems engineering around probabilistic models: the five-layer stack, the decision ladder from prompt to agent, request lineage, and architectural restraint. | [Exercises](book/exercises/ch01-exercises.md) · [Solutions](book/solutions/ch01-solutions.md) |
| 2 | [How LLMs Work, for Engineers](book/chapters/02-how-llms-work-for-engineers.md) | Just enough internals to predict behavior: tokens and cost, attention and context windows, prefill vs decode, KV cache, sampling, and the hard limits of LLMs. | [Exercises](book/exercises/ch02-exercises.md) · [Solutions](book/solutions/ch02-solutions.md) |
| 3 | [Working with LLM APIs](book/chapters/03-working-with-llm-apis.md) | Build `aie_core`: a provider-neutral client with streaming, structured output, tool calls, retries, deadlines, rate limits, fallbacks, caching, and cost accounting. | [Exercises](book/exercises/ch03-exercises.md) · [Solutions](book/solutions/ch03-solutions.md) |

## Part II: LLM Application Development

| # | Chapter | What you will learn | Practice |
|---|---|---|---|
| 4 | [Prompt Engineering as Engineering](book/chapters/04-prompt-engineering.md) | Prompts as versioned interface contracts: instruction hierarchy, anatomy, few-shot, decomposition, templates, a prompt registry, and regression tests in CI. | [Exercises](book/exercises/ch04-exercises.md) · [Solutions](book/solutions/ch04-solutions.md) |
| 5 | [Context Engineering](book/chapters/05-context-engineering.md) | The context window as an engineering budget: what goes in, ordering, compression, conversation state, and a context builder with token accounting. | [Exercises](book/exercises/ch05-exercises.md) · [Solutions](book/solutions/ch05-solutions.md) |
| 6 | [Structured Output and Extraction](book/chapters/06-structured-output-and-extraction.md) | Schema design, constrained generation, validation and repair loops. **Project 1:** a structured extraction API for invoices and tickets. | [Exercises](book/exercises/ch06-exercises.md) · [Solutions](book/solutions/ch06-solutions.md) |
| 7 | [Model Selection and Routing](book/chapters/07-model-selection-and-routing.md) | Choosing models on capability, latency, price, and context; benchmarks you can trust; cascades, routers, and fallbacks with measured trade-offs. | [Exercises](book/exercises/ch07-exercises.md) · [Solutions](book/solutions/ch07-solutions.md) |

## Part III: Embeddings and Retrieval

| # | Chapter | What you will learn | Practice |
|---|---|---|---|
| 8 | [Embeddings](book/chapters/08-embeddings.md) | What embeddings are, similarity metrics, model choice, normalization, batching, caching, and evaluating embeddings on your own data. | [Exercises](book/exercises/ch08-exercises.md) · [Solutions](book/solutions/ch08-solutions.md) |
| 9 | [Vector Search and Vector Databases](book/chapters/09-vector-search-and-vector-databases.md) | Exact vs approximate search (HNSW, IVF, PQ), filtering, pgvector. **Project 2:** a filtered semantic search system with a NumPy fallback. | [Exercises](book/exercises/ch09-exercises.md) · [Solutions](book/solutions/ch09-solutions.md) |

## Part IV: Production RAG

| # | Chapter | What you will learn | Practice |
|---|---|---|---|
| 10 | [RAG Fundamentals](book/chapters/10-rag-fundamentals.md) | When RAG is the right architecture, the stage model of a pipeline, a minimal RAG in under 200 lines, and the seven ways naive RAG fails. | [Exercises](book/exercises/ch10-exercises.md) · [Solutions](book/solutions/ch10-solutions.md) |
| 11 | [Ingestion and Chunking](book/chapters/11-ingestion-and-chunking.md) | Document model, parsing PDFs/HTML/Markdown, structure-aware chunking, metadata, ACLs, versioning, and deduplication with `ragkit`. | [Exercises](book/exercises/ch11-exercises.md) · [Solutions](book/solutions/ch11-solutions.md) |
| 12 | [Retrieval Engineering](book/chapters/12-retrieval-engineering.md) | Dense and lexical retrieval, BM25 from scratch, hybrid fusion, query rewriting, reranking, and permission-aware retrieval. | [Exercises](book/exercises/ch12-exercises.md) · [Solutions](book/solutions/ch12-solutions.md) |
| 13 | [Grounded Generation and Citations](book/chapters/13-grounded-generation-and-citations.md) | Evidence packing, the grounded answer contract, chunk-level citations, abstention, and verifying that answers are supported. | [Exercises](book/exercises/ch13-exercises.md) · [Solutions](book/solutions/ch13-solutions.md) |
| 14 | [RAG Evaluation](book/chapters/14-rag-evaluation.md) | Building a retrieval gold set, recall/MRR/nDCG, faithfulness and answer quality, and a RAG evaluation harness. | [Exercises](book/exercises/ch14-exercises.md) · [Solutions](book/solutions/ch14-solutions.md) |
| 15 | [Production RAG](book/chapters/15-production-rag.md) | Indexing pipelines, incremental updates, caching, degraded modes, tracing. **Project 3:** a production RAG assistant with an eval gate. | [Exercises](book/exercises/ch15-exercises.md) · [Solutions](book/solutions/ch15-solutions.md) |

## Part V: Tools and Workflows

| # | Chapter | What you will learn | Practice |
|---|---|---|---|
| 16 | [Tool Calling](book/chapters/16-tool-calling.md) | Tool schemas, the tool-calling loop, policy engine, approvals, idempotency, sandboxing. **Project 4:** a support assistant with governed tools. | [Exercises](book/exercises/ch16-exercises.md) · [Solutions](book/solutions/ch16-solutions.md) |
| 17 | [AI Workflows and Orchestration](book/chapters/17-ai-workflows-and-orchestration.md) | The spectrum from deterministic workflow to agent, and a workflow engine with steps, branching, retries, and human-in-the-loop. | [Exercises](book/exercises/ch17-exercises.md) · [Solutions](book/solutions/ch17-solutions.md) |
| 18 | [MCP and Tool Ecosystems](book/chapters/18-mcp-and-tool-ecosystems.md) | Model Context Protocol at the architecture level: hosts, clients, servers, tools/resources/prompts; build an MCP server and client. | [Exercises](book/exercises/ch18-exercises.md) · [Solutions](book/solutions/ch18-solutions.md) |

## Part VI: Agents

| # | Chapter | What you will learn | Practice |
|---|---|---|---|
| 19 | [The Agent Loop](book/chapters/19-the-agent-loop.md) | An agent as an explicit loop with typed state, an append-only event log, budgets, termination reasons, policy, verification, and replay (`agentkit`). | [Exercises](book/exercises/ch19-exercises.md) · [Solutions](book/solutions/ch19-solutions.md) |
| 20 | [Agent Architectures](book/chapters/20-agent-architectures.md) | ReAct, routers, planner-executor, supervisor/worker, reflection. **Project 5:** an incident-research agent with replanning and approvals. | [Exercises](book/exercises/ch20-exercises.md) · [Solutions](book/solutions/ch20-solutions.md) |
| 21 | [Memory Systems](book/chapters/21-memory-systems.md) | Memory taxonomy, stores with provenance, confidence and expiry, retrieval of memories, privacy and deletion with `memorykit`. | [Exercises](book/exercises/ch21-exercises.md) · [Solutions](book/solutions/ch21-solutions.md) |
| 22 | [Multi-Agent Systems](book/chapters/22-multi-agent-systems.md) | When multiple agents are justified, coordination patterns and their costs. **Project 6:** a multi-agent research team, where justified. | [Exercises](book/exercises/ch22-exercises.md) · [Solutions](book/solutions/ch22-solutions.md) |
| 23 | [Frameworks](book/chapters/23-frameworks.md) | Popular agent frameworks mapped back to the primitives built in this book, so you can choose, adopt, or skip them deliberately. | [Exercises](book/exercises/ch23-exercises.md) · [Solutions](book/solutions/ch23-solutions.md) |

## Part VII: Evaluation

| # | Chapter | What you will learn | Practice |
|---|---|---|---|
| 24 | [Evaluation Fundamentals](book/chapters/24-evaluation-fundamentals.md) | No evaluation, no engineering: failure taxonomies, datasets, metrics, LLM-as-judge, statistics, and the `evalkit` core. | [Exercises](book/exercises/ch24-exercises.md) · [Solutions](book/solutions/ch24-solutions.md) |
| 25 | [Evaluating AI Systems in Practice](book/chapters/25-evaluating-ai-systems-in-practice.md) | Evaluating prompts, RAG, agents, and tools in practice; online evaluation, A/B tests, and a CI release gate. | [Exercises](book/exercises/ch25-exercises.md) · [Solutions](book/solutions/ch25-solutions.md) |

## Part VIII: Security and Guardrails

| # | Chapter | What you will learn | Practice |
|---|---|---|---|
| 26 | [Threat Modeling and Prompt Injection](book/chapters/26-threat-modeling-and-prompt-injection.md) | Assets, actors and trust boundaries; direct and indirect prompt injection, data exfiltration, tool abuse, and an adversarial corpus. | [Exercises](book/exercises/ch26-exercises.md) · [Solutions](book/solutions/ch26-solutions.md) |
| 27 | [Guardrails and Safe Tooling](book/chapters/27-guardrails-and-safe-tooling.md) | Layered guardrails across input, context, output, and tools; secrets and PII redaction; safe tool execution with the `guardrails` package. | [Exercises](book/exercises/ch27-exercises.md) · [Solutions](book/solutions/ch27-solutions.md) |

## Part IX: Production AI Engineering

| # | Chapter | What you will learn | Practice |
|---|---|---|---|
| 28 | [AI Application Architecture](book/chapters/28-ai-application-architecture.md) | A reference architecture for AI applications: UI, API, orchestration, retrieval, tools, model gateway, data, and operations. | [Exercises](book/exercises/ch28-exercises.md) · [Solutions](book/solutions/ch28-solutions.md) |
| 29 | [Reliability and Scalability](book/chapters/29-reliability-and-scalability.md) | SLOs for AI services, retries, circuit breakers, bulkheads, queues, backpressure, and graceful degradation (`reliability`). | [Exercises](book/exercises/ch29-exercises.md) · [Solutions](book/solutions/ch29-solutions.md) |
| 30 | [Performance and Cost Engineering](book/chapters/30-performance-and-cost-engineering.md) | Latency anatomy (TTFT, TPOT), latency budgets, caching layers, batching, and a cost model you can put in front of finance. | [Exercises](book/exercises/ch30-exercises.md) · [Solutions](book/solutions/ch30-solutions.md) |
| 31 | [Observability for AI Systems](book/chapters/31-observability.md) | Why AI observability differs, the trace model, OpenTelemetry spans for LLM calls, metrics, logs, and alerting on semantic failures. | [Exercises](book/exercises/ch31-exercises.md) · [Solutions](book/solutions/ch31-solutions.md) |
| 32 | [Engineering Practices for AI Systems](book/chapters/32-engineering-practices.md) | Clean architecture for AI apps, testing strategy with fakes, prompt and dataset versioning, CI/CD, and code review for AI systems. | [Exercises](book/exercises/ch32-exercises.md) · [Solutions](book/solutions/ch32-solutions.md) |
| 33 | [Fine-Tuning for Engineers](book/chapters/33-fine-tuning-for-engineers.md) | What fine-tuning changes and when it pays off; data preparation, LoRA, evaluation protocol, and the decision against prompting and RAG. | [Exercises](book/exercises/ch33-exercises.md) · [Solutions](book/solutions/ch33-solutions.md) |
| 34 | [Inference and Serving Essentials](book/chapters/34-inference-and-serving-essentials.md) | Hosted APIs vs self-hosting, serving metrics, batching, quantization, KV cache sizing, and capacity planning. | [Exercises](book/exercises/ch34-exercises.md) · [Solutions](book/solutions/ch34-solutions.md) |

## Part X: AI System Design

| # | Chapter | What you will learn | Practice |
|---|---|---|---|
| 35 | [System Design Method and Cases I](book/chapters/35-system-design-method-and-cases-1.md) | A 10-step system design method applied to real cases: support assistant, enterprise search, document processing, and more. | [Exercises](book/exercises/ch35-exercises.md) · [Solutions](book/solutions/ch35-solutions.md) |
| 36 | [System Design Cases II](book/chapters/36-system-design-cases-2.md) | More design cases: research agent, analytics assistant (text-to-SQL with a SQL guard), and two AI platforms. | [Exercises](book/exercises/ch36-exercises.md) · [Solutions](book/solutions/ch36-solutions.md) |

## Part XI: Advanced Patterns

| # | Chapter | What you will learn | Practice |
|---|---|---|---|
| 37 | [Advanced Retrieval Patterns](book/chapters/37-advanced-retrieval-patterns.md) | Agentic RAG, GraphRAG, multi-hop and structured retrieval, and when each advanced pattern is worth its cost. | [Exercises](book/exercises/ch37-exercises.md) · [Solutions](book/solutions/ch37-solutions.md) |
| 38 | [Durable and Long-Running Agents](book/chapters/38-durable-and-long-running-agents.md) | Checkpointing and resumption, durable execution, idempotent side effects, long-running agents, and human approvals over days. | [Exercises](book/exercises/ch38-exercises.md) · [Solutions](book/solutions/ch38-solutions.md) |

## Part XII: Capstone

| # | Chapter | What you will learn | Practice |
|---|---|---|---|
| 39 | [Capstone: Northwind Assist](book/chapters/39-capstone.md) | **Northwind Assist:** one deployable system with UI/API, RAG, tools, memory, evaluation, observability, auth, security, caching, CI/CD, and cost monitoring. | [Exercises](book/exercises/ch39-exercises.md) · [Solutions](book/solutions/ch39-solutions.md) |

## Projects and Libraries

Reusable packages built and imported across chapters:

| Package | Built in | Purpose |
|---|---|---|
| [`aie_core`](book/projects/aie_core) | Ch 3 | Provider-neutral LLM client and gateway: retries, deadlines, rate limits, fallbacks, caching, structured output, embeddings, cost, tracing. |
| [`ragkit`](book/projects/ragkit) | Ch 11 | Parsing, structure-aware chunking, metadata, ACLs, versioning, and BM25/hybrid retrieval for RAG. |
| [`toolkit`](book/projects/toolkit) | Ch 16 | Governed tool calling: registry, schemas, policy engine, approvals, idempotency, and sandboxed execution. |
| [`agentkit`](book/projects/agentkit) | Ch 19 | A bounded, event-sourced agent loop with typed state, budgets, termination reasons, verification, and replay. |
| [`memorykit`](book/projects/memorykit) | Ch 21 | Typed memory records with provenance, confidence, and expiry; tenant-scoped stores with hard delete. |
| [`evalkit`](book/projects/evalkit) | Ch 24 | Versioned eval datasets, a runner with lineage, deterministic checks, LLM judges, statistics, and a CI gate. |
| [`guardrails`](book/projects/guardrails) | Ch 27 | Layered checks at input, context, output, and tool stages; injection heuristics, PII and secret redaction. |
| [`reliability`](book/projects/reliability) | Ch 29 | Circuit breakers, bulkheads, queues, backpressure, load shedding, and graceful degradation for AI services. |

Plus [`shared-data`](book/projects/shared-data) (the synthetic Northwind corpus every project uses) and [`examples`](book/projects/examples) (per-chapter runnable examples).

## Appendices

- [Learning Roadmap](book/00-learning-roadmap.md) — reading paths, study method, graduation criteria
- [Appendix — Interview Preparation](book/appendix-c-interview-preparation.md)
- [Glossary](book/glossary.md)
- [References](book/references.md)
- [Coverage Matrix](book/coverage-matrix.md) — which chapter covers which competency

## Running the Code

All code targets **Python 3.11+** and runs **offline** in tests (fake model and embedding clients). Real providers are enabled through environment variables documented in each project's `.env.example`.

Setup uses [uv](https://docs.astral.sh/uv/). One script creates `.venv` at the repo root with every library and project installed in editable mode:

```bash
git clone https://github.com/XayyamSadigov/ai-engineering-book && cd ai-engineering-book
book/tools/setup_dev.sh                      # about a minute; creates .venv
source .venv/bin/activate
cd book/projects/p3-rag-assistant && pytest -q   # run one project from its own folder
cd - && book/tools/verify_code.sh            # compile and test everything (exit 1 on any failure)
```

Run tests from each project's folder, not from the repo root: the projects share module names such as `tests`, so one root-level pytest run cannot collect them all. `verify_code.sh` does the per-project loop for you.

> **Do not install the projects with plain `pip install -e .`.** pip ignores the `[tool.uv.sources]` table that points each project at its sibling libraries, so installs fail, and several library names (`ragkit`, `evalkit`, `toolkit`, `agentkit`, `guardrails`, `reliability`, `memorykit`) belong to unrelated packages on PyPI that pip would fetch instead.

To call a real model, copy a project's `.env.example` to `.env`, set `LLM_PROVIDER` and the provider's API key, and run the project's CLI or demo. Nothing else changes.

To build the website locally:

```bash
pip install "mkdocs<2" "mkdocs-material==9.*"
mkdocs serve
```

## FAQ

**Do I need a machine-learning background?**
No. The book teaches model internals only as far as they explain behavior you will see in production.

**Do I need an API key to follow along?**
No. Every test runs against fake model and embedding clients. Add a key only when you want to see real model output.

**Is this tied to one vendor or framework?**
No. Concepts are vendor-independent; examples name providers only as examples. Chapter 23 maps popular frameworks back to the primitives you build yourself.

**How long does it take?**
About 16 weeks at 6–8 hours per week on the full path, or 8 weeks on the accelerated path. See the [roadmap](book/00-learning-roadmap.md).

**Where are the answers to the exercises?**
In [`book/solutions/`](book/solutions), deliberately kept apart from the chapters.

## Acknowledgements

The topic map of this book was informed by the public [AI Engineering Course](https://github.com/amitshekhariitbhu/ai-engineering-course) by Amit Shekhar / Outcome School. This book is an independent work: it reorganizes the material for working software engineers, rewrites it, and adds complete runnable code, projects, a capstone, evaluation and security implementations, production engineering, system design cases, and exercises with solutions.

## License

- **Text** (all Markdown under `book/`): [CC BY 4.0](LICENSE-CONTENT.md) — share and adapt with attribution.
- **Code**: [MIT](LICENSE).

If this book helps you, consider giving the repository a ⭐ so others can find it.
