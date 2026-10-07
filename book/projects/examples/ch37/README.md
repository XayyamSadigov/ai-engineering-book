# Chapter 37 examples: advanced retrieval patterns

Small, complete implementations of the patterns in Chapter 37, built on `aie_core` (LLM client,
structured output, token counting) and `ragkit` (parsed documents, section chunks), over the shared
Northwind corpus. Each module runs offline with a scripted `FakeLLM` and switches to a real model
when `LLM_PROVIDER` is set.

```
corpus.py              Principal + ACL rule, ragkit section chunks, LexicalIndex (TF-IDF), FullTextIndex (SQLite FTS5)
agentic_rag.py         AgenticRAG: bounded search/answer/abstain loop, EvidenceLedger, citation check, sufficiency gate
agentic_rag_runtime.py RuntimeAgenticRAG: the same ledger and gate on agentkit's AgentRuntime (event log, replay)
graphrag.py            GraphRAG: LLM extraction -> EntityResolver -> graph store -> communities -> local/global queries
toc_navigation.py      TocNavigator: vectorless retrieval by LLM navigation of heading trees, with id validation
maxsim.py              ColBERT-style MaxSim versus mean-pooled single vectors, plus index size arithmetic
map_reduce.py          MapReduceSummarizer (hierarchical, budgeted, span provenance); RecursiveReader (input as environment)
code_search.py         CodeIndex: AST symbol index, identifier-aware lexical search, callers, import graph, AST chunks
long_context_cost.py   Long context vs RAG vs hybrid cost and prefill estimates (illustrative prices)
tests/                 offline tests for all of the above
```

## Run

From the repository root:

```bash
uv pip install --python .venv/bin/python -e book/projects/aie_core -e book/projects/agentkit -e book/projects/ragkit
# optional: uv pip install --python .venv/bin/python networkx   (GraphRAG falls back to a dict graph)
.venv/bin/python -m pytest book/projects/examples/ch37 -q

cd book/projects/examples/ch37
../../../../.venv/bin/python agentic_rag.py
../../../../.venv/bin/python graphrag.py "What does the PayBridge Adapter depend on?"
../../../../.venv/bin/python toc_navigation.py "How many unused PTO days can I carry over?"
../../../../.venv/bin/python maxsim.py
../../../../.venv/bin/python map_reduce.py "How long did INC-2025-1142 last?"
../../../../.venv/bin/python code_search.py "retry backoff jitter"
../../../../.venv/bin/python long_context_cost.py
```

Plain pip works too: `pip install -e ../../aie_core -e ../../agentkit -e ../../ragkit numpy pytest`.

## Configuration

Copy `.env.example` and adjust. All variables are optional.

| Variable | Default | Effect here |
|---|---|---|
| `LLM_PROVIDER` | `fake` | `fake` uses the scripted controllers in each module; `openai` or `anthropic` use a real model |
| `LLM_MODEL` | `fake-model` | model name passed to the provider |
| `LLM_BASE_URL` | unset | OpenAI-compatible endpoint (vLLM, Ollama, a proxy) |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | unset | credentials for real providers |
| `TRACE_SINK` | `none` | `jsonl` or `otel` to record spans (see Chapter 31) |

Tests always use the fakes, so they pass without network access or keys.
