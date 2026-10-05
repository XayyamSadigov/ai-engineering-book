# Chapter 10 examples: minimal RAG and the naive failure catalogue

A complete retrieve-pack-generate pipeline over `aie_core` and the shared Northwind corpus,
plus a script that reproduces the seven naive-RAG failure modes deterministically.

```
minimal_rag.py      ingest -> chunk -> index -> retrieve -> pack -> generate -> validate
failure_modes.py    seven demos returning FailureCase(stage, observed, detected, signal, fixed)
fixtures/           hr-compensation-bands.md (acl_groups ["hr"]) for the permission-leak demo
test_ch10.py        offline tests (FakeLLM, FakeEmbeddings)
```

Run from the repository root:

```bash
uv pip install --python .venv/bin/python -e book/projects/aie_core   # or: pip install -e book/projects/aie_core
.venv/bin/python -m pytest book/projects/examples/ch10 -q
cd book/projects/examples/ch10
../../../../.venv/bin/python minimal_rag.py "How far in advance must I request a two-week vacation?"
../../../../.venv/bin/python failure_modes.py
```

Configuration (all optional; defaults run offline):

| Variable | Default | Effect here |
|---|---|---|
| `LLM_PROVIDER` | `fake` | `fake` uses an extractive scripted model; `openai` or `anthropic` use a real one via `ModelGateway` |
| `LLM_MODEL` | `fake-model` | model name passed to the provider |
| `LLM_BASE_URL` | unset | OpenAI-compatible endpoint (vLLM, Ollama, a proxy) |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | unset | credentials for real providers |
| `EMBEDDING_PROVIDER` | `fake` | `fake` uses a hashed bag of content words; `openai` uses real embeddings |
| `EMBEDDING_MODEL` | `fake-embedding` | embedding model name |

Dependencies: Python 3.11+, `aie-core` (path dependency), `pydantic>=2`, `numpy`, `pytest`.
The demos in `failure_modes.py` always use the offline fakes so their output is reproducible.
