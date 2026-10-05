# aie_core

The shared library of the AI Engineering book: one provider-neutral way to call a language
model, plus the reliability layer every chapter needs (retries, deadlines, rate limiting,
fallbacks, caching, cost accounting, tracing), structured output with repair, embeddings,
and environment-driven settings. Chapter 3 explains the design.

## Install

```bash
# from the book root, into the shared virtualenv
uv pip install --python .venv/bin/python -e book/projects/aie_core
# or plain pip
pip install -e book/projects/aie_core
# optional extras
pip install -e "book/projects/aie_core[tokens,otel,dev]"
```

Projects depend on it as a path dependency:

```toml
[project]
dependencies = ["aie-core"]

[tool.uv.sources]
aie-core = { path = "../aie_core", editable = true }
```

## Layout

```
aie_core/
  __init__.py
  settings.py            Settings (pydantic-settings) + make_llm_client / make_provider_client / make_embedding_client
  observability.py       Span (trace_id, parent_span_id via ContextVar), Tracer, JsonlTracer, OTelTracer, NoopTracer, InMemoryTracer, get_tracer, current_span
  embeddings.py          EmbeddingClient, OpenAICompatibleEmbeddings, FakeEmbeddings, CachedEmbeddings, math
  llm/
    types.py             Role, Message, ToolCall, ToolSpec, Usage, CompletionRequest, Completion, StreamEvent
    client.py            LLMClient protocol, collect_stream
    errors.py            LLMError taxonomy + map_http_error
    tokens.py            count_tokens, count_message_tokens
    structured.py        complete_structured (schema + pydantic validation + repair loop)
    gateway.py           ModelGateway, RetryPolicy, RateLimiter, ResponseCache, InMemoryResponseCache, PricingTable
    providers/
      openai_compat.py   OpenAICompatibleClient (OpenAI, vLLM, Ollama, ...)
      anthropic.py       AnthropicClient
      fake.py            FakeLLM
      _sse.py, _http.py  SSE parser and httpx plumbing
tests/                   offline tests (httpx.MockTransport, FakeLLM, injected clocks and sleeps)
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `fake` | `openai`, `anthropic`, or `fake` |
| `LLM_MODEL` | `fake-model` | model name sent to the provider |
| `LLM_BASE_URL` | unset | base URL for OpenAI-compatible servers or a proxy |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | unset | credentials |
| `EMBEDDING_PROVIDER` | `fake` | `openai` or `fake` |
| `EMBEDDING_MODEL` | `fake-embedding` | embedding model name |
| `DATABASE_URL`, `REDIS_URL` | unset | used by later chapters |
| `TRACE_SINK` | `none` | `none`, `jsonl`, `otel`, or `memory` (an `InMemoryTracer`; read spans from `client.tracer`) |
| `TRACE_PATH` | `traces.jsonl` | file for the JSONL tracer |
| `REQUEST_TIMEOUT_S` | `60` | default deadline per `complete()` |
| `LLM_MAX_ATTEMPTS` | `3` | retry budget per client |
| `LLM_MAX_CONCURRENCY` | `16` | in-flight requests per gateway |
| `LLM_REQUESTS_PER_MINUTE`, `LLM_TOKENS_PER_MINUTE` | unset | enable the token-bucket limiter |
| `LLM_RESPONSE_CACHE` | `false` | enable the exact-match cache (temperature 0 only) |
| `LLM_FALLBACK_MODEL` | unset | second model, same provider, used on retryable failures |

## Usage

```python
from aie_core import CompletionRequest, Message, make_llm_client

client = make_llm_client()  # FakeLLM by default; a ModelGateway around a real client when configured
req = CompletionRequest(messages=[Message.system("You are Northwind Assist."), Message.user("Hi")])
completion = client.complete(req)
print(completion.text, completion.usage)

for event in client.stream(req):
    if event.type == "text_delta":
        print(event.text, end="", flush=True)
```

Structured output:

```python
from pydantic import BaseModel
from aie_core.llm.structured import complete_structured

class Ticket(BaseModel):
    category: str
    priority: int

ticket, completion = complete_structured(client, req, Ticket)
```

Testing with the fakes:

```python
from aie_core.llm.providers import FakeLLM
from aie_core.embeddings import FakeEmbeddings

llm = FakeLLM(responses=["first answer", {"category": "billing", "priority": 2}])
emb = FakeEmbeddings(vocabulary=["refund", "policy", "deadline", "vpn"])
```

Notes on accounting: on a response-cache hit `Completion.raw["cost_usd"]` is `0.0` and the price the provider would have charged is in `raw["avoided_cost_usd"]` (same attributes on the span), so summing `cost_usd` never double counts. `CachedEmbeddings` keys by `space_fingerprint` (provider, model, dimensions, instruction, text-prep version) plus text, so vectors from different embedding spaces never collide. The fingerprint resolves to the innermost real client when wrappers are nested. Every embedding client raises `MalformedResponseError` when a provider returns fewer vectors than inputs or inconsistent indices; nothing is dropped silently. `Completion.truncated` is true when `finish_reason == "length"`, and `complete_structured` raises `TruncatedOutputError` immediately in that case instead of running the repair loop.

## Tests

```bash
cd book/projects/aie_core
python -m pytest -q
```

Everything runs offline. Tests that need a real provider are marked `integration` and skipped.

Multi-tenant caching: `cache_key` ignores `metadata` except `metadata["cache_scope"]` (for example `"tenant:retail"`), which partitions the response cache when permissions are enforced outside the prompt. Unscoped keys are unchanged. Pass `require_cache_scope=True` to `ModelGateway` on any gateway shared across tenants: requests without a scope are then never cached. The async concurrency semaphore is kept per event loop, so a gateway can be reused across `asyncio.run` calls.

Composition: `make_provider_client(settings, model=None)` returns the bare provider adapter for building your own `ModelGateway`. `make_llm_client(settings, tracer=..., wrap=...)` accepts an injected tracer (use the application's single tracer instead of one built from `TRACE_SINK`); `wrap=True` wraps the fake provider in a gateway too, `wrap=False` returns the bare adapter. Defaults are unchanged.
