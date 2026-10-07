# Authoring Guide (binding for every chapter author)

Read this entire file before writing. It exists so that 39 chapters written by different authors read
as one book, share one code base, one running example, and one quality bar.

## 1. Audience, voice, language

- Reader: a strong software engineer (backend/full-stack, comfortable with Python, HTTP, databases,
  testing, Docker) who is new to AI engineering. Never explain programming basics. Always explain AI
  jargon the first time it appears in your chapter (one clause is enough).
- Voice: direct, concrete, technical. Short paragraphs. No marketing language, no hype, no filler, no
  "in today's fast-paced world". Do not repeat yourself inside a chapter. Do not restate what another
  chapter owns; cross-reference it ("see Chapter 12, Retrieval Engineering").
- Language: English. American spelling. Em dashes are allowed only in chapter titles ("Chapter 3 —
  Working with LLM APIs"); in prose use commas, colons, or separate sentences.
- Vendor neutrality: concepts must never depend on one vendor. Code uses the provider-neutral
  `aie_core` interfaces. When you mention concrete providers, say "for example". Never state current
  prices, context sizes, or model names as facts; if you need a number for a worked example, label it
  "illustrative". State a model or protocol release only when it is verified against its primary
  source (specification, changelog, official announcement), and date it ("as of 2026", "the 2026-07-28
  revision").
- Every important concept must, somewhere in its section, answer: what it is, why it exists, how it
  works, when to use it, when not to, alternatives, trade-offs, how to implement, how it fails, how to
  test/evaluate it, how it behaves in production. Weave these in; do not print them as an 11-item list.

## 2. Chapter template

```
# Chapter N — Title

One paragraph: what the reader will be able to do after this chapter, and which project/code it builds.

## Why this matters
## Mental model
## Core concepts            (as many ### subsections as needed)
## How it works
## Architecture             (Mermaid diagram(s))
## Implementation           (excerpts of the code on disk, see section 4)
## Code walkthrough
## Production considerations
## Common mistakes
## Failure modes
## Tradeoffs
## Evaluation and testing
## Before you ship          (chapters 3-34, 37, 38: 8-12 checkbox items)
## Exercises                (open with a "Start here" line naming the core set)
### Knowledge questions     (K1, K2, ...)
### Engineering questions   (E1, E2, ...)
### Practical exercises     (P1, P2, ...)
### Debugging exercises     (D1, D2, ...)  - describe a broken system/trace; reader diagnoses
## Key takeaways            (6-10 bullets)
## Further reading          (3-6 items from references.md)
```

The opener under the H1 is one or two sentences, a "You will be able to" list of 4-6 bullets, and one
line with prerequisites, code path and test command, and what the chapter builds.

Adapt the template when another structure teaches better (system-design chapters use the 10-step
method per case; the capstone uses a build log), but keep Why this matters, Exercises, and Key
takeaways in every chapter. No section may be a two-paragraph stub: either develop it or merge it.

Mental-model callouts use a blockquote: `> **Mental model:** Context is a budget, not a bucket.`
Use the ten book-wide mental models (analysis/02-competency-map.md) where they genuinely apply.

Exercises contain **no answers**. Write the answers to `book/solutions/chNN-solutions.md` with the
same identifiers (K1, E1, P1, D1 ...). Solutions for practical exercises describe the expected
implementation and acceptance criteria; solutions for debugging exercises name the root cause and the
telemetry that reveals it. Aim for 4-6 knowledge, 3-4 engineering, 3-4 practical, 2-3 debugging items.

## 3. Diagrams

Use fenced ```mermaid blocks. Prefer `flowchart LR/TD`, `sequenceDiagram`, `stateDiagram-v2`. Keep
node labels short, no HTML, no special characters that break Mermaid (avoid parentheses inside labels
unless quoted: `A["retrieve (k=50)"]`). Each diagram must show a mechanism (data flow, control flow,
trust boundary, state transitions), not decorate. Chapters with architecture content need at least
two diagrams. Mark trust boundaries explicitly when relevant (`subgraph Untrusted`).

## 4. Code requirements

- Python 3.11+. Type hints everywhere. pydantic v2 for schemas and settings. `httpx` for HTTP.
  `pytest` for tests. FastAPI for services. PostgreSQL + pgvector where a database is needed, with an
  in-memory/NumPy fallback so tests run without infrastructure. Redis optional. No pseudocode except
  when explaining an algorithm, and then label the block `# pseudocode`.
- Code on disk is complete and executable; the chapter teaches from excerpts. Start each code block
  with a path comment on the first line, for example
  `# path: book/projects/p3-rag-assistant/rag/retrieval/hybrid.py`. A listing longer than about 80 lines
  becomes an excerpt of the 20-80 lines that carry the idea, labeled
  `# path: ... (excerpt; full file on disk)`, with elisions marked `# ...`. Every line in an excerpt must
  exist verbatim in the file on disk. Short files may be shown whole.
- Every project and every library module needs: directory tree, `pyproject.toml` with dependencies,
  configuration via environment variables (documented in a table and an `.env.example`),
  implementation, tests (runnable offline with fakes), and run instructions (`uv`/`pip` commands,
  `docker compose up` where applicable). Tests must pass with `pytest` without network access or API
  keys. Mark tests that need real providers with `@pytest.mark.integration` and skip them by default.
- Shared library: all code depends on `aie_core` (section 6). Do not reinvent LLM clients, retries,
  tracing, or settings; import them. Projects declare `aie_core` as a path dependency:
  `aie-core = { path = "../aie_core", editable = true }` under `[tool.uv.sources]` plus a plain
  `aie-core` entry in dependencies, and the README shows `pip install -e ../aie_core` as the fallback.
- Keep modules small and layered: `domain/` (pure logic, no I/O), `adapters/` (providers, DB), `api/`
  (FastAPI), `eval/` (datasets and evaluators), `tests/`. Favor explicit code over clever code; the
  reader must be able to trace every call.
- Secrets never appear in code or docs. Use `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `LLM_PROVIDER`,
  `LLM_MODEL`, `EMBEDDING_MODEL`, `DATABASE_URL`, `REDIS_URL` consistently.
- Where a step is genuinely too long to show in full (e.g., a 400-line parser), show the public
  interface and the critical function bodies, write the whole file to disk, and say so.

## 5. The running example: Northwind Assist

Northwind is a fictional mid-size company (about 4,000 employees, two business units that act as
tenants: `retail` and `logistics`). Throughout the book we build **Northwind Assist**, an internal
assistant for employees and support staff:

- Knowledge: HR policies, IT runbooks, product documentation, past incident reports, support tickets.
  Documents carry ACLs (`groups: ["all"]`, `["hr"]`, `["it-oncall"]`, tenant tags).
- Structured work: extract fields from invoices and support tickets (Project 1), classify tickets.
- Tools: `lookup_employee`, `search_tickets`, `create_ticket`, `draft_reply`, `send_reply` (approval
  required), `get_service_status`, `query_metrics` (read-only SQL over a semantic layer).
- Agents: incident research (Project 5), parallel research with verification (Project 6).
- Non-functional targets used in worked examples: p95 time-to-first-token under 2 s for RAG answers,
  p95 completion under 8 s, zero cross-tenant leakage, cost ceiling per answer "illustrative".

Use these names consistently. Sample documents live in `book/projects/shared-data/` (policies,
runbooks in Markdown, a few synthetic tickets and invoices in JSON/CSV). Authors who need sample data
create it there if it does not exist yet, keeping files small (under 50 KB each) and clearly synthetic.

## 6. The shared library `aie_core` (fixed public API)

Location: `book/projects/aie_core/` (package `aie_core`). Chapter 3 builds and explains it. Every other
author imports these names exactly. If you need something that is missing, implement it inside your
own project and note it; do not change `aie_core`'s public names.

```python
# aie_core.llm.types
class Role(str, Enum): SYSTEM="system"; USER="user"; ASSISTANT="assistant"; TOOL="tool"
class ContentPart(BaseModel): type: Literal["text","image_url"]; text: str|None; image_url: str|None
class ToolCall(BaseModel): id: str; name: str; arguments: dict[str, Any]
class Message(BaseModel):
    role: Role; content: str | list[ContentPart] = ""; tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None; name: str | None = None
    @classmethod system(text) / user(text) / assistant(text) / tool(tool_call_id, content)
class ToolSpec(BaseModel): name: str; description: str; parameters: dict[str, Any]   # JSON Schema
class Usage(BaseModel): input_tokens: int = 0; output_tokens: int = 0; cached_input_tokens: int = 0
class CompletionRequest(BaseModel):
    messages: list[Message]; model: str | None = None; temperature: float = 0.0
    max_tokens: int = 1024; tools: list[ToolSpec] | None = None
    tool_choice: Literal["auto","none","required"] | str = "auto"
    response_schema: dict[str, Any] | None = None   # JSON Schema for structured output
    stop: list[str] | None = None; metadata: dict[str, Any] = {}
    timeout_s: float | None = None
class Completion(BaseModel):
    message: Message; usage: Usage; finish_reason: str; model: str; provider: str; latency_ms: float
    raw: dict[str, Any] | None = None
    @property text -> str            # convenience: message.content as text
    @property tool_calls -> list[ToolCall]
class StreamEvent(BaseModel):
    type: Literal["text_delta","tool_call_delta","usage","done","error"]
    text: str | None = None; tool_call: ToolCall | None = None; usage: Usage | None = None
    error: str | None = None

# aie_core.llm.client
class LLMClient(Protocol):
    provider: str
    def complete(self, req: CompletionRequest) -> Completion: ...
    def stream(self, req: CompletionRequest) -> Iterator[StreamEvent]: ...
    async def acomplete(self, req: CompletionRequest) -> Completion: ...
    async def astream(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]: ...

# aie_core.llm.errors
class LLMError(Exception); class RateLimitError(LLMError); class TimeoutError(LLMError)
class ProviderUnavailableError(LLMError); class InvalidRequestError(LLMError)
class ContentFilterError(LLMError); class MalformedResponseError(LLMError)
# each error has .retryable: bool and .retry_after_s: float | None

# aie_core.llm.providers
class OpenAICompatibleClient(LLMClient)   # base_url, api_key, default_model; works for OpenAI, vLLM, Ollama...
class AnthropicClient(LLMClient)          # api_key, default_model
class FakeLLM(LLMClient)                  # scripted: FakeLLM(responses=[...]) returns in order; records requests
                                          # also FakeLLM(handler=callable(req)->Completion|str|dict)

# aie_core.llm.gateway
class RetryPolicy(BaseModel): max_attempts=3; base_delay_s=0.5; max_delay_s=8.0; jitter=True
class RateLimiter: token bucket (requests/min and tokens/min)
class ResponseCache(Protocol): get(key)/set(key, completion, ttl_s); InMemoryResponseCache
class ModelGateway(LLMClient):
    def __init__(self, primary: LLMClient, fallbacks: list[LLMClient] = (), retry: RetryPolicy = ...,
                 rate_limiter: RateLimiter | None = None, cache: ResponseCache | None = None,
                 max_concurrency: int = 16, pricing: PricingTable | None = None,
                 tracer: Tracer | None = None, default_timeout_s: float = 60.0)
    # complete/stream with retries, fallback chain, caching, cost accounting, tracing spans
class PricingTable: cost_usd(model, usage) -> float   # loaded from a dict; illustrative prices

# aie_core.llm.tokens
def count_tokens(text: str, model: str | None = None) -> int     # tiktoken if installed else heuristic
def count_message_tokens(messages: list[Message], model=None) -> int

# aie_core.llm.structured
def complete_structured(client: LLMClient, req: CompletionRequest, schema: type[BaseModel],
                        max_repair_attempts: int = 2) -> tuple[BaseModel, Completion]
    # uses response_schema when the provider supports it, validates with pydantic, re-asks with the
    # validation error on failure, raises MalformedResponseError when attempts are exhausted

# aie_core.embeddings
class EmbeddingClient(Protocol):
    model: str; dimensions: int
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
class OpenAICompatibleEmbeddings(EmbeddingClient); class FakeEmbeddings(EmbeddingClient)  # deterministic hashing
class CachedEmbeddings(EmbeddingClient)            # wraps another client with an in-memory/Redis cache
def cosine_similarity(a, b) -> float; def normalize(v) -> list[float]; def top_k(query, matrix, k) -> list[tuple[int,float]]

# aie_core.observability
class Span: name, attributes dict, start/end, set_attribute(), record_exception()
class Tracer:
    def span(self, name: str, **attributes) -> ContextManager[Span]
class JsonlTracer(Tracer)      # writes one JSON object per span to a file
class OTelTracer(Tracer)       # OpenTelemetry bridge if opentelemetry-sdk is installed
class NoopTracer(Tracer)
def get_tracer() -> Tracer     # from settings

# aie_core.settings
class Settings(BaseSettings):  # env-driven
    llm_provider: Literal["openai","anthropic","fake"] = "fake"; llm_model: str = "fake-model"
    llm_base_url: str | None; openai_api_key: SecretStr | None; anthropic_api_key: SecretStr | None
    embedding_provider: Literal["openai","fake"] = "fake"; embedding_model: str = "fake-embedding"
    database_url: str | None; redis_url: str | None; trace_sink: Literal["none","jsonl","otel"] = "none"
    trace_path: str = "traces.jsonl"; request_timeout_s: float = 60.0
def make_llm_client(settings: Settings | None = None) -> LLMClient      # factory honoring provider
def make_embedding_client(settings: Settings | None = None) -> EmbeddingClient
```

Conventions for using it in chapters: build a `CompletionRequest`, call `gateway.complete(req)`,
read `completion.text` or `completion.tool_calls`. Tests inject `FakeLLM` and `FakeEmbeddings`.
`FakeEmbeddings` must make semantically similar strings *not* necessarily similar (it is hashing);
when a test needs controllable similarity, use `FakeEmbeddings(vocabulary=...)` which embeds by
bag-of-words over a provided vocabulary, so "refund policy" is close to "refund policy deadline".

## 7. Ownership map (what each chapter owns; others reference)

| Topic | Owner |
|---|---|
| Five-layer stack, decision ladder, mental models | Ch 1 |
| Tokens, sampling, context window mechanics, KV cache basics, model types | Ch 2 |
| `aie_core` client/gateway/retries/streaming/structured helper | Ch 3 |
| Prompt registry and prompt tests | Ch 4 |
| ContextBuilder, compaction | Ch 5 |
| Schema design, repair loops, extraction pipelines, Project 1 | Ch 6 |
| Router and cascades | Ch 7 |
| Embedding math and non-RAG uses | Ch 8 |
| Vector stores, pgvector, Project 2 | Ch 9 |
| Minimal RAG and failure catalogue | Ch 10 |
| Parsers and chunkers | Ch 11 |
| BM25, hybrid, rerank, query transforms | Ch 12 |
| Evidence packing, citations, abstention | Ch 13 |
| Retrieval/answer metrics and stage isolation | Ch 14 |
| Indexing pipeline, ACL, tenancy, caching, Project 3 | Ch 15 |
| Tool registry, policy, idempotency, sandbox, Project 4 | Ch 16 |
| Workflow engine, decision framework | Ch 17 |
| MCP | Ch 18 |
| AgentRuntime, events, termination, replay | Ch 19 |
| Architecture patterns, Project 5 | Ch 20 |
| Memory stores | Ch 21 |
| Multi-agent, Project 6 | Ch 22 |
| Frameworks | Ch 23 |
| evalkit core, judges, statistics | Ch 24 |
| Task-specific evaluators, CI gate | Ch 25 |
| Threat model, injection catalogue | Ch 26 |
| Guardrail implementations | Ch 27 |
| Reference architecture diagrams | Ch 28 |
| Retry/circuit breaker/queues/admission control implementations | Ch 29 |
| CostModel, latency budgets, caching layers | Ch 30 |
| Tracing schema, OTel, debugging playbook | Ch 31 |
| Clean architecture, versioning, flags, CI/CD | Ch 32 |
| Fine-tuning | Ch 33 |
| Serving math and self-hosting | Ch 34 |
| System design method and cases | Ch 35-36 |
| Advanced retrieval | Ch 37 |
| Durable agents, coding agents, voice | Ch 38 |
| Capstone | Ch 39 |

## 8. Source usage

The source book's text is at `book/analysis/source-extracted-text.txt` (page markers `===== PAGE N`).
Mine your assigned modules for principles, worked numbers, failure modes, and case details. Rewrite in
your own words; never paste source paragraphs. Ignore the templated "Study lens / Production test /
Mini exercise / Mastery check" paragraphs. Where the source is thin or absent, add the material (the
brief in `04-table-of-contents.md` lists what must be covered).

## 9. Quality checklist before you finish a chapter

- [ ] Every must-cover item from the brief is present and developed (not a bullet).
- [ ] At least one complete, runnable code artifact; files written to disk; tests pass offline.
- [ ] Diagrams show mechanisms; trust boundaries marked where relevant.
- [ ] Failure modes and evaluation sections are concrete (named failure, how it shows in telemetry,
      how to test for it).
- [ ] Production considerations include latency, cost, security, and operations angles.
- [ ] Exercises in four categories, no answers inline; solutions file written.
- [ ] Cross-references use chapter numbers from `04-table-of-contents.md`.
- [ ] No vendor pricing/model facts stated as truth; no post-mid-2026 claims.
- [ ] No filler, no repeated paragraphs, no stub sections.
