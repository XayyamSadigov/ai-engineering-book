# Project 1: LLM-powered structured extraction API

A FastAPI service that turns Northwind's supplier invoices and support tickets into
validated, typed records. Chapter 6 (Structured Output and Extraction) explains every
design decision; this README is the operator's view.

Pipeline per document:

1. **Classify** the document type (invoice, support ticket, other) with abstention, unless
   the caller passes `doc_type`.
2. **Extract** a draft with `aie_core.llm.structured.complete_structured`: native schema
   mode when the provider supports it, pydantic validation, re-ask on schema errors.
3. **Post-process** deterministically: parse money, dates, rates, currency; locate every
   evidence quote in the source text; check that each quote contains its value.
4. **Validate** business rules: line sums, subtotal plus tax equals total, date sanity,
   required PO, currency enum, grounded entities.
5. **Route**: accept, one targeted repair re-ask, or the human review queue.

Batch mode runs the same pipeline with bounded concurrency and per-document error isolation.

## Layout

```
p1-extraction-api/
  pyproject.toml            dependencies; aie-core as a path dependency
  .env.example              every environment variable with its default
  Dockerfile                build from book/projects (see below)
  extraction_api/
    config.py               AppSettings (EXTRACT_* variables)
    wiring.py               composition root: settings -> client, queue, service
    domain/                 pure logic, no I/O
      common.py             DocumentType, Route, RuleViolation, EvidenceSpan
      normalize.py          money, quantity, rate, date, currency parsers; quote locator
      invoice.py            InvoiceDraft (wire schema), Invoice (domain), build_invoice
      ticket.py             SupportTicketDraft, SupportTicket, pattern entities, build_ticket
      rules.py              check_invoice, check_ticket, check_evidence
      routing.py            RoutingPolicy, field scores, decide()
    application/            use cases
      prompts.py            versioned prompts, document fencing, repair prompt
      classifier.py         DocumentClassifier (verbalized or self-consistency confidence)
      metering.py           MeteredClient: calls and tokens per document, repairs included
      pipelines.py          one DocTypeSpec per document type
      service.py            ExtractionService: extract(), extract_batch()
      ports.py              ReviewQueue protocol, ReviewItem, ReviewResolution
    adapters/
      review_queue.py       InMemoryReviewQueue, SQLiteReviewQueue
      replay_llm.py         offline stand-in model answering from shared-data
    api/
      app.py                FastAPI app: /extract, /extract/batch, /review, /healthz
      auth.py               API keys, roles, tenant scoping
      schemas.py            batch and resolve bodies
    eval/
      dataset.py            gold loader: shared-data/invoices.jsonl or local fixture
      metrics.py            per-field precision, recall, exact match; line items
      calibration.py        threshold choice, reliability bins, ECE
      run_eval.py           CLI: field table, routing quality, cost, calibration
      fixtures/invoices_sample.jsonl
  tests/                    offline: FakeLLM, ReplayLLM, InMemoryTracer, TestClient
```

## Configuration

Model and tracing variables are read by `aie_core.Settings`; service variables use the
`EXTRACT_` prefix. Copy `.env.example` to `.env` to change them.

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `fake` | `openai`, `anthropic`, or `fake` (offline ReplayLLM) |
| `LLM_MODEL` | `fake-model` | model name sent to the provider |
| `LLM_BASE_URL` | unset | OpenAI-compatible server or proxy |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | unset | credentials, never committed |
| `LLM_MAX_ATTEMPTS`, `LLM_MAX_CONCURRENCY` | `3`, `16` | gateway retry budget and in-flight cap |
| `LLM_REQUESTS_PER_MINUTE`, `LLM_TOKENS_PER_MINUTE` | unset | token-bucket rate limits |
| `TRACE_SINK`, `TRACE_PATH` | `none`, `traces.jsonl` | `jsonl` writes one span per line |
| `EXTRACT_ACCEPT_THRESHOLD` | `0.80` | minimum document score to auto-accept; set from `run_eval` calibration |
| `EXTRACT_CLASSIFY_THRESHOLD` | `0.70` | below this the classifier abstains |
| `EXTRACT_CLASSIFY_SAMPLES` | `1` | above 1, confidence is vote agreement (costs N calls) |
| `EXTRACT_MAX_SCHEMA_REPAIRS` | `2` | re-asks on invalid JSON or schema errors |
| `EXTRACT_MAX_RULE_REPAIRS` | `1` | targeted re-asks on repairable rule violations |
| `EXTRACT_REQUIRE_PO` | `true` | Northwind pays nothing without a purchase order |
| `EXTRACT_DOLLAR_MEANS` | `USD` | ISO code for a bare `$` |
| `EXTRACT_MAX_DOCUMENT_CHARS` | `50000` | larger documents get HTTP 413 before any model call |
| `EXTRACT_MAX_REQUEST_BYTES` | `8000000` | larger bodies get HTTP 413 before parsing; cap it at the proxy too |
| `EXTRACT_MAX_BATCH_SIZE` | `100` | documents per `/extract/batch` call |
| `EXTRACT_DOCUMENT_DEADLINE_S` | `120` | wall-clock budget for all model calls of one document; exhausted before extraction means 503, before a repair means review with `REPAIR_SKIPPED_DEADLINE` |
| `EXTRACT_CALL_TIMEOUT_S` | `45` | per-call timeout, capped by the remaining deadline |
| `EXTRACT_API_KEYS` | `[]` (auth off) | JSON list of `{key, principal, role, tenant}`; roles `submitter`, `reviewer`, `admin`; a tenant-bound key sees only its tenant's review items |
| `EXTRACT_BATCH_CONCURRENCY` | `8` | documents in flight per batch |
| `EXTRACT_REVIEW_BACKEND` | `memory` | `memory` or `sqlite` |
| `EXTRACT_REVIEW_DB_PATH` | `data/review.db` | SQLite file |
| `EXTRACT_SHARED_DATA_DIR` | `../shared-data` | sample data for ReplayLLM |
| `EXTRACT_GOLD_PATH` | shared-data invoices | gold file for `run_eval` |

## Run

```bash
cd book/projects/p1-extraction-api
uv venv && uv pip install -e ".[dev]"          # resolves aie-core from ../aie_core
# or, with plain pip:
pip install -e ../aie_core && pip install -e ".[dev]"

python -m pytest -q                             # offline, no keys needed
python -m extraction_api.eval.run_eval          # field metrics over shared-data invoices
uvicorn extraction_api.api.app:app_factory --factory --reload
```

Try it (with the default offline ReplayLLM):

```bash
python - <<'EOF'
import json, httpx
inv = json.loads(open("../shared-data/invoices.jsonl").readline())
r = httpx.post("http://127.0.0.1:8000/extract", json={"document_id": inv["id"], "text": inv["text"]})
print(r.json()["route"], r.json()["data"]["total"])
EOF
curl -s http://127.0.0.1:8000/review | python -m json.tool
curl -s -X POST http://127.0.0.1:8000/review/<review_id>/resolve \
     -H 'content-type: application/json' -d '{"decision":"approve","reviewer":"ap-clerk-1"}'
```

With a real provider:

```bash
export LLM_PROVIDER=openai LLM_MODEL=<model> OPENAI_API_KEY=...   # or anthropic, or LLM_BASE_URL for vLLM/Ollama
python -m extraction_api.eval.run_eval --out data/report.json     # measure before serving
```

Docker (the build context is `book/projects` so the image contains `aie_core` and samples):

```bash
cd book/projects
docker build -f p1-extraction-api/Dockerfile -t northwind/extraction-api .
docker run --rm -p 8000:8000 -v "$PWD/p1-extraction-api/data:/app/data" northwind/extraction-api
```

## API

| Method and path | Purpose |
|---|---|
| `POST /extract` | one document: `{document_id?, text, doc_type?, tenant?}` -> `ExtractionResult` |
| `POST /extract/batch` | `{documents: [...]}` -> summary plus one item per document; infrastructure failures come back as `error`, not as review items |
| `GET /review?status=pending&limit=50` | review items (`pending`, `approved`, `corrected`, `rejected`, `all`) |
| `GET /review/{id}` | one review item, including the source text and the machine's attempt |
| `POST /review/{id}/resolve` | `{decision: approve or correct or reject, reviewer, corrected_data?, note?}`; corrections are validated against the domain schema; 409 if already resolved |
| `GET /healthz` | liveness and review-queue counts (unauthenticated) |

With `EXTRACT_API_KEYS` set, every endpoint except `/healthz` requires
`Authorization: Bearer <key>`: `/extract*` needs role `submitter` or `admin`, `/review*` needs
`reviewer` or `admin`. A tenant-bound key stamps its tenant on submitted documents (403 if the
body names another tenant), lists only its tenant's review items, and gets 404 for any other
tenant's item. The recorded reviewer is the key's principal, not the `reviewer` field in the body.
Provider errors map to 503 with `Retry-After` when retryable and to 502 when not (for example a
schema the provider rejects), so clients do not retry a call that cannot succeed.

Every response carries `X-Request-ID` (the caller's, if it is a safe token, else a new one),
and every result carries `prompt_version`, `llm_calls`, and token usage.
