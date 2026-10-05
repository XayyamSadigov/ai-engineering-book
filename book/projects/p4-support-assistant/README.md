# Project 4: Northwind support assistant with governed tools

Built in Chapter 16 (Tool Calling). A chat assistant for Northwind service desk staff that can
look up employees, search tickets, check service status, create tickets, draft replies, and send
replies. The model only *proposes* tool calls. Every call goes through the `toolkit` executor:
schema validation, policy (permissions, recipient allowlist, rate limits, group denies),
idempotency, human approval bound to the exact arguments, retries for transient errors only,
result truncation, and an audit trail.

## Layout

```
p4-support-assistant/
  pyproject.toml  .env.example  Dockerfile  README.md
  data/injected_tickets.jsonl     security fixture: a ticket carrying an indirect injection
  support_assistant/
    config.py                     AssistantSettings (P4_* environment variables)
    domain/
      directory.py                Employee, StaffAccount (-> ToolContext), tenant-scoped Directory
      tickets.py                  TicketStore over shared-data tickets.jsonl (via shared_data.py)
      services.py                 StatusBoard (fault injection), DraftStore, Outbox (the external system)
    tools.py                      six tools, argument models, build_registry, build_policy
    prompts.py                    system prompt
    adapters/demo_llm.py          keyword-driven model so the service runs without an API key
    wiring.py                     composition root: build_container(settings, llm=...)
    assistant.py                  SupportAssistant: sessions, tool loop, approval channel
    api/app.py                    FastAPI: /chat, /approvals, /approvals/{id}/approve|reject, /healthz
  tests/
    test_scenarios.py             acceptance scenarios with scripted FakeLLM turns
    test_api.py                   HTTP flows, tenant isolation, session ownership, demo model
```

## Tools

| Tool | Side effect | Permission | Approval | Idempotent | Notes |
|---|---|---|---|---|---|
| `lookup_employee` | read | `directory:read` | no | yes | tenant-scoped; never returns HR fields; 20/min per user |
| `search_tickets` | read | `tickets:read` | no | yes | results labeled as requester-written data |
| `get_service_status` | read | `status:read` | no | yes | upstream 503s are retried |
| `create_ticket` | reversible_write | `tickets:write` | P1 by non-lead only | no | duplicate suppression + reconcile |
| `draft_reply` | reversible_write | `replies:draft` | no | no | recipient allowlist |
| `send_reply` | external | `replies:send` | always | no | recipient allowlist; contractors denied; 30/h |

Staff accounts (synthetic): `ana` (retail agent), `sam` (retail lead, can approve), `lee`
(logistics, read-only), `kai` (logistics contractor; may not send).

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER`, `LLM_MODEL`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `LLM_BASE_URL` | `fake` | model settings from `aie_core` |
| `P4_SHARED_DATA_DIR` | `../shared-data` | where `shared_data.py` and `tickets.jsonl` live |
| `P4_EXTRA_TICKETS_PATH` | `data/injected_tickets.jsonl` | local ticket fixtures (set empty to disable) |
| `P4_ALLOWED_RECIPIENT_DOMAINS` | `["northwind.example"]` | JSON list; send/draft allowlist |
| `P4_ALLOWED_RECIPIENTS` | `[]` | JSON list of individually allowed addresses |
| `P4_APPROVAL_TTL_S` | `900` | approval expiry |
| `P4_FOUR_EYES` | `false` | `true` forbids approving your own send |
| `P4_IDEMPOTENCY_DB` | `:memory:` | SQLite file to keep duplicate suppression across restarts |
| `P4_AUDIT_LOG_PATH` | unset | JSONL audit file; unset keeps events in memory |
| `P4_MAX_ROUNDS` | `6` | tool-loop round limit |
| `P4_TOOL_MAX_ATTEMPTS` | `3` | attempts for transient tool errors |
| `P4_LOOKUP_RATE_PER_MINUTE`, `P4_SEND_RATE_PER_HOUR` | `20`, `30` | per-user rate limits |
| `P4_DEMO_LLM` | `true` | with `LLM_PROVIDER=fake`, use the keyword demo model |

## Run

```bash
# from book/projects
uv pip install --python ../../.venv/bin/python -e aie_core -e toolkit -e "p4-support-assistant[dev]"
# or: pip install -e ../aie_core -e ../toolkit -e ".[dev]"   (from this directory)

cd p4-support-assistant
python -m pytest -q                                   # offline, no keys
uvicorn support_assistant.api.app:app --reload        # http://localhost:8000/docs
```

Try it with the offline demo model:

```bash
curl -s localhost:8000/chat -H 'X-User-Id: ana' -H 'content-type: application/json' \
  -d '{"message": "what is the status of the vpn"}'
curl -s localhost:8000/chat -H 'X-User-Id: ana' -H 'content-type: application/json' \
  -d '{"message": "send TCK-2026-0001 to priya.raman@northwind.example: Card payments are restored."}'
curl -s localhost:8000/approvals -H 'X-User-Id: sam'
curl -s -X POST localhost:8000/approvals/<id>/approve -H 'X-User-Id: sam' -H 'content-type: application/json' -d '{}'
```

Docker (build from `book/projects`):

```bash
docker build -f p4-support-assistant/Dockerfile -t northwind/support-assistant .
docker run --rm -p 8000:8000 -v p4-state:/app/state northwind/support-assistant
```

`X-User-Id` stands in for an identity set by an authenticating gateway; never expose this service
directly. With a real model, set `LLM_PROVIDER` and the key; tests never need one.
