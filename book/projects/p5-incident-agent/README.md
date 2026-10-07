# Project 5: Northwind incident-research agent

An agentic workflow for on-call engineers, built in Chapter 20. Given an alert, it plans an
investigation, runs each step as a small bounded agent with one tool, replans when a
deterministic rule detects a deviation, writes a cited incident report, revises it in an
evaluator-optimizer loop until a deterministic Definition of Done passes and a rubric judge is
satisfied, and posts it to a channel only after a human approves.

Everything runs offline. With `LLM_PROVIDER=fake` (the default) a scripted stand-in model plays
every role; set a real provider to drive it with a model.

## What it demonstrates

| Concern | Where |
|---|---|
| Planner-executor with typed plans, plan validation, one repair round | `incident_agent/agent.py` (`plan`, `validate`) |
| One bounded agentkit `AgentRuntime` per step, least-privilege tools | `agent.py` (`execute`) |
| Replanning only on named deviations, with a replan budget | `domain/deviation.py` |
| Retrieval over shared-data runbooks and incidents: ragkit chunking + BM25 + ACL | `adapters/corpus.py` |
| Fake metrics with anomaly flags and dependency health; deploy history | `adapters/telemetry.py`, `data/` |
| Evidence ledger built from tool data, not parsed from text | `agent.py` (`execute`), `tools.py` |
| Deterministic Definition of Done: sections, citations, grounding, runbook exists | `domain/dod.py` |
| Rubric judge (evalkit `LLMJudge`) that only reads DoD-passing drafts | `judge.py` |
| Evaluator-optimizer with best-so-far, plateau stop, round budget | `agent.py` (`write_and_evaluate`) |
| Approval-gated, idempotent posting via agentkit's approval pause and resume | `agent.py` (`Publisher`), `tools.py` |
| Global model-call ceiling across all roles | `adapters/metering.py` |
| Trajectory tests, agentkit replay, counterfactual replay, cassette replay | `tests/` |
| CLI and FastAPI endpoint over one service class | `cli.py`, `api.py`, `service.py` |

## Layout

```
p5-incident-agent/
  pyproject.toml  .env.example  Dockerfile  README.md
  data/
    alerts.json          two synthetic alerts (Trackline latency, RoutePilot latency)
    telemetry.json       metric series, service dependencies, deploy history
  incident_agent/
    config.py            P5Settings (P5_ env vars), user directory
    domain/
      models.py          Alert, Plan, PlanStep, Evidence, StepRecord, Problem, RoundRecord, Investigation, Status
      report.py          required sections, claim extraction, citation syntax
      dod.py             check_report(): the deterministic Definition of Done
      deviation.py       replanning rules
    adapters/
      corpus.py          KnowledgeBase: ragkit load + MarkdownSectionChunker + BM25, ACL-filtered
      telemetry.py       alerts, metrics with anomaly rule, deploys
      channel.py         InMemoryChannel, JsonlChannel (idempotent by key)
      store.py           investigation records (memory, JSON files)
      metering.py        MeteredLLM with a hard call ceiling
      cassette.py        RecordingLLM / ReplayLLM for whole-pipeline replay
      scripted.py        deterministic stand-in model for every role
    tools.py             research tools (read-only) and post_report (EXTERNAL, approval required)
    judge.py             incident-report rubric on evalkit
    agent.py             IncidentResearchAgent, Publisher
    service.py           IncidentService: investigate, decide, get
    cli.py  api.py
  tests/                 39 offline tests
```

## Install and run

```bash
# from the book root, into the shared virtualenv
uv pip install --python .venv/bin/python -e book/projects/aie_core -e book/projects/agentkit \
    -e book/projects/ragkit -e book/projects/evalkit -e book/projects/p5-incident-agent
# or, from this directory: pip install -e ../aie_core -e ../agentkit -e ../ragkit -e ../evalkit -e .

cd book/projects/p5-incident-agent
python -m pytest -q                                   # offline, about one second

# CLI (state goes to $P5_STATE_DIR, default ./.p5-state)
p5 investigate ALR-2026-0914-01 --user oncall-logistics
p5 show <investigation-id> --report
p5 approve <investigation-id> --user ic-logistics --reason "matches dashboards"
p5 replay <investigation-id>                          # re-runs each step from its event log; no tool executes

# HTTP API
uvicorn incident_agent.api:app --port 8000
curl -s -X POST localhost:8000/investigations -H 'X-User: oncall-logistics' \
     -H 'content-type: application/json' -d '{"alert_id": "ALR-2026-0914-01"}'
curl -s -X POST localhost:8000/investigations/<id>/decision -H 'X-User: ic-logistics' \
     -H 'content-type: application/json' -d '{"approve": true, "reason": "ok"}'

# Docker (build context is book/projects)
docker build -f Dockerfile -t northwind/incident-agent ..   # run from this directory
docker run -p 8000:8000 -v p5-state:/app/state northwind/incident-agent
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER`, `LLM_MODEL`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `LLM_BASE_URL` | `fake` | model for planner, executor, writer, judge (aie_core); `fake` uses the scripted model |
| `TRACE_SINK`, `TRACE_PATH` | `none` | aie_core tracing (spans `incident.investigate`, `agent.run`, `agent.step`, `agent.tool`) |
| `P5_STATE_DIR` | `.p5-state` | investigation records, agentkit JSONL event logs, channel log |
| `P5_CHANNEL` | `#incidents` | where approved reports are posted |
| `P5_MAX_PLAN_STEPS` | 8 | executed steps per investigation, across replans |
| `P5_MAX_REPLANS` | 2 | deviations that may trigger a new plan; one more fails the investigation |
| `P5_MAX_REVISIONS` | 3 | writer rounds in the evaluator-optimizer loop |
| `P5_MIN_IMPROVEMENT` | 0.05 | score gain below which a round counts as a plateau |
| `P5_JUDGE_PASS_SCORE` | 4 | rubric score (1-5) the judge requires |
| `P5_JUDGE_MODEL` | unset | a different model for the judge |
| `P5_STEP_MAX_STEPS`, `P5_STEP_MAX_TOOL_CALLS` | 3, 2 | budget of each step agent |
| `P5_MAX_LLM_CALLS` | 60 | hard ceiling on model calls per investigation, all roles |
| `P5_CHUNK_TOKENS` | 250 | ragkit section-chunk size |
| `P5_SHARED_DATA_DIR`, `P5_DATA_DIR` | `../shared-data`, `./data` | corpora and fake telemetry |

The template is `.env.example` (copy it to `.env`). Users and their groups are a fixed directory in
`config.py`, standing in for an identity provider; the API reads identity from `X-User` only.

## Statuses

`awaiting_approval` (DoD passed; judge verdict attached as advisory), `needs_revision` (DoD never
passed within the revision budget; cannot be approved), `published`, `rejected`, `failed`
(planning failed, replan or call budget exhausted). Only `awaiting_approval` accepts a decision;
anything else returns 409.

## Limits and honest caveats

- The scripted model makes a deliberately flawed first draft so the loop runs on every offline
  execution. Real models fail less predictably; keep the trajectory tests and add recorded
  cassettes from real runs as regression cases.
- The anomaly rule is a ratio against a median baseline, enough to make deviations happen. Real
  systems use your monitoring stack's detectors.
- Investigations run synchronously inside the request. For production, enqueue them and let the
  API return 202 with a status URL (Chapter 29); durable multi-worker resumption is Chapter 38.
