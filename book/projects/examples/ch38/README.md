# Chapter 38 examples: durable and long-running agents

Extensions to `agentkit` (Chapter 19) and `toolkit` (Chapter 16). Nothing here re-implements
the agent loop, the idempotency store, or the sandbox; it adds durability, interrupts,
long-horizon context, and two harness case studies on top of them. Offline, no API keys.

```
durable.py          SqliteEventStore (runs projection), LeaseManager with fencing, DurableRunner
                    (start/resume/recover), ReconcilingTool (outcome reconciliation over
                    toolkit.SQLiteIdempotencyStore), SimulatedCrash, CrashingStore
interrupts.py       InterruptManager: approvals with escalation chain and expiry (default deny),
                    timers and webhook events as approvals granted by the clock or the world,
                    wait_until / wait_for_event tools, tick() for the scheduler
ledger.py           TaskLedger + update_ledger tool (ledger lives in the event log), harness_facts,
                    digest_observation, compact_messages, CompactingLLM (LLMClient decorator)
coding_harness.py   Workspace, unified-diff parser/applier, CodingTools (search_code, read_file,
                    apply_patch, run_tests in toolkit.SandboxRunner, show_diff), coding_dod
skills.py           SKILL.md folders, progressive disclosure, select_skills, SkillLock, audit_skill,
                    load_skill / read_skill_resource tools
voice_gate.py       TurnGate (partials never unlock side effects), PlaybackController (barge-in),
                    LatencyBudget
fakes.py            FakeClock, FakeTicketSystem (crash before/after commit, optional lookup)
skills/             three sample Northwind skills
demo.py             crash after a remote commit, recover on another worker, one ticket
test_ch38_*.py      offline tests
```

## Run

```bash
# from the book root, shared virtualenv
uv pip install --python .venv/bin/python -e book/projects/aie_core -e book/projects/agentkit -e book/projects/toolkit
.venv/bin/python -m pytest book/projects/examples/ch38 -q
cd book/projects/examples/ch38 && ../../../../.venv/bin/python demo.py          # in-memory
../../../../.venv/bin/python demo.py /tmp/runs.db                                # inspect the SQLite file after
```

Fallback without uv: `pip install -e ../../aie_core -e ../../agentkit -e ../../toolkit pytest`.

## Configuration

These examples read no environment variables. The model client comes from `aie_core`
(`LLM_PROVIDER`, `LLM_MODEL`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) when you replace the scripted
`FakeLLM` with `aie_core.make_llm_client()`. The coding harness's sandbox passes only `PATH`,
`LANG`, `LC_ALL`, and `TZ` to the test process.

## Production notes

SQLite stands in for PostgreSQL. The SQL ports directly (`INSERT ... ON CONFLICT`, conditional
`UPDATE ... WHERE fence = ?`); use `SELECT ... FOR UPDATE SKIP LOCKED` for the timer table when
several schedulers run `tick()`. `toolkit.SandboxRunner` bounds time and resources but does not
isolate the filesystem or network on macOS; run coding agents in a container or microVM.
