# reliability

System-level reliability primitives for AI services, built in Chapter 29 of the AI Engineering
book. `aie_core.ModelGateway` (Chapter 3) protects one model call with retries, backoff,
fallback, and a client-side rate limiter. This package protects everything around it: chains,
agents, tools, retrieval, and background workers.

## Install

```bash
# from the book root, into the shared virtualenv
uv pip install --python .venv/bin/python -e book/projects/aie_core -e "book/projects/reliability[dev]"
# or plain pip
pip install -e ../aie_core && pip install -e ".[dev]"
```

## Layout

```
reliability/
  errors.py      DeadlineExceeded, CircuitOpenError, Overloaded, BulkheadFullError, AdmissionRejected, is_retryable
  clock.py       ManualClock for deterministic tests
  deadline.py    Deadline: one end-to-end budget, child budgets, cancellation, header propagation
  retry.py       RetryBudget, call_with_retry / acall_with_retry for non-LLM dependencies
  circuit.py     CircuitBreaker (rolling window, half-open probes), registry, CircuitBreakerClient
  bulkhead.py    Bulkhead / AsyncBulkhead / Bulkheads: per-workload concurrency pools
  admission.py   AdmissionController: tenant quotas, priority classes, wait-based load shedding, degrade
  degrade.py     DegradePolicy and DegradedPlan: reduced, minimal, static, read-only
  malformed.py   complete_with_recovery: validate, repair once, fallback, salvage, default
  chain.py       run_chain: partial-failure semantics for multi-step chains
  queue.py       JobQueue protocol, InMemoryJobQueue, RedisJobQueue (Lua, atomic)
  worker.py      Worker loop, graceful shutdown, at-least-once, once() for side effects
  slo.py         SLI evaluation, burn rate, multi-window paging rule
  chaos.py       FaultPlan, ChaosLLM, ChaosFunction: scheduled outages, 429s, latency, malformed output
examples/
  ticket_chain.py  Northwind ticket triage chain wired with every primitive
  worker_main.py   worker process entry point (Redis or in-memory demo)
tests/             offline; Redis contract tests run on fakeredis, real Redis tests are `integration`
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `REDIS_URL` | unset | Redis for `RedisJobQueue`; the worker falls back to in-memory without it |
| `WORKER_VISIBILITY_TIMEOUT_S` | `120` | lease length; a dead worker's job reappears after this |
| `WORKER_POLL_INTERVAL_S` | `1.0` | idle sleep between empty polls |
| `JOB_MAX_ATTEMPTS` | `5` | attempts before a job is dead-lettered |
| `ADMISSION_CAPACITY` | `32` | concurrent executions one replica serves within SLO |
| `ADMISSION_MAX_QUEUE` | `32` | interactive requests allowed to wait beyond capacity |

## Run

```bash
python -m examples.worker_main --demo                      # in-memory, processes two jobs and exits
REDIS_URL=redis://localhost:6379/0 python -m examples.worker_main
docker run --rm -p 6379:6379 redis:7-alpine                # a local Redis for the above
```

## Tests

```bash
cd book/projects/reliability
python -m pytest -q                                       # offline
REDIS_URL=redis://localhost:6379/15 python -m pytest -q -m integration   # real Redis (uses and flushes db 15)
```
