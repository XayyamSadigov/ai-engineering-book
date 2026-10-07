# Chapter 30 examples: performance and cost engineering

Offline, runnable code for Chapter 30. Depends only on `aie_core` (see `book/projects/aie_core`).

| File | What it holds |
|---|---|
| `attribution.py` | `AttributingTracer`, `bind()`: stamp tenant, request id and feature onto every span |
| `latency.py` | `LatencyBudget`, `StageTimeouts`, `LatencyTracker`: allocate, enforce and audit a latency SLO |
| `caching.py` | `EmbeddingCache`, `RetrievalCache`, `ScopedResponseCache`, `SemanticCache`, `lint_cache_key` |
| `cost.py` | `CostScenario`, `CostModel` (per successful task, mixes, monthly), `chargeback` from trace JSONL |
| `budgets.py` | `SpendPolicy`, `SpendGuard`, `BudgetedClient`, `TaskTokenBudget` |
| `parallel.py` | `fan_out` under a deadline, `Prefetcher`, `MicroBatcher` |
| `demo.py` | prints the chapter's optimization ladder |
| `test_ch30.py` | offline tests |

## Run

```bash
# from the book root
uv pip install --python .venv/bin/python -e book/projects/aie_core   # or: pip install -e book/projects/aie_core
.venv/bin/python -m pytest book/projects/examples/ch30 -q
cd book/projects/examples/ch30 && ../../../../.venv/bin/python demo.py
```

## Configuration

Nothing here reads the environment directly. Pricing is passed in as an `aie_core.PricingTable`;
load it from configuration (`PricingTable.from_json`) in a real service. Spend policies are plain
dataclasses; load them from your tenant configuration store. Every price in the code is illustrative.
