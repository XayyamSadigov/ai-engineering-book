# Chapter 7 examples: model selection and routing

A model catalog, an evaluation-driven selection harness, a capability-aware router with a
confidence cascade, and an offline cascade evaluator. Everything runs offline: two `FakeLLM`
instances from `aie_core` play a "small" and a "large" model on the 60 synthetic Northwind
tickets in `book/projects/shared-data/tickets.jsonl`. All prices, latencies, and model names
are illustrative.

```
catalog.py        ModelProfile, Requirements (derived from a CompletionRequest), capability gaps, ModelCatalog
catalog.json      the illustrative Northwind catalog as data
tasks.py          ticket-classification eval cases, label schema, scorer, confidence reader
fakes.py          make_small_model (keyword scorer) and make_large_model (near-oracle) as FakeLLM handlers
selection.py      run_selection: quality with Wilson CI, p50/p95 latency, cost, errors, Pareto front, choose()
router.py         Router: rules, optional EmbeddingRouteClassifier, cascade escalation, capability-aware fallbacks
cascade_eval.py   collect outcomes once, simulate any threshold, utility with misroute cost, calibration
config.py         RouterSettings from environment variables
demo.py           prints the selection table, cascade sweeps under three error costs, routing decisions
compose.py        Part II end to end: registry prompt, ContextBuilder, Router, ModelGateway, complete_structured
prompt_files/      the versioned prompt and alias map compose.py renders
test_ch07.py      offline tests
test_compose.py   offline tests for the composed path
```

## Run

From the repository root:

```bash
book/tools/setup_dev.sh                                              # once, creates .venv
.venv/bin/python -m pytest book/projects/examples/ch07 -q
cd book/projects/examples/ch07 && ../../../../.venv/bin/python demo.py
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_CATALOG_PATH` | unset | JSON catalog; the built-in illustrative catalog when unset |
| `ROUTER_MIN_CONFIDENCE` | `0.8` | escalate the small model's answer below this confidence |
| `ROUTER_LONG_CONTEXT_TOKENS` | `100000` | prompt plus output size that selects the long-context route |
| `ROUTER_CLASSIFIER_MIN_CONFIDENCE` | `0.5` | below this, the classifier abstains and the default route is used |
| `LLM_PROVIDER`, `LLM_MODEL`, API keys | see `aie_core` | used when fakes are replaced by real clients |

## Using real models

Build one `aie_core.ModelGateway` per catalog alias (retries and same-model fallbacks live there)
and pass them as the `clients` mapping to `RouterSettings().build_router(clients, confidence=...,
validator=...)`. The router sets `req.model` to the alias's pinned `model_id` and passes
`reasoning_effort` in `req.metadata` for models whose profile lists effort levels.
