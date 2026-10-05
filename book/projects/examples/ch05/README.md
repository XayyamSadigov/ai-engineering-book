# Chapter 5 examples: context engineering

A `context` package that assembles the model's input under a token budget, keeps
conversation state compact without losing exact facts, keeps the cacheable prefix stable,
and measures position effects. Everything runs offline with `FakeLLM`.

```
context/
  items.py          ContextItem, Trust, Section: the typed unit of context
  filters.py        RequestScope, acl_filter (permission), min_score_filter (relevance)
  labels.py         render_item, untrusted-data blocks, tag neutralization
  builder.py        ContextBuilder, BudgetPolicy, SectionLimits, BuildResult, manifest
  state.py          ConversationState, Fact, Turn, LLMSummarizer, compaction guards,
                    StateSnapshot + InMemoryStateStore (compare-and-set on version)
  layout.py         shared_prefix_tokens, lint_stable_items, PrefixStabilityTracker
  experiments/
    position.py     lost-in-the-middle sweep + SimulatedPositionalReader
demo.py             one Northwind Assist turn: compaction, build, manifest
tests/              offline tests
```

## Run

```bash
# from the book root, using the shared virtualenv
uv pip install --python .venv/bin/python -e book/projects/aie_core   # or: pip install -e book/projects/aie_core
.venv/bin/python -m pytest book/projects/examples/ch05 -q

cd book/projects/examples/ch05
../../../../.venv/bin/python demo.py
../../../../.venv/bin/python -m context.experiments.position --trials 60          # simulated reader
LLM_PROVIDER=openai LLM_MODEL=... OPENAI_API_KEY=... \
  ../../../../.venv/bin/python -m context.experiments.position --live            # your real model
```

## Configuration

| Variable | Default | Used by |
|---|---|---|
| `LLM_PROVIDER` | `fake` | `--live` position sweep (`aie_core.settings.make_llm_client`) |
| `LLM_MODEL` | `fake-model` | same |
| `LLM_BASE_URL`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | unset | same |
| `TRACE_SINK`, `TRACE_PATH` | `none`, `traces.jsonl` | pass `aie_core.observability.get_tracer()` to `ContextBuilder` |

See `.env.example`. Dependencies: Python 3.11+, `aie-core` (path dependency), `pydantic>=2`, `pytest`.
