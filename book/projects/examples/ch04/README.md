# Chapter 4 examples: prompts as versioned, tested contracts

A `prompts/` package with sandboxed templates that delimit untrusted data, a registry of
versioned prompt files with content hashes and a lock file, prompt identity on tracing
spans, and a regression harness that compares prompt versions. Everything runs offline.

```
ch04/
  prompts/
    template.py       PromptTemplate, VariableSpec, Untrusted, sanitize_untrusted, taint check
    registry.py       PromptSpec (front matter), PromptVersion, RenderedPrompt, PromptRegistry
    tracing.py        traced_complete: `prompt.call` span with prompt.id/version/hash
    regression.py     Case, Assertion, run_suite, compare, Comparison.gate, render_report
    judge.py          LLMJudge (groundedness, itself a registered prompt)
    examples.py       FewShotExample, select_examples (similarity + label diversity + budget)
    rollout.py        PromptRollout: sticky canary split, last-known-good alias reload
  prompt_files/
    ticket.classify/1.0.0.md 1.1.0.md 1.2.0.md
    assist.answer/1.0.0.md
    judge.groundedness/1.0.0.md
    schemas/*.json    output schemas referenced from front matter
    aliases.toml      prod / canary pointers
  prompts.lock        id@version -> sha256, checked in CI
  cases/              ticket_classify.jsonl, assist_answer.jsonl
  demo_model.py       simulated router used when LLM_PROVIDER=fake
  prompts_cli.py      verify | lock | render | compare
  tests/
```

## Install and run

```bash
# from the book root, using the shared virtualenv
uv pip install --python .venv/bin/python -e book/projects/aie_core -e book/projects/examples/ch04
# or: pip install -e ../../aie_core && pip install -e .
.venv/bin/python -m pytest book/projects/examples/ch04 -q

cd book/projects/examples/ch04
../../../../.venv/bin/python prompts_cli.py verify
../../../../.venv/bin/python prompts_cli.py compare ticket.classify \
    --baseline 1.0.0 --candidate 1.1.0 --cases cases/ticket_classify.jsonl
```

YAML front matter works when `pyyaml` is installed (`pip install -e ".[yaml]"`); the shipped
prompt files use TOML, which needs nothing beyond the standard library.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `fake` | `fake` uses the simulated router in `compare`; `openai`/`anthropic` use `aie_core.make_llm_client()` |
| `LLM_MODEL`, `LLM_BASE_URL`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | see aie_core | provider settings |
| `TRACE_SINK`, `TRACE_PATH` | `none`, `traces.jsonl` | where `prompt.call` and `llm.complete` spans go |
| `PROMPT_DIR` | `prompt_files` | registry root |
| `PROMPT_LOCK` | `prompts.lock` | lock file path |

The simulated router is not a model. It reads decision rules from the system prompt and
applies them literally, so prompt edits visibly change results; its numbers say nothing
about how a real model would score these prompts.
