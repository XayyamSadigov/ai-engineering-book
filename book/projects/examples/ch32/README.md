# Chapter 32 examples: engineering practices for an AI feature

Northwind ticket triage built with clean architecture: a pure domain, a use case behind
ports, adapters for the LLM, prompts, tools, and HTTP, and the practices around it:
version manifest on every trace, deterministic feature flags with a kill switch,
record/replay of provider HTTP, property and contract tests, prompt snapshots, an offline
eval gate, experiment arithmetic, and CI pipelines for GitHub Actions and GitLab.

```
ch32/
  northwind_triage/
    domain/            models.py (vocabulary, business rules), parsing.py (total parser)
    application/       ports.py (ClassifierPort, PromptVersion, ...), triage_service.py
    adapters/          llm_classifier.py (anti-corruption layer over aie_core), prompt_store.py,
                       tools.py (create_ticket contract), http_api.py, simulated_model.py
    prompt_files/      triage.classify/1.0.0.md, 1.1.0.md, prompts.lock
    version_manifest.py  flags.py  record_replay.py  experiments.py
    config.py          AppSettings (TRIAGE_*) + aie_core Settings, validated at startup
    composition.py     the composition root
    eval_gate.py       offline eval gate (console script `triage-eval`)
  eval/                golden.jsonl (60 synthetic tickets), baseline.json
  flags/prod.json      example flag file
  tests/               property, contract, snapshot, architecture, replay, service, CI tests
    cassettes/         committed provider recordings (recorded from a local mock in this book)
    snapshots/         rendered prompts
    tool_schemas.lock
  ci/                  github-actions.yml, gitlab-ci.yml
  adr/                 0001-template.md, 0002-classifier-port-and-recorded-fixtures.md
  Dockerfile  pyproject.toml  .env.example
```

## Install and test

```bash
# from the book root
uv pip install --python .venv/bin/python -e book/projects/aie_core -e "book/projects/examples/ch32[dev]"
# or: pip install -e ../../aie_core && pip install -e ".[dev]"
.venv/bin/python -m pytest book/projects/examples/ch32 -q
```

All tests run offline. The one test that re-records a cassette against a real provider is
marked `integration` and skipped by default.

## Run

```bash
cd book/projects/examples/ch32
python -m northwind_triage.eval_gate --prompt-version 1.1.0 --report reports/eval.json
python -m northwind_triage.experiments sample-size --baseline 0.80 --mde 0.03
python -m northwind_triage.adapters.prompt_store            # verify prompts.lock
UPDATE_SNAPSHOTS=1 pytest tests/test_prompt_snapshots.py    # accept a prompt rendering change
TRIAGE_FLAGS_PATH=flags/prod.json TRIAGE_PROMPT_TREATMENT_VERSION=1.1.0 \
  uvicorn --factory northwind_triage.composition:create_app_from_env --port 8080
```

With `LLM_PROVIDER=fake` the classifier is a keyword heuristic (`simulated_model.py`); its
accuracy numbers describe the heuristic, not any model.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER`, `LLM_MODEL`, keys | `fake`, `fake-model` | from `aie_core.Settings`; control model |
| `TRIAGE_ENVIRONMENT` | `dev` | `dev`, `test`, `staging`, `prod`; staging/prod forbid the fake provider |
| `TRIAGE_APP_VERSION`, `TRIAGE_GIT_SHA` | `0.0.0-dev`, `unknown` | code identity; git sha required in staging/prod |
| `TRIAGE_PROMPT_CONTROL_VERSION` | `1.0.0` | prompt served to control |
| `TRIAGE_PROMPT_TREATMENT_VERSION` | unset | prompt served to the `treatment` flag variant |
| `TRIAGE_MODEL_CANDIDATE` | unset | model served to the `candidate` flag variant |
| `TRIAGE_FLAGS_PATH` | unset | JSON flag file; required in prod |
| `TRIAGE_CLASSIFIER_TIMEOUT_S` | `8` | per-call deadline |
| `TRIAGE_DATASET_VERSION`, `TRIAGE_EVALUATOR_VERSION` | unset | recorded in the manifest when known |
| `CASSETTE_MODE` | `replay` | `replay`, `record`, `auto` for record/replay fixtures |
