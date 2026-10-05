# Chapter 23 examples — the primitive side of framework concepts

Plain Python, no framework installed or imported. Each module shows what a framework concept does
under the hood; the chapter maps the framework vocabulary onto it.

| File | Shows | Framework concept it mirrors |
|---|---|---|
| `runnable.py` | composable pipeline: `invoke` / `batch` / `stream`, `\|` sequences, dict fan-out, retry wrapper that logs attempts | LCEL runnables, `with_retry` |
| `signature.py` | typed prompt spec (`Signature`), a `Predict` module, a `BootstrapFewShot` optimizer driven by a metric | DSPy signatures, modules, optimizers |
| `ports.py` | domain Protocols (`Retriever`, `LLMClient`, `Tool`), a framework-object adapter, `RecordingLLM` / `ReplayLLM` fixtures | ports and adapters; testing with recorded fixtures |

Run the tests (offline, no API keys):

```bash
.venv/bin/python -m pytest book/projects/examples/ch23 -q
```

Dependencies: Python 3.11+, `pydantic>=2`, `pytest`. No `aie_core` import is required here; the
chapter text references `aie_core` names for the production equivalents.
