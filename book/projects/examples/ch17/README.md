# Chapter 17 examples: workflows and orchestration

Plain-Python workflow engine plus the Northwind ticket-triage workflow built twice
(fixed pipeline and graph with conditional edges). Everything runs offline with a
scripted `FakeModel`; no API keys.

```
workflow_engine.py   Graph, Step, RetryPolicy, Checkpointer, pause/resume (handle bound to state hash),
                     replay, step_key() idempotency keys, optional per-node tracer spans
patterns.py          sequence, branch, fan_out, map_reduce, retry, fallback
triage_domain.py     TriageState (pydantic), the five step functions, FakeModel
triage_pipeline.py   classify -> policy -> draft -> validate -> approve/send as a function
triage_graph.py      the same workflow as a Graph with routers and a paused approval node
compare.py           orchestration overhead measured separately from model time
test_ch17.py         offline tests
```

Run from the repository root:

```bash
.venv/bin/python -m pytest book/projects/examples/ch17 -q
cd book/projects/examples/ch17 && ../../../../.venv/bin/python compare.py
```

Dependencies: Python 3.11+, `pydantic>=2`, `pytest`. In the full book stack the
`FakeModel` callable is replaced by `aie_core.llm.gateway.ModelGateway.complete`
(Chapter 3); the step functions do not change.
