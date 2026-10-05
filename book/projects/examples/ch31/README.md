<!-- path: book/projects/examples/ch31/README.md -->
# Chapter 31: Observability for AI systems

Tracing schema, instrumentation, OpenTelemetry wiring, a trace store, debugging queries,
metric definitions, alert rules, and a label-join script. Builds on `aie_core.observability`
(`AITracer` is an `aie_core` `Tracer`, so it plugs into `ModelGateway(tracer=...)`).

```
ch31/
  semconv.py              span names, attribute keys, error taxonomy, GenAI aliases
  instrument.py           AITracer, CapturePolicy, redact, stage helpers and decorators
  otel_setup.py           build_provider, OTelAITracer, tracer_from_env
  trace_store.py          SpanRecord, TraceTree, TraceStore (trees, joins, filters)
  analysis.py             stage triage, version compare, retrieved-not-cited, latency, cost, trajectories
  metrics.py              metric registry and alert evaluator
  alerts.yaml             alert rules
  dashboards.md           metric definitions per dashboard panel
  join_signals.py         CLI: join eval results and feedback to traces
  incident_sim.py         synthetic Northwind workload through the real instrumentation
  incident_walkthrough.py the debugging playbook, end to end
  tests/                  offline tests
```

## Run

```bash
# from the book root, with the shared virtualenv
uv pip install --python .venv/bin/python -e book/projects/aie_core pyyaml
cd book/projects/examples/ch31
../../../../.venv/bin/python -m pytest -q
../../../../.venv/bin/python incident_walkthrough.py
../../../../.venv/bin/python incident_sim.py /tmp/incident
../../../../.venv/bin/python join_signals.py --traces /tmp/incident/traces.jsonl \
    --evals /tmp/incident/eval_results.jsonl --feedback /tmp/incident/feedback.jsonl --out /tmp/joined.jsonl
```

OTLP export needs `pip install -e ".[otlp]"`; tests use the in-memory exporter.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `TRACE_SINK` | `none` | `none`, `jsonl`, or `otel` |
| `TRACE_PATH` | `traces.jsonl` | JSONL file when `TRACE_SINK=jsonl` |
| `OTEL_SERVICE_NAME` | `northwind-assist` | resource `service.name` |
| `OTEL_EXPORTER` | `otlp` | `otlp`, `console`, `memory`, `none` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | unset | collector endpoint (standard OTel variable) |
| `OTEL_SAMPLE_RATIO` | `1.0` | head sampling ratio, parent-based |
| `EMIT_GENAI_ALIASES` | `false` | also write `gen_ai.*` keys (check current conventions) |
| `CAPTURE_MODE` | `hashed` | `off`, `hashed`, `redacted`, `full` |
| `CAPTURE_SAMPLE_RATE` | `0` | fraction of traces upgraded to redacted content |
| `CAPTURE_SALT` | `change-me` | HMAC key for content and user hashes; keep it in a secret store |
| `DEPLOY_ENV`, `APP_VERSION` | `dev` | resource attributes |
