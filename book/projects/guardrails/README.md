# guardrails

Layered guardrails for LLM applications, built in Chapter 27 of the AI Engineering book. One pipeline
runs ordered checks at four stages (input, context, output, tool); every check returns allow, flag,
redact or block with a reason and a score, and declares whether it fails closed or open.

## Install

```bash
# from the book root, into the shared virtualenv
uv pip install --python .venv/bin/python -e book/projects/aie_core -e book/projects/guardrails
# fallback
pip install -e ../aie_core && pip install -e .
```

## Layout

```
guardrails/
  pipeline.py     Stage, Action, FailMode, Verdict, GuardContext, Check, GuardrailPipeline
  input.py        SizeLimitCheck, DenyPatternCheck, score_injection, InjectionHeuristicCheck, LLMInjectionClassifier
  context.py      neutralize_untrusted, wrap_untrusted, render_untrusted_context, ContextSanitizerCheck
  pii.py          detect_pii (email, phone, card+Luhn, IBAN+mod97, IP), redact_pii, PIIVault, PIIRedactionCheck
  secrets.py      detect_secrets (known formats + entropy), redact_secrets, SecretsCheck
  output.py       SchemaCheck, UrlAllowlistCheck, CanaryCheck, CitationCheck, ActiveContentCheck, escape_html, ANSWER_PANE_CSP
  moderation.py   Moderator protocol, KeywordModerator, LLMModerator, ModerationCheck
  tenancy.py      assert_tenant_scope, tenant_guarded, scoped_cache_key, TenantScopedCache
  tools.py        ToolPolicyCheck, ToolRule, approval_token, argument constraints,
                  guard_tool_call, rehydrate_arguments, rehydrate_kinds, token_tolerant_schema
  telemetry.py    RedactingTracer (scrubs on write and at span end, before any backend), scrub
  presets.py      rag_pipeline, agent_pipeline (Northwind Projects 3 and 4)
  eval/           datasets, end-to-end red team, measure (FP rate, bypass rate)
data/             labeled benign and attack cases
tests/            offline tests
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER`, `LLM_MODEL`, API keys | `fake` | used only by the model-based checks, through `aie_core` |
| `TRACE_SINK`, `TRACE_PATH` | `none` | tracer for guardrail spans; wrap it in `RedactingTracer` (wrap the tracer, do not use it as a sink: a sink runs after backends such as OpenTelemetry have copied the attributes) |
| `GUARDRAILS_CH26_DIR` | `../examples/ch26` | location of the Chapter 26 attack corpus |

## Usage

```python
from guardrails import GuardContext, PIIVault, agent_pipeline

ctx = GuardContext(tenant="retail", user_id="u1", groups=frozenset({"all"}))
ctx.vault = PIIVault(tenant="retail", scope_id="req-1")
pipe = agent_pipeline(canaries=["NW-CANARY-..."])
res = pipe.check_input("My card 4111 1111 1111 1111 was charged twice", ctx)
res.text  # card replaced by a reversible token
```

## Tests and measurement

```bash
cd book/projects/guardrails
python -m pytest -q
python -m guardrails.eval.measure          # FP and bypass table, end-to-end effect rate
python -m guardrails.eval.measure --json   # for a CI gate; non-zero exit if any effect bypasses
```

## Tool boundary: PII tokens and Project 4

`presets.support_tool_rules()` uses Project 4's argument names and limits (`query`, `subject`,
`ticket_id`, `to`, ...); `tests/test_p4_alignment.py` runs Project 4's real tool specs through it.
The input guardrail replaces e-mail addresses with tokens such as `<PII:email:3f2a9c1b07>`, which fail
a tool's e-mail pattern. Call `guard_tool_call(pipeline, call, ctx)` at the tool boundary: it re-hydrates
each rule's `rehydrate_args` from the request's `PIIVault` (e-mail only by default), runs the TOOL stage
on the re-hydrated call, and returns the call to execute. Give the model `token_tolerant_schema(spec)` so
strict structured-output modes accept a token; validate the re-hydrated call against the original.
