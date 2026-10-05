# Project 6: a multi-agent research team, where justified

A Northwind policy-research team that answers cross-cutting employee questions such as
"What do I need to do before travelling abroad with a company laptop?". A supervisor decomposes
the question, researchers run in parallel with read-only document search, a verifier checks
every claim against the passage it cites, and the supervisor writes the answer. Every agent is an
`agentkit.AgentRuntime` (Chapter 19); this project adds only coordination: contracts, budgets,
spawn control, verification, trace propagation, and a benchmark against single-agent baselines.

Chapter 22 explains the design. The ADR at the end of this file records why the team exists and
what the benchmark says about it, including where it does not pay off.

## Layout

```
p6-research-team/
  pyproject.toml  README.md  env.example  Dockerfile
  research_team/
    contracts.py      TaskEnvelope, ResultEnvelope, BudgetSlice, Plan, ResearchFindings, Claim, ...
    corpus.py         shared-data docs split into section passages, BM25 search, ACL filter
    tools.py          search_docs and read_passage (read-only, principal from ToolContext)
    ledger.py         TeamBudget, BudgetLedger (reserve/settle, spawn cap, duplicates), TeamLog
    tracing.py        PropagatingTracer: trace id and parent span id on every agent span
    roles.py          prompts, tool sets, Definitions of Done; AgentFactory builds one runtime per task
    verification.py   verifier agent AND deterministic guard; degrades to the guard alone
    checks.py         deterministic_support, find_conflicts, DoD verifiers for JSON outputs
    team.py           ResearchTeam: plan, admit, dispatch in parallel, verify, follow up, synthesize
    baseline.py       SingleAgent (optionally with the verification step)
    render.py         deterministic answer rendering and citation parsing
    scripted.py       offline policy standing in for the model, simulated latency
    text.py           tokenization, numbers, quantities, Markdown-to-sentence units
    config.py, cli.py, __main__.py
    eval/
      questions.jsonl 8 questions: 6 cross-cutting, 2 single-policy controls
      scoring.py      doc recall, fact recall, citation validity, 4-point rubric
      benchmark.py    single, single-batched, single+verify, team
  tests/              offline, FakeLLM-driven
```

## Install and run

```bash
# from the book root, into the shared virtualenv
uv pip install --python .venv/bin/python -e book/projects/aie_core -e book/projects/agentkit \
    -e book/projects/p6-research-team
# or, inside book/projects/p6-research-team:
#   pip install -e ../aie_core -e ../agentkit -e .

cd book/projects/p6-research-team
python -m pytest -q                                   # offline, about one second
python -m research_team ask "What do I need to do before travelling abroad with a company laptop?"
python -m research_team ask --single "..."            # baseline; --verify adds the verification step
python -m research_team trace .runs/<trace_id>        # coordination tree plus each agent's outcome
python -m research_team.eval.benchmark                # offline benchmark, about 30 seconds
python -m research_team.eval.benchmark --live         # with the model from LLM_PROVIDER / LLM_MODEL
```

Docker (build from `book/projects` so the path dependencies are in the context):

```bash
docker build -f p6-research-team/Dockerfile -t northwind/research-team .
docker run --rm northwind/research-team ask "How many unused PTO days can I carry over?"
```

## Configuration

Library classes take explicit arguments. Only the CLI and the container read the environment
(`research_team/config.py`). Copy `env.example` to `.env`.

| Variable | Default | Meaning |
|---|---|---|
| `P6_OFFLINE` | `true` | scripted offline policy; `false` uses `aie_core.make_llm_client()` |
| `LLM_PROVIDER`, `LLM_MODEL`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | from aie_core | the model when `P6_OFFLINE=false` |
| `P6_SHARED_DATA_DIR` | `../shared-data` | corpus location |
| `P6_LOG_DIR` | `.runs` | one directory per run: `team.jsonl` plus `agents/<run_id>.jsonl` |
| `P6_MAX_TOKENS` | `80000` | global token budget for one team run |
| `P6_MAX_COST_USD` | unset | global cost budget (needs a pricing table) |
| `P6_DEADLINE_S` | `120` | global deadline; children inherit the remaining time |
| `P6_MAX_CHILDREN` | `6` | spawn cap: researcher tasks per run, follow-ups included |
| `P6_MAX_PARALLEL` | `4` | researchers running at once |
| `P6_CHILD_MAX_TOKENS`, `P6_CHILD_MAX_STEPS` | `12000`, `6` | requested slice per researcher |
| `P6_MAX_ROUNDS` | `2` | round 2 retries subquestions that produced no verified claim |
| `TRACE_SINK`, `TRACE_PATH` | `none` | aie_core tracer (spans carry `trace.id`, `parent.span_id`) |

## Run logs

```
.runs/<trace_id>/
  team.jsonl                     coordination: dispatched, refused (reason), finished, verified, conflicts
  agents/<trace_id>.planner.jsonl
  agents/<trace_id>.r1-sq1.jsonl  each agent's full agentkit event log; GoalSet.metadata carries
  agents/<trace_id>.verify-r1.jsonl   task_id, parent_id, trace_id, parent_span_id, role, depth
  agents/<trace_id>.synth.jsonl
```

Run ids follow the book-wide rule from Chapter 20: one `.` per level below the trace id and `-` inside
a segment, so `<trace_id>.r1-sq1` is a direct child of the run and a tool that counts dots reads its
depth correctly.

Any agent log can be replayed with `agentkit.replay(store.load(run_id))`.

## Benchmark results (offline)

The offline policy is the same for every configuration: same facet selection, same queries, same
extraction. Offline numbers therefore measure coordination cost, context growth, and verification,
not model quality. Latency and prices are simulated and illustrative. Results from
`python -m research_team.eval.benchmark` (one simulated hallucination per four numeric claims):

| config | rubric (0-4) | unsupported claims shipped | tokens per question | model calls | wall ms |
|---|---|---|---|---|---|
| single | 2.62 | 7 | 17,213 | 6.5 | 736 |
| single-batched | 2.62 | 7 | 7,479 | 3.0 | 463 |
| single+verify | 3.25 | 0 | 21,387 | 8.5 | 1,057 |
| team | 3.25 | 0 | 19,313 | 12.2 | 1,175 |

The script prints the verdict: the team beats `single` only through verification, and
`single+verify` recovers 100% of that gain. Against `single+verify` the team does not pay off on
either question kind. Without injected hallucinations (`--fabricate-every 0`) the team does not
pay off against any baseline.

## ADR-006: a multi-agent research team for cross-cutting policy questions

**Status:** accepted as an experiment; `single+verify` is the production default until the live
benchmark shows otherwise.

**Context.** Cross-cutting questions touch two to four policies owned by different teams (People
Operations, Finance, IT, Security). A single agent must search each area in one growing context.
Wrong numbers in answers about expenses, leave, and data incidents cause real harm, so every claim
needs a source a reviewer can check. Northwind's targets: p95 completion under 8 s, zero
cross-tenant leakage, a cost ceiling per answer.

**Why multiple agents here (the hypotheses).**

1. *Context isolation.* Each researcher sees one policy area and three passages, not twelve. With
   real models, long mixed contexts degrade answer quality; isolated contexts should not.
2. *Parallelism.* Policy areas are independent; researchers can run at the same time.
3. *Independent verification.* The verifier never sees the researcher's reasoning, only the claim
   and the cited passage, so it does not inherit the researcher's mistakes.
4. *Failure and budget isolation.* One area failing or exhausting its slice produces a partial
   answer that names the gap, not a failed run.

There is no separate permission domain: every agent acts with the asking employee's principal and
read-only tools, so permission separation is not a reason for this design.

**What the offline benchmark says.** Hypothesis 3 holds but does not require a team: a fixed
verification step after a single agent gives the same quality for less coordination. Hypothesis 2
is outweighed by the serial phases (plan, verify, synthesize): the team is slower than a sequential
single agent on these questions and much slower than a single agent that batches tool calls.
Context isolation lowers tokens per call (the team uses fewer tokens than the sequential single
agent with verification) but the planner, verifier, and duplicated reads across workers cost more
than a batched single agent. Hypothesis 1 cannot be tested offline, because the scripted policy does
not degrade with context length. Hypothesis 4 is real (see the budget tests) but has not been
valued yet.

**Decision.** Keep the team behind a flag. Use `single+verify` by default. Promote the team only if
the live benchmark shows a rubric gain of at least 0.5 on cross-cutting questions over
`single+verify` at no more than 1.5 times its tokens, or the same quality at least 25% faster.

**Consequences.** Every change to prompts or models reruns the four-way benchmark. The verifier and
the deterministic guard are shared by both paths, so improving verification improves both. The team
keeps its hard limits (spawn cap, global budget, depth 1, deadline propagation) even behind the flag.

## Public API

`research_team.__all__`: `ResearchTeam`, `TeamConfig`, `SingleAgent`, `Corpus`, `Passage`,
`TaskEnvelope`, `ResultEnvelope`, `BudgetSlice`, `TeamBudget`, `BudgetLedger`, `Admission`,
`TeamLog`, `TeamEvent`, `Plan`, `SubQuestion`, `ResearchFindings`, `Claim`, `EvidenceRef`,
`ClaimVerdict`, `VerificationReport`, `RejectedClaim`, `Conflict`, `AnswerReport`, `AgentUsage`,
`ErrorInfo`, `Role`, `TaskStatus`, `PropagatingTracer`, `span_tree`.
