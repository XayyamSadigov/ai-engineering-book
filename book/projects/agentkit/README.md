# agentkit

A bounded, event-sourced agent loop for the AI Engineering book. Chapter 19 builds and explains
it; Chapters 20 (architectures, Project 5), 21 (memory), 22 (multi-agent, Project 6), 38 (durable
agents), and the capstone import it.

What it gives you:

- `AgentRuntime.run(goal)`: model proposes, harness validates, authorizes, executes, records.
- Immutable events (`GoalSet`, `ModelDecision`, `ToolCallRequested`, `ToolCallApproved`,
  `ToolCallDenied`, `ToolResult`, `StepCompleted`, `BudgetUpdated`, `Note`, `FinalAnswer`,
  `Resumed`, `Stopped`) and an `AgentState` derived from them by one pure fold.
- Budgets (steps, tokens, cost, deadline, tool calls), repeated-action and no-progress detection,
  consecutive-error limits, and a `TerminationReason` for every way a run can end.
- A policy hook (visibility and per-call authorization), human approval as a resumable pause,
  error classes (transient, validation, semantic, permission, impossible, fatal).
- A composable Definition of Done that rejects premature final answers and tells the model why.
- Replay: recorded tool results served by action key, recorded decisions served in order;
  counterfactual replay with a new model or prompt reports the first divergence and misses.
- Event stores: in-memory and JSONL (one file per run, fsync per event).
- Tracing spans per run, step, and tool through `aie_core.observability`.

## Layout

```
agentkit/
  pyproject.toml  README.md  .env.example
  agentkit/
    __init__.py      public API (explicit __all__)
    events.py        immutable event types, JSON round trip, action_key
    state.py         AgentState, apply(), derive_state()
    budget.py        Budget, BudgetUsage, TerminationReason
    errors.py        ErrorClass, ToolError subclasses, classify_error
    tools.py         Tool protocol, FunctionTool, ToolOutput, adapt_tool, executor_tools,
                     validate_arguments, DefaultPolicy
    observations.py  truncate_observation (head and tail)
    dod.py           verifiers, all_of/any_of, DefinitionOfDone
    store.py         EventStore protocol, InMemoryEventStore, JsonlEventStore
    runtime.py       AgentRuntime, LoopConfig, RunResult
    replay.py        RecordedLLM, RecordedToolResults, replay(), ReplayReport
  examples/northwind_incident.py   incident research over shared-data, approval pause, replay
  tests/                           offline tests with FakeLLM
```

## Install and run

```bash
# from the book root, into the shared virtualenv
uv pip install --python .venv/bin/python -e book/projects/aie_core -e book/projects/agentkit
# or: pip install -e ../aie_core && pip install -e .

cd book/projects/agentkit
python -m pytest -q                       # offline; toolkit integration test skips if toolkit is absent
python examples/northwind_incident.py     # writes JSONL runs to $AGENTKIT_EVENT_DIR (default .agent-runs)
```

## Configuration

agentkit reads no environment variables itself. The model client comes from `aie_core`.

| Variable | Used by | Meaning |
|---|---|---|
| `LLM_PROVIDER`, `LLM_MODEL`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | `aie_core.make_llm_client()` | which model drives the loop |
| `TRACE_SINK`, `TRACE_PATH` | `aie_core.observability.get_tracer()` | where spans go |
| `AGENTKIT_EVENT_DIR` | the example only | directory for JSONL event logs |

## Usage

```python
from aie_core import make_llm_client
from agentkit import (AgentRuntime, Budget, DefaultPolicy, DefinitionOfDone, FunctionTool,
                      JsonlEventStore, citations_grounded, tool_was_called)

search = FunctionTool("search_docs", "Search runbooks.",
                      {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
                      fn=lambda query: "[it-vpn-access-runbook] clear the cached profile")
rt = AgentRuntime(
    make_llm_client(), [search],
    budget=Budget(max_steps=8, max_cost_usd=0.50, deadline_s=60),
    policy=DefaultPolicy(allowed={"search_docs"}),
    dod=DefinitionOfDone(tool_was_called("search_docs"), citations_grounded(1)),
    store=JsonlEventStore(".agent-runs"),
)
result = rt.run("VPN login loops for a store manager. What should support do?")
print(result.stop_reason, result.final_answer, result.trajectory())
```

Approval pause and resume (the second call can happen in another process):

```python
paused = rt.run("Reply to TCK-9", run_id="r1")          # stop_reason == APPROVAL_REQUIRED
rt.resume("r1", approve=True, reason="checked by on-call")
```

Replay without executing tools:

```python
from agentkit import replay
report = replay(store.load("r1"))                       # recorded decisions: harness regression test
report = replay(store.load("r1"), new_llm, system_prompt="...")  # counterfactual: new planner
print(report.summary(), report.first_divergence, report.misses)
```

Driving Chapter 16's governed executor:

```python
from agentkit import executor_tools
tools = executor_tools(tool_executor, toolkit_ctx)      # policy, idempotency, audit stay in toolkit
rt = AgentRuntime(llm, tools)
```

## Public API

See `agentkit/__init__.py`. Stable names other chapters rely on: `AgentRuntime`, `LoopConfig`,
`RunResult`, `Budget`, `BudgetUsage`, `TerminationReason`, `AgentState`, `AgentStatus`,
`derive_state`, every event class, `EventStore`, `InMemoryEventStore`, `JsonlEventStore`,
`Tool`, `FunctionTool`, `function_tool`, `ToolOutput`, `ToolContext`, `SideEffect`, `adapt_tool`,
`executor_tools`, `DefaultPolicy`, `PolicyDecision`, `ErrorClass`, `classify_error`,
`DefinitionOfDone`, the verifier helpers, `replay`, `RecordedLLM`, `ReplayReport`.
