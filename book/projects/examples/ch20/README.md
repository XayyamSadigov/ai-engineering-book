# Chapter 20 examples: agent architectures over agentkit

Nine patterns, each a small composition over `agentkit.AgentRuntime` (Chapter 19). None of them
contains its own model-and-tools loop.

```
patterns/
  common.py             role-tagged prompts, Meter (calls/tokens per role), PatternResult, make_agent
  fixtures.py           tiny in-memory Northwind tools (runbooks, policies, incidents, status, metrics)
  script.py             scripted FakeLLM handlers that dispatch on the role tag
  react.py              ReAct
  router.py             AgentRouter: rules first, closed-label model classification, human fallback
  planner_executor.py   typed plan, per-step agents, deviation rules, bounded replanning
  supervisor.py         Supervisor/Worker, delegation ledger, SpawnBudget, derived run ids
  hierarchical.py       Team trees built from supervisors
  reflection.py         CriticCheck inside the Definition of Done
  evaluator_optimizer.py independent evaluator, best-so-far, plateau and round stops
  parallel.py           fan_out with all / quorum / any completeness policies
  sequential.py         prompt chains whose steps are model calls, agents, or code, with gates
compare.py              runs all nine on one task and prints calls, runs, tool calls, tokens
test_patterns.py        24 offline tests
```

```bash
# from the book root
.venv/bin/python -m pytest book/projects/examples/ch20 -q
.venv/bin/python book/projects/examples/ch20/compare.py
```

Requires `aie_core` and `agentkit` installed (see their READMEs). No environment variables.
