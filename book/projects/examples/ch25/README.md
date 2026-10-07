# Chapter 25: task-specific evaluators and the CI release gate

Task evaluators built on `evalkit` (Chapter 24): prompts, RAG answers, agent trajectories read from
`agentkit` event logs (Chapter 19) with counterfactual replay, extraction, classification, summarization, and tool use; synthetic data generation with
validation filters; a pytest plugin and release gate for CI; online evaluation (feedback joins
and a sequential canary rule). Everything runs offline with deterministic stand-in models.

## Layout

```
ch25/
  pyproject.toml  conftest.py  .env.example
  taskevals/
    prompts.py          PromptContractEvaluator, reply_judge (rubric)
    rag.py              RagAnswerEvaluator (groundedness, relevance, citations, abstention), judge_evaluators
    trajectory.py       thin Trajectory JSON view, TrajectorySpec, assertions, TrajectoryEvaluator, pass_at_k
    replay.py           agentkit event logs -> Trajectory export (trajectory_from_events), recorded_run_target,
                        agentkit_replay_target (counterfactual replay via agentkit.replay), replay_fidelity_evaluator
    extraction.py       weighted field P/R, critical fields, line items, evidence location
    classification.py   LabelEvaluator, classification_aggregates (macro-F1, recall, ECE), evaluate_cascade
    summarization.py    key-fact coverage, lexical faithfulness, qualifiers, compression, FAITHFULNESS rubric
    tools.py            ToolUseEvaluator (selection, hallucinated tools, argument validity/correctness)
    synthetic.py        SyntheticGenerator, validate_candidates, difficulty tags, bias_report
    online.py           join_feedback, outcome_metrics, corrections_to_cases, CanaryMonitor
    standins.py         FakeLLM-based classifier, extractor, planner, tool selector (baseline/candidate/regressed)
    suites.py           the four fast suites (classification, extraction, agent, tools) and run_suite
    text_support.py     lexical helpers
  ci/
    run_suite.py        writes one evalkit Run JSON per suite
    release_gate.py     Run JSON + gates.toml -> summary.md, gate.json, reports/, exit 0/1/2
    gates.toml          per-suite evalkit GateConfig + run-level aggregate rules
    pytest_evalplugin.py  --eval-suite / eval_fast / eval_full / record_run
    baselines/          committed baseline runs (refresh with run_suite.py --system baseline)
    github/eval-gate.yml    GitHub Actions workflow (copy to .github/workflows/)
    gitlab/.gitlab-ci.yml   GitLab CI jobs (include from the root pipeline)
  data/
    agent_tasks.jsonl   agent tasks with trajectory specs
    tool_cases.jsonl    tool selection cases
    agent_runs/recorded/      agentkit JSONL event logs of the baseline agent; replay reads these
    agent_runs/production/    four real agentkit runs under misconfigurations (unapproved send, loop,
                              missing allow-list, schema drift)
    trajectories/             thin JSON exports of the same logs, for reading
    build_agent_runs.py       re-records the agentkit runs and exports
  tests/
```

## Install and run

```bash
# from the book root, into the shared virtualenv
uv pip install --python .venv/bin/python -e book/projects/aie_core -e book/projects/evalkit -e book/projects/agentkit pyyaml
# or: pip install -e ../../aie_core -e ../../evalkit -e ../../agentkit pyyaml

cd book/projects/examples/ch25
python -m pytest -q                                            # unit tests, eval suites skipped
python -m pytest -q --eval-suite fast --eval-out eval-out/runs -m eval_fast
python ci/release_gate.py --config ci/gates.toml --runs eval-out/runs --baselines ci/baselines --out eval-out
EVAL_SYSTEM=regressed python -m pytest -q --eval-suite fast --eval-out eval-out/runs -m eval_fast   # then the gate fails
```

## Configuration

| Variable | Used by | Meaning |
|---|---|---|
| `EVAL_SYSTEM` | pytest plugin | stand-in system under evaluation: baseline, candidate, regressed |
| `LLM_PROVIDER`, `LLM_MODEL` | real targets and judges | `aie_core` settings when you replace the stand-ins |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | real targets and judges | never in code, datasets, or workflow files |
| `GITHUB_STEP_SUMMARY` | release gate | set by GitHub Actions; the gate appends its summary |

## Public names other chapters may import

Put this directory on `sys.path`: `from taskevals import Trajectory, TrajectorySpec, TrajectoryEvaluator,
trajectory_from_events, recorded_run_target, agentkit_replay_target, ExtractionEvaluator, LabelEvaluator, classification_aggregates,
SummaryEvaluator, RagAnswerEvaluator, ToolUseEvaluator, SyntheticGenerator, validate_candidates,
join_feedback, CanaryMonitor`.
