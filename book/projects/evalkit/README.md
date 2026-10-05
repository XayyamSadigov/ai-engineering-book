# evalkit

The evaluation core of the AI Engineering book (Chapter 24). Versioned datasets of evaluation
cases, a runner that records outputs with full lineage, deterministic and classification
metrics, rubric-based LLM judges with calibration against human labels, statistics for finite
eval sets, Markdown reports, and release-gate thresholds as configuration.

Chapter 14 (RAG evaluation), Chapter 25 (task-specific evaluators and the CI gate), and the
projects import it. It depends only on `aie_core` and pydantic.

## Install

```bash
# from the book root, into the shared virtualenv
uv pip install --python .venv/bin/python -e book/projects/aie_core -e book/projects/evalkit
# or plain pip
pip install -e ../aie_core && pip install -e .
```

As a path dependency of another project:

```toml
[project]
dependencies = ["aie-core", "evalkit"]

[tool.uv.sources]
aie-core = { path = "../aie_core", editable = true }
evalkit = { path = "../evalkit", editable = true }
```

## Layout

```
evalkit/
  cases.py                  EvalCase, Dataset (JSONL, content hash, group split), check_leakage
  runner.py                 Score, Evaluator, FunctionEvaluator, TargetResult, Run, run_target, score_run
  metrics/deterministic.py  exact_match, contains, forbids, numeric_close, set P/R, field_prf, json_schema_valid
  metrics/classification.py ConfusionMatrix, threshold_sweep, best_threshold, calibration_bins, ECE, Brier
  judges.py                 Rubric, LLMJudge, PairwiseJudge, pairwise_summary, cohens_kappa, calibrate_judge
  stats.py                  bootstrap_ci, paired_bootstrap, per_case_deltas, slices, mde_proportion
  report.py                 render_report (Markdown)
  gate.py                   GateConfig (TOML), evaluate_gate
data/
  northwind_tickets_v1.jsonl  frozen example dataset (60 tickets + 4 injection cases)
  gate.toml                   example release gate
examples/ticket_triage_eval.py  baseline vs candidate prompt, report, gate (offline, FakeLLM)
tests/                          offline tests
```

## Configuration

evalkit reads no environment variables itself. Targets and judges receive an `LLMClient`
from the caller, normally `aie_core.make_llm_client()`, which honors `LLM_PROVIDER`,
`LLM_MODEL`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `TRACE_SINK` (see `aie_core/README.md`).
Use a separate client (and model) for judges than for the system under test where you can.
`.env.example` lists the usual variables.

## Usage

```python
from aie_core import make_llm_client
from evalkit import (Dataset, FunctionEvaluator, GateConfig, LLMJudge, CORRECTNESS, RunVersions,
                     evaluate_gate, render_report, run_target)
from evalkit.metrics import exact_match

dataset = Dataset.load_jsonl("data/northwind_tickets_v1.jsonl")
dev, holdout = dataset.split(0.3, group_by="customer_id", seed="v1")

correct = FunctionEvaluator("category_correct",
                            lambda case, out: exact_match(out["category"], case.expected["category"]))
judge = LLMJudge(make_llm_client(), CORRECTNESS, model="judge-model").as_evaluator(
    answer_fn=lambda out: out["reply"], reference_fn=lambda case: case.expected["reply"])

run = run_target(my_target, holdout, evaluators=[correct, judge], concurrency=8,
                 versions=RunVersions(target="triage", prompt="triage-v2", model="model-x"))
gate = evaluate_gate(GateConfig.from_toml("data/gate.toml"), run, baseline_run)
print(render_report(run, baseline=baseline_run, gate=gate))
```

Run the worked example:

```bash
python examples/ticket_triage_eval.py run    # writes examples/out/report.md, exit 1 when the gate fails
```

## Public API

| Module | Names |
|---|---|
| `evalkit.cases` | `EvalCase(id, input, expected=None, rubric=[], tags=[], metadata={})`, `.group_key(group_by)`; `Dataset(cases, *, name, version="0", description="")`: `len`, iteration, `in`, `get(id)`, `ids`, `content_hash`, `fingerprint`, `verify_hash(prefix)`, `load_jsonl(path)`, `save_jsonl(path)`, `subset(ids)`, `filter(predicate, *, tags)`, `tag_counts()`, `slices(prefix)`, `split(holdout_fraction=0.3, *, group_by=None, seed=0) -> (dev, holdout)`; `check_leakage(a, b, *, group_by=None) -> LeakageReport`; `DatasetError` |
| `evalkit.runner` | `Score(name, value, passed=None, detail=None)`; `Evaluator` protocol (`name`, `version`, optional `metric_names`, `__call__(case, output)`); `FunctionEvaluator(name, fn, *, version="1", pass_threshold=1.0, metric_names=None)`; `@evaluator(name, version=...)`; `TargetResult(output, cost_usd, input_tokens, output_tokens, trace_id, metadata)`; `CaseResult`; `RunVersions(target, prompt, model, dataset, evaluators, extra)`; `Run` with `case_scores(m)`, `mean(m)`, `pass_rate(m)`, `failing_cases(m)`, `flaky_cases(m)`, `errors`, `error_rate`, `evaluator_error_count`, `latency_percentile(p)`, `total_cost_usd`, `cost_per_case_usd`, `by_case()`, `metric_names()`, `save_json`, `load_json`; `run_target(target, dataset, *, versions, evaluators, concurrency=8, repeats=1, tracer, error_score=0.0, on_result)`; `arun_target(...)`; `score_run(run, dataset, evaluators, *, replace=False)` |
| `evalkit.metrics` | `PRF`, `FieldScores`, `prf_from_counts(tp, fp, fn, *, zero_division=1.0)`, `normalize_text`, `exact_match`, `contains(text, phrases, *, mode)`, `forbids`, `parse_number`, `numeric_close(p, e, *, abs_tol, rel_tol)`, `set_precision_recall`, `field_prf(pred, gold, *, fields, normalize, numeric_tol)`, `json_schema_valid(output, schema) -> (ok, errors)`; `ConfusionMatrix(y_true, y_pred, labels)` with `per_label`, `macro`, `micro`, `accuracy`, `most_confused`, `to_markdown`; `classification_report`; `binary_counts`; `threshold_sweep(y, scores, thresholds, *, cost_fp, cost_fn, cost_per_flag) -> [ThresholdPoint]`; `best_threshold(points, *, by="cost", min_recall, min_precision)`; `calibration_bins`, `CalibrationBin`, `expected_calibration_error`, `brier_score` |
| `evalkit.judges` | `Rubric(name, task, levels, pass_threshold, version, flagged_label, examples)`, `RubricLevel`; built-ins `GROUNDEDNESS` (0-3), `CORRECTNESS` (0-2), `RELEVANCE` (0-2); `LLMJudge(client, rubric, *, model, max_repair_attempts=2)` with `build_request`, `judge(*, input, answer, reference, evidence) -> JudgeResult`, `version`, `as_evaluator(*, input_fn, answer_fn, reference_fn, evidence_fn) -> JudgeEvaluator`; `PairwiseJudge(client, criterion, *, seed, both_orders=False, allow_tie=True, model)` with `compare(input, a, b, *, case_id) -> PairwiseResult`; `pairwise_summary(results) -> PairwiseSummary`; `agreement`, `cohens_kappa(a, b, *, labels, weights)`, `calibrate_judge(judge_labels, human_labels, *, pass_threshold, ordinal_labels) -> JudgeCalibration` |
| `evalkit.stats` | `CI`, `bootstrap_ci(values, *, statistic, n_resamples, confidence, seed)`; `PairedDelta` (`.significant`), `paired_bootstrap(baseline, candidate, *, n_resamples, confidence, seed, groups=None)` (pass `groups` for a cluster bootstrap over correlated cases), `compare_runs(base_run, cand_run, metric)`; `CaseDelta`, `per_case_deltas(base, cand, metric)`; `SliceStat`, `slice_breakdown(run, metric, *, slices, min_n)`; `SliceDelta`, `compare_slices(base, cand, metric, *, slices, min_n)`; `mde_proportion(n, p, *, paired, discordance, alpha, power)`, `sample_size_for_mde(mde, p, ...)`; `ALL` |
| `evalkit.report` | `render_report(run, *, baseline=None, metrics=None, gate=None, title=None, slice_min_n=3, max_case_deltas=15)` |
| `evalkit.gate` | `GateConfig` (`from_toml`, `from_dict`; fields `metrics`, `slices`, `critical`, `max_error_rate`, `max_evaluator_errors`, `max_p95_latency_ms`, `max_cost_per_case_usd`, `min_cases`, `pinned_dataset_hash`, `require_baseline`), `MetricRule`, `SliceRule`, `CriticalRule`, `GateCheck`, `GateResult` (`passed`, `failures`, `to_markdown`), `evaluate_gate(config, candidate, baseline=None, *, groups=None)` |

Everything listed is also importable from the top-level `evalkit` package, except the metric
functions, which live in `evalkit.metrics`.

## Tests

```bash
cd book/projects/evalkit
python -m pytest -q
```

All tests run offline. Judges are tested with `aie_core`'s `FakeLLM`; two tests cross-check
kappa and macro-F1 against scikit-learn when it is installed and skip otherwise.
