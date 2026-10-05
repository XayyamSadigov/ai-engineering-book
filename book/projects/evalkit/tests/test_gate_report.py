# path: book/projects/evalkit/tests/test_gate_report.py
import pytest

from evalkit import (
    Dataset,
    EvalCase,
    FunctionEvaluator,
    GateConfig,
    RunVersions,
    evaluate_gate,
    render_report,
    run_target,
)


def _dataset() -> Dataset:
    cases = [EvalCase(id=f"c{i:02d}", input=i, expected=i, tags=["billing" if i < 20 else "vpn"]) for i in range(40)]
    cases += [EvalCase(id=f"adv{i}", input=100 + i, expected=100 + i, tags=["adversarial", "critical"]) for i in range(3)]
    return Dataset(cases, name="gate-demo", version="1")


ok = FunctionEvaluator("ok", lambda c, o: o == c.expected)
schema = FunctionEvaluator("schema", lambda c, o: isinstance(o, int))


def _run(ds, fail_ids=(), prompt="p1", bad_type_ids=()):
    def target(case):
        if case.id in bad_type_ids:
            return str(case.input)
        return -1 if case.id in fail_ids else case.input

    return run_target(target, ds, evaluators=[ok, schema], versions=RunVersions(target="t", prompt=prompt))


CONFIG = GateConfig.from_dict(
    {
        "name": "demo",
        "min_cases": 40,
        "metrics": [
            {"metric": "schema", "must_pass_all": True},
            {"metric": "ok", "min_mean": 0.85, "max_regression": 0.02},
        ],
        "slices": [{"metric": "ok", "max_regression": 0.1, "min_n": 5}],
        "critical": [{"tag": "critical", "metric": "ok"}],
    }
)


def test_gate_passes_a_clean_candidate():
    ds = _dataset()
    base = _run(ds, fail_ids=["c01", "c02"])
    cand = _run(ds, fail_ids=["c01"], prompt="p2")
    res = evaluate_gate(CONFIG.model_copy(update={"pinned_dataset_hash": ds.content_hash[:12]}), cand, base)
    assert res.passed, res.to_markdown()


def test_gate_blocks_critical_failure_even_when_average_improves():
    ds = _dataset()
    base = _run(ds, fail_ids=["c01", "c02", "c03"])
    cand = _run(ds, fail_ids=["adv1"], prompt="p2")
    res = evaluate_gate(CONFIG, cand, base)
    assert not res.passed
    by_name = {c.name: c for c in res.failures}
    assert "ok mean" not in by_name and "ok regression" not in by_name  # the average got better
    assert "adv1" in by_name["critical [critical] ok"].detail
    assert "slice [adversarial] ok" not in by_name  # n=3 is below min_n: reported, not gated


def test_gate_blocks_slice_regression_and_schema_breaks():
    ds = _dataset()
    base = _run(ds)
    cand = _run(ds, fail_ids=["c20", "c21", "c22"], bad_type_ids=["c05"], prompt="p2")
    res = evaluate_gate(CONFIG, cand, base)
    failed = {c.name for c in res.failures}
    assert "schema all pass" in failed
    assert "slice [vpn] ok" in failed
    # c05 also fails "ok" ("5" != 5), but one case in 20 is -0.05: inside the 0.10 slice tolerance
    assert "slice [billing] ok" not in failed
    assert "ok regression" in failed  # 4 of 43 cases lost overall is beyond the 0.02 tolerance


def test_gate_checks_lineage_errors_latency_and_cost():
    ds = _dataset()
    other = Dataset([*ds.cases[:-1]], name="gate-demo", version="1")
    base = _run(other)
    cand = _run(ds)
    for r in cand.results:
        r.latency_ms, r.cost_usd = 3000.0, 0.01
    cfg = GateConfig(max_p95_latency_ms=2000, max_cost_per_case_usd=0.001, pinned_dataset_hash="deadbeef",
                     require_baseline=True)
    failed = {c.name for c in evaluate_gate(cfg, cand, base).failures}
    assert {"dataset pinned", "same dataset as baseline", "p95 latency ms", "cost per case usd"} <= failed
    assert "baseline present" in {c.name for c in evaluate_gate(cfg, cand).failures}


def test_gate_counts_target_errors_and_missing_metrics():
    ds = _dataset()

    def flaky(case):
        if case.id == "c00":
            raise TimeoutError("deadline")
        return case.input

    cand = run_target(flaky, ds, evaluators=[ok])
    cfg = GateConfig(metrics=[{"metric": "nonexistent", "min_mean": 0.5}])
    failed = {c.name for c in evaluate_gate(cfg, cand).failures}
    assert failed == {"target error rate", "nonexistent present"}


def test_gate_from_toml(tmp_path):
    p = tmp_path / "gate.toml"
    p.write_text('name = "x"\nmax_error_rate = 0.05\n[[metrics]]\nmetric = "ok"\nmin_ci_low = 0.5\n'
                 '[[critical]]\ntag = "critical"\nmetric = "ok"\n')
    cfg = GateConfig.from_toml(p)
    assert cfg.metrics[0].min_ci_low == 0.5 and cfg.critical[0].tag == "critical"


def test_report_without_and_with_baseline():
    ds = _dataset()
    base = _run(ds, fail_ids=["c01", "c25"])
    cand = _run(ds, fail_ids=["c02"], prompt="p2")
    single = render_report(cand)
    for section in ["## Lineage", "## Operations", "## Metrics", "## Slices", "## Failing cases", ds.fingerprint]:
        assert section in single
    gate = evaluate_gate(CONFIG, cand, base)
    diff = render_report(cand, baseline=base, metrics=["ok"], gate=gate, title="Demo")
    assert diff.startswith("# Demo: t")
    assert "Gate `demo`" in diff and "## Per-case changes" in diff
    assert "| c02 | 1.000 | 0.000 | -1.000 | billing |" in diff
    assert "| c25 | 0.000 | 1.000 | +1.000 | vpn |" in diff
    assert diff.index("c02") < diff.index("c25")  # regressions listed before fixes


@pytest.mark.parametrize("repeats", [2])
def test_report_mentions_flaky_cases_when_repeated(repeats):
    ds = Dataset([EvalCase(id="a", input=1, expected=1)], name="r")
    toggle = iter([1, 0])
    run = run_target(lambda c: next(toggle), ds, evaluators=[ok], repeats=repeats, concurrency=1)
    assert "Flaky cases" in render_report(run)
