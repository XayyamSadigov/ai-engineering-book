# path: book/projects/evalkit/tests/test_stats.py
import random

import pytest

from evalkit import (
    Dataset,
    EvalCase,
    FunctionEvaluator,
    bootstrap_ci,
    compare_runs,
    compare_slices,
    mde_proportion,
    paired_bootstrap,
    per_case_deltas,
    run_target,
    sample_size_for_mde,
    slice_breakdown,
)
from evalkit.stats import ALL


def test_bootstrap_ci_brackets_the_mean_and_narrows_with_n():
    rng = random.Random(0)
    small = [1.0 if rng.random() < 0.8 else 0.0 for _ in range(50)]
    large = [1.0 if rng.random() < 0.8 else 0.0 for _ in range(2000)]
    a, b = bootstrap_ci(small), bootstrap_ci(large)
    assert a.low <= a.estimate <= a.high
    assert (b.high - b.low) < (a.high - a.low) / 3
    assert bootstrap_ci([]).n == 0
    med = bootstrap_ci([1, 2, 3, 100], statistic=lambda x: float(sorted(x)[len(x) // 2]))
    assert med.estimate == 3


def test_paired_bootstrap_detects_consistent_small_gain_that_unpaired_view_would_miss():
    rng = random.Random(1)
    difficulty = {f"c{i}": rng.random() for i in range(300)}
    base = {k: float(d > 0.3) for k, d in difficulty.items()}  # ~70% pass
    # candidate fixes 15 baseline failures and breaks none
    fixed = [k for k, v in base.items() if v == 0][:15]
    cand = {k: (1.0 if k in fixed else v) for k, v in base.items()}
    pd = paired_bootstrap(base, cand, seed=3)
    assert pd.delta == pytest.approx(15 / 300)
    assert pd.significant and pd.low > 0 and pd.p_value < 0.01
    assert (pd.wins, pd.losses) == (15, 0)
    # the two separate CIs overlap even though the paired delta is unambiguous
    a, b = bootstrap_ci(list(base.values())), bootstrap_ci(list(cand.values()))
    assert a.high > b.low


def test_paired_bootstrap_with_no_effect_is_not_significant():
    rng = random.Random(2)
    base = {f"c{i}": float(rng.random() < 0.7) for i in range(200)}
    passing = [k for k, v in base.items() if v == 1.0][:10]
    failing = [k for k, v in base.items() if v == 0.0][:10]
    cand = {k: (0.0 if k in passing else 1.0 if k in failing else v) for k, v in base.items()}
    pd = paired_bootstrap(base, cand)
    assert (pd.wins, pd.losses) == (10, 10)
    assert pd.delta == 0 and not pd.significant and pd.p_value > 0.5
    assert pd.low < 0 < pd.high
    with pytest.raises(ValueError):
        paired_bootstrap({"a": 1.0}, {"b": 1.0})


def test_cluster_bootstrap_widens_the_interval_when_cases_are_correlated():
    # 20 conversations x 10 turns; the candidate fixes every turn of 3 conversations and nothing else.
    groups = {f"conv{g}-t{t}": f"conv{g}" for g in range(20) for t in range(10)}
    base = {k: 0.0 if int(k[4:].split("-")[0]) < 3 else 1.0 for k in groups}
    cand = {k: 1.0 for k in groups}
    naive = paired_bootstrap(base, cand, seed=0)
    clustered = paired_bootstrap(base, cand, seed=0, groups=groups)
    assert naive.delta == clustered.delta == pytest.approx(0.15)
    assert naive.wins == clustered.wins == 30
    # Treating 200 rows as independent claims far more certainty than 3 improved conversations support.
    assert (clustered.high - clustered.low) > 2 * (naive.high - naive.low)
    assert clustered.p_value > naive.p_value


def _runs():
    cases = [EvalCase(id=f"c{i}", input=i, expected=i, tags=["billing" if i < 10 else "vpn"]) for i in range(30)]
    ds = Dataset(cases, name="s")
    ev = FunctionEvaluator("ok", lambda c, o: o == c.expected)
    base = run_target(lambda c: c.input if c.input % 5 else -1, ds, evaluators=[ev])  # fails 0,5,10,15,20,25
    # candidate fixes vpn failures, breaks two billing cases
    cand = run_target(lambda c: -1 if c.input in (1, 2) else (c.input if c.input >= 10 or c.input % 5 else -1),
                      ds, evaluators=[ev])
    return base, cand


def test_per_case_deltas_lists_regressions_first():
    base, cand = _runs()
    deltas = per_case_deltas(base, cand, "ok")
    assert [d.case_id for d in deltas if d.delta < 0] == ["c1", "c2"]
    assert {d.case_id for d in deltas if d.delta > 0} == {"c10", "c15", "c20", "c25"}
    assert deltas[0].tags == ["billing"]


def test_slices_reveal_a_regression_the_average_hides():
    base, cand = _runs()
    overall = compare_runs(base, cand, "ok")
    assert overall.delta > 0
    by_slice = {s.slice: s for s in compare_slices(base, cand, "ok")}
    assert by_slice["billing"].delta == pytest.approx(-0.2)
    assert by_slice["vpn"].delta == pytest.approx(0.2)
    stats = slice_breakdown(cand, "ok")
    assert stats[0].slice == ALL and {s.slice for s in stats} == {ALL, "billing", "vpn"}


def test_minimum_detectable_effect_rule_of_thumb():
    # 200 cases, pass rate near 0.8: an unpaired comparison cannot see less than ~11 points
    assert mde_proportion(200, 0.8) == pytest.approx(0.112, abs=0.002)
    # paired on the same cases with 10% of verdicts flipping: ~6 points
    assert mde_proportion(200, 0.8, paired=True, discordance=0.10) == pytest.approx(0.0626, abs=0.002)
    n = sample_size_for_mde(0.05, 0.8)
    assert 1000 < n < 1020  # Lehr's rule: 16 * p(1-p) / d^2 = 1024 per group
    assert mde_proportion(n, 0.8) <= 0.05
    with pytest.raises(ValueError):
        mde_proportion(0)
