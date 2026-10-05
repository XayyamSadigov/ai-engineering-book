# path: book/projects/evalkit/evalkit/gate.py
"""Release-gate thresholds as configuration, evaluated against a candidate run.

A gate turns an evaluation into a decision. The config is data (TOML or dict), reviewed like
code, and versioned next to the prompts and models it protects. Chapter 25 wires
`evaluate_gate` into CI; this module only decides pass or fail and explains why.
"""
from __future__ import annotations

import math
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .runner import Run
from .stats import ALL, bootstrap_ci, compare_slices, paired_bootstrap


class MetricRule(BaseModel):
    metric: str
    min_mean: float | None = None  # point estimate must reach this
    min_ci_low: float | None = None  # lower bound of the bootstrap CI must reach this
    must_pass_all: bool = False  # every case must pass (deterministic checks at 100%)
    max_regression: float | None = None  # vs baseline: point delta must be >= -max_regression
    fail_on_significant_regression: bool = False  # vs baseline: fail if the delta CI lies below 0


class SliceRule(BaseModel):
    metric: str
    max_regression: float
    min_n: int = 5  # slices smaller than this are reported, not gated: too noisy to block on
    slices: list[str] | None = None  # None means every tag in the candidate run


class CriticalRule(BaseModel):
    """Every case carrying `tag` must pass `metric`. One failure blocks the release."""

    tag: str
    metric: str


class GateConfig(BaseModel):
    name: str = "release"
    metrics: list[MetricRule] = Field(default_factory=list)
    slices: list[SliceRule] = Field(default_factory=list)
    critical: list[CriticalRule] = Field(default_factory=list)
    max_error_rate: float | None = 0.0
    max_evaluator_errors: int | None = 0
    max_p95_latency_ms: float | None = None
    max_cost_per_case_usd: float | None = None
    min_cases: int | None = None
    pinned_dataset_hash: str | None = None  # the frozen holdout this gate is valid for
    require_baseline: bool = False

    @classmethod
    def from_toml(cls, path: str | Path) -> "GateConfig":
        with Path(path).open("rb") as f:
            return cls.model_validate(tomllib.load(f))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GateConfig":
        return cls.model_validate(data)


class GateCheck(BaseModel):
    name: str
    passed: bool
    observed: str
    threshold: str
    detail: str = ""


class GateResult(BaseModel):
    config_name: str
    passed: bool
    checks: list[GateCheck]

    @property
    def failures(self) -> list[GateCheck]:
        return [c for c in self.checks if not c.passed]

    def to_markdown(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        lines = [f"**Gate `{self.config_name}`: {verdict}**", "", "| check | result | observed | threshold | detail |", "|---|---|---|---|---|"]
        for c in self.checks:
            lines.append(f"| {c.name} | {'pass' if c.passed else 'FAIL'} | {c.observed} | {c.threshold} | {c.detail} |")
        return "\n".join(lines)


def _fmt(x: float) -> str:
    return "nan" if isinstance(x, float) and math.isnan(x) else f"{x:.3f}"


def evaluate_gate(
    config: GateConfig,
    candidate: Run,
    baseline: Run | None = None,
    *,
    n_resamples: int = 2000,
    seed: int = 0,
    groups: dict[str, str] | None = None,
) -> GateResult:
    """Apply `config` to `candidate` (and `baseline` for regression rules).

    `groups` maps case id to a group key (conversation, customer, source document). When given,
    metric regression intervals use a cluster bootstrap, so correlated cases are not counted as
    independent evidence.
    """
    checks: list[GateCheck] = []
    add = checks.append

    if config.pinned_dataset_hash is not None:
        ok = candidate.dataset_hash.startswith(config.pinned_dataset_hash)
        add(GateCheck(name="dataset pinned", passed=ok, observed=candidate.dataset_hash[:12],
                      threshold=config.pinned_dataset_hash[:12],
                      detail="" if ok else "run is not on the frozen holdout this gate was set for"))
    if baseline is not None and baseline.dataset_hash != candidate.dataset_hash:
        add(GateCheck(name="same dataset as baseline", passed=False, observed=candidate.dataset_hash[:12],
                      threshold=baseline.dataset_hash[:12], detail="deltas across different datasets are meaningless"))
    if config.require_baseline and baseline is None:
        add(GateCheck(name="baseline present", passed=False, observed="none", threshold="required"))
    n_cases = len(candidate.by_case())
    if config.min_cases is not None:
        add(GateCheck(name="min cases", passed=n_cases >= config.min_cases, observed=str(n_cases), threshold=f">= {config.min_cases}"))
    if config.max_error_rate is not None:
        add(GateCheck(name="target error rate", passed=candidate.error_rate <= config.max_error_rate,
                      observed=_fmt(candidate.error_rate), threshold=f"<= {config.max_error_rate}"))
    if config.max_evaluator_errors is not None:
        n_err = candidate.evaluator_error_count
        add(GateCheck(name="evaluator errors", passed=n_err <= config.max_evaluator_errors,
                      observed=str(n_err), threshold=f"<= {config.max_evaluator_errors}"))
    if config.max_p95_latency_ms is not None:
        p95 = candidate.latency_percentile(95)
        add(GateCheck(name="p95 latency ms", passed=p95 <= config.max_p95_latency_ms, observed=f"{p95:.0f}",
                      threshold=f"<= {config.max_p95_latency_ms:.0f}"))
    if config.max_cost_per_case_usd is not None:
        cpc = candidate.cost_per_case_usd
        add(GateCheck(name="cost per case usd", passed=cpc <= config.max_cost_per_case_usd, observed=f"{cpc:.5f}",
                      threshold=f"<= {config.max_cost_per_case_usd}"))

    for rule in config.critical:
        tagged = {r.case_id for r in candidate.results if rule.tag in r.tags}
        failing = sorted(i for i in candidate.failing_cases(rule.metric) if i in tagged)
        missing = sorted(i for i in tagged if i not in candidate.case_scores(rule.metric))
        bad = failing + missing
        add(GateCheck(name=f"critical [{rule.tag}] {rule.metric}", passed=not bad and bool(tagged),
                      observed=f"{len(tagged) - len(bad)}/{len(tagged)} pass", threshold="all",
                      detail=", ".join(bad[:5]) if bad else ("" if tagged else "no cases carry this tag")))

    for rule in config.metrics:
        scores = list(candidate.case_scores(rule.metric).values())
        if not scores:
            add(GateCheck(name=f"{rule.metric} present", passed=False, observed="missing", threshold="scored"))
            continue
        ci = bootstrap_ci(scores, n_resamples=n_resamples, seed=seed)
        if rule.min_mean is not None:
            add(GateCheck(name=f"{rule.metric} mean", passed=ci.estimate >= rule.min_mean, observed=str(ci),
                          threshold=f">= {rule.min_mean}"))
        if rule.min_ci_low is not None:
            add(GateCheck(name=f"{rule.metric} CI low", passed=ci.low >= rule.min_ci_low, observed=_fmt(ci.low),
                          threshold=f">= {rule.min_ci_low}"))
        if rule.must_pass_all:
            failing = candidate.failing_cases(rule.metric)
            add(GateCheck(name=f"{rule.metric} all pass", passed=not failing, observed=f"{len(failing)} failing",
                          threshold="0", detail=", ".join(failing[:5])))
        if baseline is not None and (rule.max_regression is not None or rule.fail_on_significant_regression):
            pd = paired_bootstrap(baseline.case_scores(rule.metric), candidate.case_scores(rule.metric),
                                  n_resamples=n_resamples, seed=seed, groups=groups)
            if rule.max_regression is not None:
                add(GateCheck(name=f"{rule.metric} regression", passed=pd.delta >= -rule.max_regression,
                              observed=str(pd), threshold=f">= -{rule.max_regression}"))
            if rule.fail_on_significant_regression:
                add(GateCheck(name=f"{rule.metric} significant regression", passed=not pd.high < 0,
                              observed=f"CI high {pd.high:+.3f}", threshold="CI high >= 0"))

    for srule in config.slices:
        if baseline is None:
            continue
        wanted = None
        if srule.slices is not None:
            wanted = {s: [r.case_id for r in candidate.results if s in r.tags and r.repeat == 0] for s in srule.slices}
        for sd in compare_slices(baseline, candidate, srule.metric, slices=wanted, n_resamples=n_resamples, seed=seed):
            gated = sd.n >= srule.min_n
            ok = (sd.delta >= -srule.max_regression) or not gated
            label = "all" if sd.slice == ALL else sd.slice
            add(GateCheck(name=f"slice [{label}] {srule.metric}", passed=ok,
                          observed=f"{sd.delta:+.3f} (n={sd.n})", threshold=f">= -{srule.max_regression}",
                          detail="" if gated else f"n < {srule.min_n}: reported, not gated"))

    return GateResult(config_name=config.name, passed=all(c.passed for c in checks), checks=checks)


__all__ = ["MetricRule", "SliceRule", "CriticalRule", "GateConfig", "GateCheck", "GateResult", "evaluate_gate"]
