# path: book/projects/evalkit/evalkit/stats.py
"""Statistics for finite evaluation sets: how sure are we, and is the change real?

An eval score is an estimate of performance on a larger distribution, computed from a sample
of a few hundred cases at best. This module answers four questions:

* How uncertain is a score?            `bootstrap_ci`
* Is candidate better than baseline?   `paired_bootstrap` (CI on the delta + permutation p-value)
* Where did it change?                 `per_case_deltas`, `slice_breakdown`, `compare_slices`
* Could this dataset even detect it?   `mde_proportion`, `sample_size_for_mde`
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from statistics import NormalDist
from typing import Any

import numpy as np
from pydantic import BaseModel

from .runner import Run

ALL = "__all__"


class CI(BaseModel):
    estimate: float
    low: float
    high: float
    n: int
    confidence: float = 0.95

    def __str__(self) -> str:
        return f"{self.estimate:.3f} [{self.low:.3f}, {self.high:.3f}] (n={self.n})"


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,  # type: ignore[assignment]
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> CI:
    """Percentile bootstrap: resample cases with replacement, recompute, take quantiles."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return CI(estimate=math.nan, low=math.nan, high=math.nan, n=0, confidence=confidence)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_resamples, x.size))
    stats = np.apply_along_axis(statistic, 1, x[idx]) if statistic is not np.mean else x[idx].mean(axis=1)
    alpha = (1 - confidence) / 2
    return CI(
        estimate=float(statistic(x)),
        low=float(np.quantile(stats, alpha)),
        high=float(np.quantile(stats, 1 - alpha)),
        n=int(x.size),
        confidence=confidence,
    )


# ============================================================================ paired comparison
class PairedDelta(BaseModel):
    n: int
    baseline_mean: float
    candidate_mean: float
    delta: float
    low: float
    high: float
    p_value: float  # two-sided paired permutation (sign-flip) test
    wins: int  # cases where candidate > baseline
    losses: int
    ties: int
    confidence: float = 0.95

    @property
    def significant(self) -> bool:
        """The CI on the delta excludes zero."""
        return self.low > 0 or self.high < 0

    def __str__(self) -> str:
        return (
            f"delta {self.delta:+.3f} [{self.low:+.3f}, {self.high:+.3f}] p={self.p_value:.3f} "
            f"(W/L/T {self.wins}/{self.losses}/{self.ties}, n={self.n})"
        )


def paired_bootstrap(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    n_resamples: int = 5000,
    confidence: float = 0.95,
    seed: int = 0,
    groups: Mapping[str, str] | None = None,
) -> PairedDelta:
    """Compare two systems on the SAME cases.

    Pairing removes case difficulty from the noise: the quantity resampled is the per-case
    difference, whose variance is usually far smaller than the variance of either score.

    Cases are not always independent: turns of one conversation, or several questions about
    one document, tend to pass and fail together. Pass `groups` (case id -> group key) to
    resample whole groups (a cluster bootstrap) and flip signs per group, so the interval
    reflects the number of independent units rather than the number of rows. Case ids missing
    from `groups` form their own group.
    """
    ids = sorted(set(baseline) & set(candidate))
    if not ids:
        raise ValueError("baseline and candidate share no case ids")
    b = np.array([baseline[i] for i in ids], dtype=float)
    c = np.array([candidate[i] for i in ids], dtype=float)
    d = c - b
    rng = np.random.default_rng(seed)
    alpha = (1 - confidence) / 2
    observed = float(d.mean())
    if groups is None:
        idx = rng.integers(0, d.size, size=(n_resamples, d.size))
        boot = d[idx].mean(axis=1)
        signs = rng.choice([-1.0, 1.0], size=(n_resamples, d.size))
        perm = np.abs((signs * d).mean(axis=1))
    else:
        boot, perm = _cluster_resamples(d, [str(groups.get(i, i)) for i in ids], n_resamples, rng)
    p_value = float((np.sum(perm >= abs(observed) - 1e-12) + 1) / (n_resamples + 1))
    return PairedDelta(
        n=int(d.size),
        baseline_mean=float(b.mean()),
        candidate_mean=float(c.mean()),
        delta=observed,
        low=float(np.quantile(boot, alpha)),
        high=float(np.quantile(boot, 1 - alpha)),
        p_value=min(1.0, p_value),
        wins=int(np.sum(d > 0)),
        losses=int(np.sum(d < 0)),
        ties=int(np.sum(d == 0)),
        confidence=confidence,
    )


def _cluster_resamples(
    d: np.ndarray, keys: Sequence[str], n_resamples: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap and sign-flip distributions of the mean delta, resampling whole groups."""
    labels = sorted(set(keys))
    pos = {k: j for j, k in enumerate(labels)}
    gid = np.array([pos[k] for k in keys])
    sums = np.bincount(gid, weights=d, minlength=len(labels))
    sizes = np.bincount(gid, minlength=len(labels)).astype(float)
    draw = rng.integers(0, len(labels), size=(n_resamples, len(labels)))
    boot = sums[draw].sum(axis=1) / sizes[draw].sum(axis=1)
    signs = rng.choice([-1.0, 1.0], size=(n_resamples, len(labels)))
    perm = np.abs((signs * sums).sum(axis=1) / d.size)
    return boot, perm


def compare_runs(baseline: Run, candidate: Run, metric: str, **kwargs: Any) -> PairedDelta:
    return paired_bootstrap(baseline.case_scores(metric), candidate.case_scores(metric), **kwargs)  # type: ignore[arg-type]


class CaseDelta(BaseModel):
    case_id: str
    baseline: float
    candidate: float
    delta: float
    tags: list[str] = []


def per_case_deltas(baseline: Run, candidate: Run, metric: str, *, only_changed: bool = True) -> list[CaseDelta]:
    """Per-case score changes, regressions first. Read these before trusting any average."""
    b, c = baseline.case_scores(metric), candidate.case_scores(metric)
    tags = {r.case_id: r.tags for r in candidate.results}
    out = [
        CaseDelta(case_id=i, baseline=b[i], candidate=c[i], delta=c[i] - b[i], tags=tags.get(i, []))
        for i in sorted(set(b) & set(c))
    ]
    if only_changed:
        out = [x for x in out if x.delta != 0]
    return sorted(out, key=lambda x: (x.delta, x.case_id))


# ============================================================================ slices
class SliceStat(BaseModel):
    slice: str
    n: int
    mean: float
    low: float
    high: float


class SliceDelta(BaseModel):
    slice: str
    n: int
    baseline_mean: float
    candidate_mean: float
    delta: float
    low: float
    high: float


def _default_slices(run: Run) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {ALL: []}
    for r in run.results:
        if r.repeat:
            continue
        out[ALL].append(r.case_id)
        for t in r.tags:
            out.setdefault(t, []).append(r.case_id)
    return out


def slice_breakdown(
    run: Run,
    metric: str,
    *,
    slices: Mapping[str, Sequence[str]] | None = None,
    min_n: int = 1,
    n_resamples: int = 1000,
    seed: int = 0,
) -> list[SliceStat]:
    """Mean and bootstrap CI per slice (by tag unless explicit slices are given)."""
    scores = run.case_scores(metric)
    out = []
    for name, ids in (slices or _default_slices(run)).items():
        vals = [scores[i] for i in ids if i in scores]
        if len(vals) < min_n:
            continue
        ci = bootstrap_ci(vals, n_resamples=n_resamples, seed=seed)
        out.append(SliceStat(slice=name, n=ci.n, mean=ci.estimate, low=ci.low, high=ci.high))
    return sorted(out, key=lambda s: (s.slice != ALL, s.mean))


def compare_slices(
    baseline: Run,
    candidate: Run,
    metric: str,
    *,
    slices: Mapping[str, Sequence[str]] | None = None,
    min_n: int = 1,
    n_resamples: int = 2000,
    seed: int = 0,
) -> list[SliceDelta]:
    """Paired delta per slice, worst regression first."""
    b, c = baseline.case_scores(metric), candidate.case_scores(metric)
    out = []
    for name, ids in (slices or _default_slices(candidate)).items():
        shared = [i for i in ids if i in b and i in c]
        if len(shared) < min_n:
            continue
        pd = paired_bootstrap({i: b[i] for i in shared}, {i: c[i] for i in shared}, n_resamples=n_resamples, seed=seed)
        out.append(
            SliceDelta(
                slice=name, n=pd.n, baseline_mean=pd.baseline_mean, candidate_mean=pd.candidate_mean,
                delta=pd.delta, low=pd.low, high=pd.high,
            )
        )
    return sorted(out, key=lambda s: (s.delta, s.slice))


# ============================================================================ sample size
def _z(alpha: float, power: float) -> float:
    nd = NormalDist()
    return nd.inv_cdf(1 - alpha / 2) + nd.inv_cdf(power)


def mde_proportion(
    n: int,
    p: float = 0.5,
    *,
    paired: bool = False,
    discordance: float | None = None,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """Minimum detectable effect (absolute) for a pass rate near `p` with `n` cases per system.

    Unpaired (two independent samples): MDE = z * sqrt(2 p (1 - p) / n), z = z_{1-a/2} + z_power
    (about 2.8 for a = 0.05, power 0.8). Paired on the same cases: MDE ~= z * sqrt(d / n), where
    d is the discordance rate, the share of cases whose pass/fail flips between systems. If d is
    unknown it defaults to 2 p (1 - p), which is what independent outcomes would produce.
    A rule of thumb, not a power analysis: use it to reject datasets that cannot answer the question.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    z = _z(alpha, power)
    if paired:
        d = discordance if discordance is not None else 2 * p * (1 - p)
        return z * math.sqrt(d / n)
    return z * math.sqrt(2 * p * (1 - p) / n)


def sample_size_for_mde(
    mde: float,
    p: float = 0.5,
    *,
    paired: bool = False,
    discordance: float | None = None,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Inverse of `mde_proportion`: cases needed (per system) to detect an absolute change `mde`."""
    if mde <= 0:
        raise ValueError("mde must be positive")
    z = _z(alpha, power)
    var = (discordance if discordance is not None else 2 * p * (1 - p)) if paired else 2 * p * (1 - p)
    return math.ceil(z * z * var / (mde * mde))


__all__ = [
    "ALL",
    "CI",
    "bootstrap_ci",
    "PairedDelta",
    "paired_bootstrap",
    "compare_runs",
    "CaseDelta",
    "per_case_deltas",
    "SliceStat",
    "SliceDelta",
    "slice_breakdown",
    "compare_slices",
    "mde_proportion",
    "sample_size_for_mde",
]
