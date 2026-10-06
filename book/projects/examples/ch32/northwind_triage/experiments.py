# path: book/projects/examples/ch32/northwind_triage/experiments.py
"""Experiment arithmetic and rollout decisions: sample size, significance, canary verdicts,
and shadow comparison. Pure functions; the CI pipeline calls the CLI at the bottom.

    python -m northwind_triage.experiments sample-size --baseline 0.80 --mde 0.03
    python -m northwind_triage.experiments canary --baseline b.json --canary c.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Callable, Generic, Literal, Sequence, TypeVar

from pydantic import BaseModel, Field, model_validator

_Z = NormalDist()


# --------------------------------------------------------------------------- sample size
def sample_size_per_arm(p_baseline: float, mde_abs: float, alpha: float = 0.05,
                        power: float = 0.8) -> int:
    """Units per arm to detect an absolute change `mde_abs` in a proportion, two-sided.

    Normal approximation for two independent proportions. It assumes independent units: if
    one user sends many requests, randomize and count by user, or inflate n by the design
    effect, otherwise the test is overconfident.
    """
    if not 0 < p_baseline < 1:
        raise ValueError("p_baseline must be in (0, 1)")
    p2 = p_baseline + mde_abs
    if not 0 < p2 < 1 or mde_abs == 0:
        raise ValueError("p_baseline + mde_abs must be in (0, 1) and mde_abs non-zero")
    z_alpha = _Z.inv_cdf(1 - alpha / 2)
    z_beta = _Z.inv_cdf(power)
    p_bar = (p_baseline + p2) / 2
    numerator = (z_alpha * math.sqrt(2 * p_bar * (1 - p_bar))
                 + z_beta * math.sqrt(p_baseline * (1 - p_baseline) + p2 * (1 - p2))) ** 2
    return math.ceil(numerator / mde_abs ** 2)


@dataclass(frozen=True)
class ProportionTest:
    diff: float          # treatment - control
    z: float
    p_value: float
    ci_low: float
    ci_high: float


def two_proportion_test(success_a: int, n_a: int, success_b: int, n_b: int,
                        alpha: float = 0.05) -> ProportionTest:
    """Control a, treatment b. Pooled z for the p-value, unpooled SE for the interval."""
    if min(n_a, n_b) <= 0:
        raise ValueError("both arms need at least one unit")
    pa, pb = success_a / n_a, success_b / n_b
    pooled = (success_a + success_b) / (n_a + n_b)
    se_pooled = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    diff = pb - pa
    z = diff / se_pooled if se_pooled > 0 else 0.0
    p_value = 2 * (1 - _Z.cdf(abs(z)))
    se = math.sqrt(pa * (1 - pa) / n_a + pb * (1 - pb) / n_b)
    half = _Z.inv_cdf(1 - alpha / 2) * se
    return ProportionTest(diff=diff, z=z, p_value=p_value, ci_low=diff - half, ci_high=diff + half)


# --------------------------------------------------------------------------- canary
class ArmStats(BaseModel):
    """Aggregates for one arm over the canary window, exported from tracing/metrics."""

    requests: int = Field(ge=0)
    task_successes: int = Field(ge=0)       # e.g. ticket routed correctly per agent feedback
    errors: int = Field(ge=0)               # 5xx, classifier_unavailable, parse_error
    safety_violations: int = Field(ge=0)    # guardrail blocks that reached the user, leaks
    p95_latency_ms: float = Field(ge=0)
    cost_per_request: float = Field(ge=0)

    @model_validator(mode="after")
    def _counts_fit(self) -> "ArmStats":
        if self.task_successes > self.requests or self.errors > self.requests:
            raise ValueError("successes and errors cannot exceed requests")
        return self

    @property
    def success_rate(self) -> float:
        return self.task_successes / self.requests if self.requests else 0.0

    @property
    def error_rate(self) -> float:
        return self.errors / self.requests if self.requests else 0.0


class CanaryPolicy(BaseModel):
    """Rollback criteria written down BEFORE the canary starts. Hard limits roll back;
    insufficient data holds; everything else promotes."""

    min_requests: int = 400
    max_error_rate_increase: float = 0.01        # absolute
    max_p95_latency_ratio: float = 1.20
    max_cost_ratio: float = 1.15
    max_safety_violations: int = 0
    max_success_drop: float = 0.02               # absolute, on the primary metric
    alpha: float = 0.05


Action = Literal["promote", "hold", "rollback"]


class CanaryDecision(BaseModel):
    action: Action
    reasons: list[str]


def canary_decision(baseline: ArmStats, canary: ArmStats, policy: CanaryPolicy | None = None) -> CanaryDecision:
    policy = policy or CanaryPolicy()
    breaches: list[str] = []
    # Guardrails that can roll back on little data: safety is zero-tolerance.
    if canary.safety_violations > policy.max_safety_violations:
        breaches.append(f"safety violations {canary.safety_violations} > {policy.max_safety_violations}")
    if canary.requests and canary.error_rate - baseline.error_rate > policy.max_error_rate_increase:
        breaches.append(f"error rate {canary.error_rate:.3f} vs {baseline.error_rate:.3f}")
    if baseline.p95_latency_ms and canary.p95_latency_ms > baseline.p95_latency_ms * policy.max_p95_latency_ratio:
        breaches.append(f"p95 latency {canary.p95_latency_ms:.0f}ms > "
                        f"{policy.max_p95_latency_ratio}x {baseline.p95_latency_ms:.0f}ms")
    if baseline.cost_per_request and canary.cost_per_request > baseline.cost_per_request * policy.max_cost_ratio:
        breaches.append(f"cost/request {canary.cost_per_request:.5f} > "
                        f"{policy.max_cost_ratio}x {baseline.cost_per_request:.5f}")
    if breaches:
        return CanaryDecision(action="rollback", reasons=breaches)

    if canary.requests < policy.min_requests:
        return CanaryDecision(action="hold", reasons=[f"{canary.requests} < {policy.min_requests} requests"])

    test = two_proportion_test(baseline.task_successes, baseline.requests,
                               canary.task_successes, canary.requests, policy.alpha)
    # Roll back only when the drop is both larger than tolerated and statistically real.
    if test.ci_high < 0 and -test.diff > policy.max_success_drop:
        return CanaryDecision(action="rollback", reasons=[
            f"success rate dropped {test.diff:+.3f} (95% CI {test.ci_low:+.3f}..{test.ci_high:+.3f})"])
    if test.ci_low < -policy.max_success_drop:
        return CanaryDecision(action="hold", reasons=[
            f"cannot yet exclude a drop larger than {policy.max_success_drop} "
            f"(CI low {test.ci_low:+.3f}); keep collecting"])
    return CanaryDecision(action="promote", reasons=[
        f"success {test.diff:+.3f} (CI {test.ci_low:+.3f}..{test.ci_high:+.3f}); guardrails within limits"])


# --------------------------------------------------------------------------- shadow
I = TypeVar("I")
O = TypeVar("O")


@dataclass
class ShadowReport(Generic[O]):
    total: int = 0
    agreements: int = 0
    shadow_errors: int = 0
    disagreements: list[tuple[int, O, O | None]] = field(default_factory=list)

    @property
    def agreement_rate(self) -> float:
        compared = self.total - self.shadow_errors
        return self.agreements / compared if compared else 0.0


def run_shadow(inputs: Sequence[I], primary: Callable[[I], O], shadow: Callable[[I], O],
               same: Callable[[O, O], bool] = lambda a, b: a == b,
               keep_disagreements: int = 50) -> tuple[list[O], ShadowReport[O]]:
    """Serve `primary`'s outputs; run `shadow` on the same inputs and only compare.

    The shadow's result is never returned and its exceptions never propagate: a shadow that
    can affect users is a canary without the safety review. In production the shadow call is
    asynchronous and side-effect free (no tool writes, no emails), and its cost is real.
    """
    served: list[O] = []
    report: ShadowReport[O] = ShadowReport()
    for i, item in enumerate(inputs):
        out = primary(item)
        served.append(out)
        report.total += 1
        try:
            candidate = shadow(item)
        except Exception:  # noqa: BLE001 - shadow failures are data, not incidents
            report.shadow_errors += 1
            continue
        if same(out, candidate):
            report.agreements += 1
        elif len(report.disagreements) < keep_disagreements:
            report.disagreements.append((i, out, candidate))
    return served, report


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="northwind_triage.experiments")
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample-size")
    s.add_argument("--baseline", type=float, required=True)
    s.add_argument("--mde", type=float, required=True)
    s.add_argument("--alpha", type=float, default=0.05)
    s.add_argument("--power", type=float, default=0.8)
    c = sub.add_parser("canary")
    c.add_argument("--baseline", type=Path, required=True)
    c.add_argument("--canary", type=Path, required=True)
    c.add_argument("--policy", type=Path)
    args = parser.parse_args(argv)

    if args.cmd == "sample-size":
        print(sample_size_per_arm(args.baseline, args.mde, args.alpha, args.power))
        return 0
    baseline = ArmStats.model_validate_json(args.baseline.read_text())
    canary = ArmStats.model_validate_json(args.canary.read_text())
    policy = CanaryPolicy.model_validate_json(args.policy.read_text()) if args.policy else CanaryPolicy()
    decision = canary_decision(baseline, canary, policy)
    print(json.dumps(decision.model_dump(), indent=2))
    # Exit codes the pipeline branches on: 0 promote, 3 hold, 1 rollback.
    return {"promote": 0, "hold": 3, "rollback": 1}[decision.action]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
