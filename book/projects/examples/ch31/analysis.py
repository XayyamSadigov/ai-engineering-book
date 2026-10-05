# path: book/projects/examples/ch31/analysis.py
"""Debugging queries over a TraceStore.

Each function answers one question from the debugging playbook:

* gold_funnel / triage_by_stage   where along the pipeline does gold evidence get lost?
* compare_versions                does a version split explain the change, with a CI?
* retrieved_not_cited             which traces had the right evidence and still answered wrong?
* latency_breakdown               which stage owns the p95?
* cost_by_tenant                  who pays for what, per request and per successful request?
* error_distribution              which failure classes moved?
* trajectory_view / issues        what did the agent actually do, step by step?
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from semconv import Attr, SpanName
from trace_store import TraceStore, TraceTree

FUNNEL_STAGES: tuple[str, ...] = ("retrieved", "reranked", "in_context", "cited", "answer_passed")
_STAGE_SPAN = {"retrieved": SpanName.RETRIEVAL, "reranked": SpanName.RERANK, "in_context": SpanName.CONTEXT,
               "cited": SpanName.GENERATE, "answer_passed": SpanName.EVAL}


# ============================================================================ small statistics
def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile, p in [0, 100]. NaN for an empty sequence."""
    if not values:
        return math.nan
    xs = sorted(values)
    k = (len(xs) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] if lo == hi else xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def bootstrap_diff_ci(a: Sequence[float], b: Sequence[float], *, n_boot: int = 2000, seed: int = 0,
                      alpha: float = 0.05) -> tuple[float, float, float]:
    """Difference of means b - a with a percentile bootstrap interval."""
    if not a or not b:
        return math.nan, math.nan, math.nan
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        sa = [a[rng.randrange(len(a))] for _ in a]
        sb = [b[rng.randrange(len(b))] for _ in b]
        diffs.append(sum(sb) / len(sb) - sum(sa) / len(sa))
    point = sum(b) / len(b) - sum(a) / len(a)
    return point, percentile(diffs, 100 * alpha / 2), percentile(diffs, 100 * (1 - alpha / 2))


def _rate(flags: Iterable[bool]) -> float:
    xs = list(flags)
    return sum(xs) / len(xs) if xs else math.nan


# ============================================================================ evidence funnel
def _ids(span_attr: Any) -> set[str]:
    return set(span_attr or [])


def evidence_path(tree: TraceTree) -> dict[str, bool] | None:
    """For a labeled trace: did the gold evidence survive each stage? None if unlabeled."""
    gold = set(tree.gold_ids or [])
    if not gold:
        return None
    retrieval, rerank, context = tree.first(SpanName.RETRIEVAL), tree.first(SpanName.RERANK), tree.first(SpanName.CONTEXT)
    retrieved = gold <= _ids(retrieval.attributes.get(Attr.RETRIEVAL_IDS)) if retrieval else False
    reranked = gold <= _ids(rerank.attributes.get(Attr.RERANK_IDS)) if rerank else retrieved
    in_context = gold <= _ids(context.attributes.get(Attr.CONTEXT_IDS)) if context else reranked
    cited = bool(gold & _ids(tree.get(Attr.CITATION_IDS)))
    passed = bool(tree.eval_passed)
    # survival is cumulative: a stage cannot recover evidence an earlier stage lost
    path = {"retrieved": retrieved}
    path["reranked"] = path["retrieved"] and reranked
    path["in_context"] = path["reranked"] and in_context
    path["cited"] = path["in_context"] and cited
    path["answer_passed"] = passed
    return path


def gold_funnel(store: TraceStore) -> dict[str, float]:
    paths = [p for t in store if (p := evidence_path(t)) is not None]
    out = {stage: _rate(p[stage] for p in paths) for stage in FUNNEL_STAGES}
    out["n"] = float(len(paths))
    return out


@dataclass
class TriageReport:
    baseline: dict[str, float]
    candidate: dict[str, float]
    conditional_drop: dict[str, float]
    first_failing_stage: str | None
    error_delta: dict[str, float]
    tolerance: float

    def render(self) -> str:
        lines = [f"{'stage':<14}{'baseline':>10}{'candidate':>11}{'cond. drop':>12}",
                 f"{'(labeled n)':<14}{self.baseline['n']:>10.0f}{self.candidate['n']:>11.0f}"]
        for stage in FUNNEL_STAGES:
            mark = "  <-- first failing stage" if stage == self.first_failing_stage else ""
            lines.append(f"{stage:<14}{self.baseline[stage]:>10.2f}{self.candidate[stage]:>11.2f}"
                         f"{self.conditional_drop[stage]:>12.2f}{mark}")
        if self.error_delta:
            lines.append("error classes (rate per trace, candidate minus baseline):")
            for k, v in sorted(self.error_delta.items(), key=lambda kv: -abs(kv[1])):
                lines.append(f"  {k:<26}{v:+.3f}")
        return "\n".join(lines)


def _conditional(f: dict[str, float]) -> dict[str, float]:
    """P(survive stage | survived previous stage). Absolute rates hide where loss happens."""
    out, prev = {}, 1.0
    for stage in FUNNEL_STAGES:
        rate = f[stage]
        out[stage] = rate / prev if prev and not math.isnan(rate) and stage != "answer_passed" else rate
        prev = rate if stage != "answer_passed" else prev
    return out


def error_distribution(store: TraceStore) -> Counter[str]:
    return Counter(c for t in store for c in t.error_classes)


def triage_by_stage(baseline: TraceStore, candidate: TraceStore, *, tolerance: float = 0.05) -> TriageReport:
    fb, fc = gold_funnel(baseline), gold_funnel(candidate)
    cb, cc = _conditional(fb), _conditional(fc)
    drop = {s: cb[s] - cc[s] for s in FUNNEL_STAGES}
    first = next((s for s in FUNNEL_STAGES[:-1] if drop[s] > tolerance), None)
    eb, ec = error_distribution(baseline), error_distribution(candidate)
    nb, nc = max(1, len(baseline)), max(1, len(candidate))
    error_delta = {k: ec[k] / nc - eb[k] / nb for k in set(eb) | set(ec)}
    return TriageReport(fb, fc, drop, first, error_delta, tolerance)


# ============================================================================ version comparison
@dataclass
class GroupSummary:
    label: str
    n: int
    eval_pass_rate: float
    labeled: int
    negative_feedback_rate: float
    feedback_n: int
    p50_ms: float
    p95_ms: float
    mean_input_tokens: float
    mean_output_tokens: float
    cost_per_request: float
    errors: Counter[str] = field(default_factory=Counter)


def _generation_tokens(t: TraceTree, key: str) -> float:
    # llm.generate holds the logical totals; llm.complete attempts would double count retries
    return float(sum(s.attributes.get(key) or 0 for s in t.find(SpanName.GENERATE)))


def summarize(store: TraceStore, label: str = "") -> GroupSummary:
    traces = store.traces()
    passed = [t.eval_passed for t in traces if t.eval_passed is not None]
    fb = [t.feedback_value for t in traces if t.feedback_value is not None]
    durations = [t.duration_ms for t in traces]
    n = max(1, len(traces))
    return GroupSummary(
        label=label, n=len(traces),
        eval_pass_rate=_rate(passed), labeled=len(passed),
        negative_feedback_rate=_rate(v < 0 for v in fb), feedback_n=len(fb),
        p50_ms=percentile(durations, 50), p95_ms=percentile(durations, 95),
        mean_input_tokens=sum(_generation_tokens(t, Attr.LLM_INPUT_TOKENS) for t in traces) / n,
        mean_output_tokens=sum(_generation_tokens(t, Attr.LLM_OUTPUT_TOKENS) for t in traces) / n,
        cost_per_request=sum(t.cost_usd for t in traces) / n,
        errors=error_distribution(store),
    )


@dataclass
class VersionComparison:
    key: str
    a: GroupSummary
    b: GroupSummary
    pass_rate_diff: tuple[float, float, float]
    neg_feedback_diff: tuple[float, float, float]

    def render(self) -> str:
        rows = [("requests", "n", "{:.0f}"), ("labeled", "labeled", "{:.0f}"), ("eval pass rate", "eval_pass_rate", "{:.3f}"),
                ("neg. feedback rate", "negative_feedback_rate", "{:.3f}"), ("p50 latency ms", "p50_ms", "{:.0f}"),
                ("p95 latency ms", "p95_ms", "{:.0f}"), ("input tokens/req", "mean_input_tokens", "{:.0f}"),
                ("output tokens/req", "mean_output_tokens", "{:.0f}"), ("cost/req (illus.)", "cost_per_request", "{:.5f}")]
        lines = [f"{self.key:<20}{self.a.label:>14}{self.b.label:>14}"]
        for title, attr, fmt in rows:
            lines.append(f"{title:<20}{fmt.format(getattr(self.a, attr)):>14}{fmt.format(getattr(self.b, attr)):>14}")
        p, lo, hi = self.pass_rate_diff
        lines.append(f"pass-rate diff (b-a): {p:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
        p, lo, hi = self.neg_feedback_diff
        lines.append(f"neg-feedback diff (b-a): {p:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
        return "\n".join(lines)


def compare_versions(store: TraceStore, key: str, a: Any, b: Any, *, seed: int = 0) -> VersionComparison:
    sa, sb = store.filter(versions={key: a}), store.filter(versions={key: b})
    pa = [float(t.eval_passed) for t in sa if t.eval_passed is not None]
    pb = [float(t.eval_passed) for t in sb if t.eval_passed is not None]
    fa = [float(t.feedback_value < 0) for t in sa if t.feedback_value is not None]
    fb = [float(t.feedback_value < 0) for t in sb if t.feedback_value is not None]
    return VersionComparison(key, summarize(sa, str(a)), summarize(sb, str(b)),
                             bootstrap_diff_ci(pa, pb, seed=seed), bootstrap_diff_ci(fa, fb, seed=seed))


# ============================================================================ evidence present, answer wrong
@dataclass
class EvidenceFinding:
    trace_id: str
    gold: list[str]
    retrieved_rank: int | None
    in_context: bool
    dropped_by_packer: bool
    cited: list[str]


def retrieved_not_cited(store: TraceStore) -> list[EvidenceFinding]:
    """Labeled traces where retrieval found the gold evidence but the answer did not cite it.
    These are the cases prompt tweaking is tempted to fix and usually cannot."""
    out = []
    for t in store:
        path = evidence_path(t)
        if path is None or not path["retrieved"] or path["cited"]:
            continue
        gold = list(t.gold_ids or [])
        retrieval, context = t.first(SpanName.RETRIEVAL), t.first(SpanName.CONTEXT)
        returned = list(retrieval.attributes.get(Attr.RETRIEVAL_IDS) or []) if retrieval else []
        dropped = set(context.attributes.get(Attr.CONTEXT_DROPPED) or []) if context else set()
        rank = min((returned.index(g) + 1 for g in gold if g in returned), default=None)
        out.append(EvidenceFinding(t.trace_id, gold, rank, path["in_context"], bool(set(gold) & dropped),
                                   list(t.get(Attr.CITATION_IDS) or [])))
    return out


# ============================================================================ latency and cost
def latency_breakdown(store: TraceStore, percentiles: Sequence[float] = (50, 95, 99)) -> dict[str, dict[str, float]]:
    """Per span name: duration percentiles, and the median share of its request's duration."""
    durations: dict[str, list[float]] = defaultdict(list)
    shares: dict[str, list[float]] = defaultdict(list)
    for t in store:
        total = t.duration_ms or 1.0
        per_trace: dict[str, float] = defaultdict(float)
        for s in t.spans:
            per_trace[s.name] += s.duration_ms
        for name, d in per_trace.items():
            durations[name].append(d)
            shares[name].append(d / total)
    return {name: {**{f"p{int(p)}": percentile(ds, p) for p in percentiles}, "share_p50": percentile(shares[name], 50),
                   "n": float(len(ds))} for name, ds in sorted(durations.items(), key=lambda kv: -percentile(kv[1], 95))}


def render_latency(breakdown: dict[str, dict[str, float]]) -> str:
    lines = [f"{'span':<20}{'p50':>9}{'p95':>9}{'p99':>9}{'share':>8}"]
    for name, row in breakdown.items():
        lines.append(f"{name:<20}{row['p50']:>9.0f}{row['p95']:>9.0f}{row['p99']:>9.0f}{row['share_p50']:>8.0%}")
    return "\n".join(lines)


def cost_by_tenant(store: TraceStore) -> dict[str, dict[str, float]]:
    """Cost per request and per *successful* request. Success = no error class and, when the
    trace is labeled, a passing eval; failed work still costs money and is charged to success."""
    groups: dict[str, list[TraceTree]] = defaultdict(list)
    for t in store:
        groups[t.tenant or "unknown"].append(t)
    out = {}
    for tenant, traces in sorted(groups.items()):
        cost = sum(t.cost_usd for t in traces)
        ok = [t for t in traces if not t.error_classes and t.eval_passed is not False]
        out[tenant] = {"requests": float(len(traces)), "cost_total": cost, "cost_per_request": cost / len(traces),
                       "avoided_by_cache": sum(t.avoided_cost_usd for t in traces),
                       "successful": float(len(ok)), "cost_per_success": cost / len(ok) if ok else math.inf,
                       "llm_calls_per_request": sum(len(t.find(SpanName.LLM_ATTEMPT)) for t in traces) / len(traces)}
    return out


# ============================================================================ agent trajectories
def trajectory_issues(tree: TraceTree, *, loop_threshold: int = 3) -> list[str]:
    issues = []
    tools = tree.find(SpanName.TOOL)
    fingerprints = Counter((s.attributes.get(Attr.TOOL_NAME), s.attributes.get("tool.args_fingerprint")) for s in tools)
    for (name, fp), count in fingerprints.items():
        if count >= loop_threshold:
            issues.append(f"loop: {name} called {count}x with identical arguments ({fp})")
    for s in tools:
        if s.attributes.get(Attr.TOOL_SIDE_EFFECT) and s.attributes.get(Attr.TOOL_APPROVAL) not in ("granted",):
            issues.append(f"side effect without approval: {s.attributes.get(Attr.TOOL_NAME)} "
                          f"(approval={s.attributes.get(Attr.TOOL_APPROVAL)})")
        if s.attributes.get(Attr.TOOL_STATUS) in ("error", "denied", "timeout"):
            issues.append(f"tool {s.attributes.get(Attr.TOOL_NAME)} status={s.attributes.get(Attr.TOOL_STATUS)}")
    run = tree.first(SpanName.AGENT_RUN)
    if run is not None:
        stop = run.attributes.get(Attr.AGENT_STOP)
        steps = len(tree.find(SpanName.AGENT_STEP))
        if stop in ("max_steps", "budget"):
            issues.append(f"terminated by {stop} after {steps} steps")
    return issues


def trajectory_view(tree: TraceTree) -> str:
    """Agent run as numbered steps: action, tools with argument digests, status, tokens."""
    lines = [f"trace {tree.trace_id}  tenant={tree.tenant}  route={tree.get(Attr.ROUTE)}"]
    run = tree.first(SpanName.AGENT_RUN)
    if run is not None:
        lines.append(f"agent={run.attributes.get(Attr.AGENT_NAME)} max_steps={run.attributes.get(Attr.AGENT_MAX_STEPS)} "
                     f"stop={run.attributes.get(Attr.AGENT_STOP)}")
    for step in tree.find(SpanName.AGENT_STEP):
        a = step.attributes
        lines.append(f"  step {a.get(Attr.AGENT_STEP)}: {a.get(Attr.AGENT_ACTION):<16} {step.duration_ms:7.0f} ms"
                     f"  tokens={a.get(Attr.AGENT_TOKENS, '-')}")
        for child in tree.children.get(step.span_id, []):
            c = child.attributes
            if child.name == SpanName.TOOL:
                args = c.get(f"{Attr.TOOL_ARGS}.content") or f"#{c.get(f'{Attr.TOOL_ARGS}.hash', '?')}"
                lines.append(f"      tool {c.get(Attr.TOOL_NAME)}({args}) -> {c.get(Attr.TOOL_STATUS)}"
                             f" [{c.get(Attr.TOOL_RESULT_CHARS, 0)} chars]")
            elif child.name == SpanName.GUARDRAIL:
                lines.append(f"      guard {c.get(Attr.GUARD_NAME)} -> {c.get(Attr.GUARD_DECISION)} {c.get(Attr.GUARD_REASON, '')}")
            elif child.name in (SpanName.GENERATE, SpanName.LLM_ATTEMPT):
                lines.append(f"      {child.name} in={c.get(Attr.LLM_INPUT_TOKENS)} out={c.get(Attr.LLM_OUTPUT_TOKENS)}")
    issues = trajectory_issues(tree)
    lines.append("issues: " + ("; ".join(issues) if issues else "none"))
    return "\n".join(lines)


def top_traces(store: TraceStore, key: Callable[[TraceTree], float], n: int = 5) -> list[TraceTree]:
    return sorted(store, key=key, reverse=True)[:n]


__all__ = [
    "percentile", "bootstrap_diff_ci", "evidence_path", "gold_funnel", "triage_by_stage", "TriageReport",
    "summarize", "compare_versions", "VersionComparison", "retrieved_not_cited", "EvidenceFinding",
    "latency_breakdown", "render_latency", "cost_by_tenant", "error_distribution", "trajectory_issues",
    "trajectory_view", "top_traces", "FUNNEL_STAGES",
]
