# path: book/projects/examples/ch31/metrics.py
"""Metric definitions computed from traces, and an alert evaluator for alerts.yaml.

In production these metrics are emitted as counters/histograms at request time (cheap,
always on) and the traces are sampled. Here they are derived from the trace store so the
definitions, the dashboards (dashboards.md), and the alert rules share one source of truth
and can be tested offline. Every name used in alerts.yaml must exist in METRICS.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from analysis import percentile
from semconv import Attr, ErrorClass, SpanName
from trace_store import TraceStore, TraceTree

MetricFn = Callable[[TraceStore], float]


def _rate(store: TraceStore, pred: Callable[[TraceTree], bool | None]) -> float:
    vals = [v for t in store if (v := pred(t)) is not None]
    return sum(bool(v) for v in vals) / len(vals) if vals else math.nan


def _span_rate(store: TraceStore, name: str, pred: Callable[[Any], bool]) -> float:
    spans = [s for t in store for s in t.find(name)]
    return sum(1 for s in spans if pred(s)) / len(spans) if spans else math.nan


def _per_request(store: TraceStore, fn: Callable[[TraceTree], float]) -> float:
    traces = store.traces()
    return sum(fn(t) for t in traces) / len(traces) if traces else math.nan


def _span_p(store: TraceStore, name: str, p: float) -> float:
    return percentile([s.duration_ms for t in store for s in t.find(name)], p)


# Service-level objectives (Chapter 29, reliability.slo.SLOTargets). Illustrative Northwind values.
SLO_AVAILABILITY = 0.995          # requests answered without a failed root span
SLO_COMPLETION_MS = 8000.0        # completion target
SLO_COMPLETION_OBJECTIVE = 0.95   # share of requests that must meet it


def _burn(bad_ratio: float, objective: float) -> float:
    """Error-budget burn rate: 1.0 spends exactly the budget over the SLO window."""
    return bad_ratio / (1.0 - objective) if not math.isnan(bad_ratio) else math.nan


METRICS: dict[str, MetricFn] = {
    # traffic
    "traffic.requests": lambda st: float(len(st)),
    # quality
    "quality.eval_pass_rate": lambda st: _rate(st, lambda t: t.eval_passed),
    "quality.feedback_rate": lambda st: _rate(st, lambda t: t.feedback_value is not None),
    "quality.negative_feedback_rate": lambda st: _rate(st, lambda t: None if t.feedback_value is None else t.feedback_value < 0),
    "quality.abstention_rate": lambda st: _rate(st, lambda t: bool(t.get(Attr.ABSTAINED))),
    "quality.context_truncation_rate": lambda st: _span_rate(st, SpanName.CONTEXT, lambda s: bool(s.attributes.get(Attr.CONTEXT_TRUNCATED))),
    "quality.guardrail_block_rate": lambda st: _span_rate(st, SpanName.GUARDRAIL, lambda s: s.attributes.get(Attr.GUARD_DECISION) == "block"),
    # latency
    "latency.request_p50_ms": lambda st: percentile([t.duration_ms for t in st], 50),
    "latency.request_p95_ms": lambda st: percentile([t.duration_ms for t in st], 95),
    "latency.retrieval_p95_ms": lambda st: _span_p(st, SpanName.RETRIEVAL, 95),
    "latency.llm_p95_ms": lambda st: _span_p(st, SpanName.LLM_ATTEMPT, 95),
    # SLO burn rates (Chapter 29): bad share divided by the error budget
    "slo.availability_burn_rate": lambda st: _burn(
        _rate(st, lambda t: t.root is not None and t.root.is_error), SLO_AVAILABILITY),
    "slo.completion_burn_rate": lambda st: _burn(
        # over served requests only, as in Chapter 29: a fast failure is not a fast answer
        _rate(st, lambda t: None if (t.root is not None and t.root.is_error) else t.duration_ms > SLO_COMPLETION_MS),
        SLO_COMPLETION_OBJECTIVE),
    # cost and tokens
    "cost.per_request_usd": lambda st: _per_request(st, lambda t: t.cost_usd),
    "cost.input_tokens_per_request": lambda st: _per_request(
        st, lambda t: sum(s.attributes.get(Attr.LLM_INPUT_TOKENS) or 0 for s in t.find(SpanName.GENERATE))),
    "cost.llm_calls_per_request": lambda st: _per_request(st, lambda t: float(len(t.find(SpanName.LLM_ATTEMPT)))),
    # errors
    "errors.trace_error_rate": lambda st: _rate(st, lambda t: bool(t.error_classes)),
    "errors.tool_failure_rate": lambda st: _span_rate(st, SpanName.TOOL, lambda s: s.attributes.get(Attr.TOOL_STATUS) != "ok"),
    # telemetry health: observability that silently degrades is worse than none
    "telemetry.incomplete_rate": lambda st: _rate(st, lambda t: not t.complete),
}

_CLASS_METRIC = re.compile(r"^errors\.class_rate:(?P<cls>[a-z_]+)$")


def compute(name: str, store: TraceStore) -> float:
    """Named metric, including the parametric family `errors.class_rate:<error_class>`."""
    if name in METRICS:
        return METRICS[name](store)
    m = _CLASS_METRIC.match(name)
    if m:
        cls = ErrorClass(m.group("cls")).value
        return _rate(store, lambda t: cls in t.error_classes)
    raise KeyError(f"unknown metric {name!r}")


# How many observations a metric actually rests on. A pass rate over 300 requests of which
# 12 are labeled is a 12-sample estimate, and min_samples must see 12.
SAMPLES: dict[str, Callable[[TraceStore], int]] = {
    "quality.eval_pass_rate": lambda st: sum(1 for t in st if t.eval_passed is not None),
    "quality.negative_feedback_rate": lambda st: sum(1 for t in st if t.feedback_value is not None),
    "quality.context_truncation_rate": lambda st: sum(len(t.find(SpanName.CONTEXT)) for t in st),
    "quality.guardrail_block_rate": lambda st: sum(len(t.find(SpanName.GUARDRAIL)) for t in st),
    "errors.tool_failure_rate": lambda st: sum(len(t.find(SpanName.TOOL)) for t in st),
}


def sample_count(name: str, store: TraceStore) -> int:
    return SAMPLES[name](store) if name in SAMPLES else len(store)


def is_known(name: str) -> bool:
    if name in METRICS:
        return True
    m = _CLASS_METRIC.match(name)
    return bool(m) and m.group("cls") in {e.value for e in ErrorClass}


# ============================================================================ alerts
_DURATION = re.compile(r"^(?P<n>\d+)(?P<u>[smhd])$")
_UNIT_S = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_duration(text: str) -> float:
    m = _DURATION.match(text.strip())
    if not m:
        raise ValueError(f"bad duration {text!r}; use e.g. 30m, 6h, 1d")
    return int(m.group("n")) * _UNIT_S[m.group("u")]


@dataclass
class AlertRule:
    name: str
    metric: str
    op: str                       # ">" or "<"
    window: float                 # seconds
    severity: str = "ticket"
    threshold: float | None = None
    baseline_ratio: float | None = None   # compare to the same metric over the baseline window
    baseline_window: float = 86400.0
    min_samples: int = 30
    group_by: str | None = None
    runbook: str = ""
    confirm_window: float | None = None   # multi-window rule: a short window must breach too

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AlertRule":
        cond = d["condition"]
        rule = cls(name=d["name"], metric=d["metric"], op=cond["op"], window=parse_duration(d["window"]),
                   severity=d.get("severity", "ticket"), threshold=cond.get("threshold"),
                   baseline_ratio=cond.get("baseline_ratio"),
                   baseline_window=parse_duration(d.get("baseline_window", "24h")),
                   min_samples=int(d.get("min_samples", 30)), group_by=d.get("group_by"), runbook=d.get("runbook", ""),
                   confirm_window=parse_duration(d["confirm_window"]) if d.get("confirm_window") else None)
        if rule.op not in (">", "<"):
            raise ValueError(f"{rule.name}: op must be > or <")
        if (rule.threshold is None) == (rule.baseline_ratio is None):
            raise ValueError(f"{rule.name}: set exactly one of threshold or baseline_ratio")
        if not is_known(rule.metric):
            raise ValueError(f"{rule.name}: unknown metric {rule.metric}")
        if rule.confirm_window is not None and (rule.threshold is None or rule.confirm_window >= rule.window):
            raise ValueError(f"{rule.name}: confirm_window needs an absolute threshold and must be shorter than window")
        return rule


def load_rules(path: str | Path) -> list[AlertRule]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [AlertRule.from_dict(d) for d in data["rules"]]


@dataclass
class AlertResult:
    rule: str
    group: str
    fired: bool
    value: float
    reference: float
    samples: int
    severity: str
    reason: str


def _groups(store: TraceStore, key: str | None) -> dict[str, TraceStore]:
    if key is None:
        return {"all": store}
    values = sorted({str(t.get(key)) for t in store})
    return {v: store.filter(predicate=lambda t, v=v: str(t.get(key)) == v) for v in values}


def evaluate(rules: list[AlertRule], store: TraceStore, now: float) -> list[AlertResult]:
    """Evaluate each rule over [now - window, now), per group. Baseline-relative rules compare
    with [now - window - baseline_window, now - window). Too few samples never fires: a noisy
    alert on ten requests trains people to ignore the pager. A rule with `confirm_window` fires
    only when the short window [now - confirm_window, now) breaches as well: the long window
    proves it is not a blip, the short one proves it is still happening (SLO burn-rate alerts)."""
    results = []
    for rule in rules:
        # Telemetry-health and error-class rules must also see traces that lost their root span:
        # those are the ones that reveal fragmentation or contamination.
        loose = rule.metric.startswith(("telemetry.", "errors."))
        current = store.filter(since=now - rule.window, until=now, require_root=not loose)
        baseline = store.filter(since=now - rule.window - rule.baseline_window, until=now - rule.window,
                                require_root=not loose)
        base_groups = _groups(baseline, rule.group_by)
        for group, cur in _groups(current, rule.group_by).items():
            value = compute(rule.metric, cur)
            if rule.threshold is not None:
                reference = float(rule.threshold)
            else:
                base = base_groups.get(group)
                enough = base is not None and len(base) and sample_count(rule.metric, base) >= rule.min_samples
                base_value = compute(rule.metric, base) if enough else math.nan   # a thin baseline is no reference
                reference = base_value * float(rule.baseline_ratio or 1.0)
            samples = sample_count(rule.metric, cur)
            if samples < rule.min_samples or math.isnan(value) or math.isnan(reference):
                results.append(AlertResult(rule.name, group, False, value, reference, samples, rule.severity, "insufficient data"))
                continue
            fired = value > reference if rule.op == ">" else value < reference
            if fired and rule.confirm_window is not None:
                recent = _groups(store.filter(since=now - rule.confirm_window, until=now, require_root=not loose),
                                 rule.group_by).get(group)
                # the short window also needs enough samples: one failed request must not confirm a page
                min_short = max(1, rule.min_samples // 4)
                short = (compute(rule.metric, recent)
                         if recent is not None and len(recent) and sample_count(rule.metric, recent) >= min_short
                         else math.nan)
                confirmed = not math.isnan(short) and (short > reference if rule.op == ">" else short < reference)
                if not confirmed:
                    results.append(AlertResult(rule.name, group, False, value, reference, samples, rule.severity,
                                               f"long window breached, short window {short:.4g} did not"))
                    continue
            results.append(AlertResult(rule.name, group, fired, value, reference, samples, rule.severity,
                                       f"{rule.metric}={value:.4g} {rule.op} {reference:.4g}" if fired else "ok"))
    return results


__all__ = ["METRICS", "SAMPLES", "sample_count", "compute", "is_known", "AlertRule", "AlertResult", "load_rules", "evaluate", "parse_duration"]
