# path: book/projects/p3-rag-assistant/rag_assistant/observability/metrics.py
"""Stage metrics: counters and latency/size distributions with labels, rendered for Prometheus.

Traces (aie_core spans) explain one request; metrics show the population. Every stage of the
request path and the ingestion path reports here, with the same stage names as the spans:

    rag_requests_total{mode, cache}            rag_stage_latency_ms{stage}
    rag_degraded_total{reason}                 rag_security_events_total{kind}
    rag_ingest_jobs_total{change}              rag_freshness_lag_s
    rag_cache_lookups_total{cache, result}     rag_evidence_tokens

The reservoir keeps the last N observations per series, which is enough for a single replica's
dashboard and tests. A deployment exports to Prometheus or OpenTelemetry instead (Chapter 31).
"""
from __future__ import annotations

import threading
from collections import defaultdict, deque
from typing import Any

Labels = tuple[tuple[str, str], ...]


def _labels(kw: dict[str, Any]) -> Labels:
    return tuple(sorted((k, str(v)) for k, v in kw.items()))


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


class Metrics:
    def __init__(self, reservoir: int = 2000) -> None:
        self._counters: dict[str, dict[Labels, float]] = defaultdict(lambda: defaultdict(float))
        self._dists: dict[str, dict[Labels, deque[float]]] = defaultdict(dict)
        self._reservoir = reservoir
        self._lock = threading.Lock()

    def inc(self, name: str, value: float = 1.0, **labels: Any) -> None:
        with self._lock:
            self._counters[name][_labels(labels)] += value

    def observe(self, name: str, value: float, **labels: Any) -> None:
        with self._lock:
            series = self._dists[name].setdefault(_labels(labels), deque(maxlen=self._reservoir))
            series.append(float(value))

    def counter(self, name: str, **labels: Any) -> float:
        with self._lock:
            return self._counters.get(name, {}).get(_labels(labels), 0.0)

    def values(self, name: str, **labels: Any) -> list[float]:
        with self._lock:
            return list(self._dists.get(name, {}).get(_labels(labels), []))

    def all_values(self, name: str) -> list[float]:
        with self._lock:
            return [v for series in self._dists.get(name, {}).values() for v in series]

    def render_prometheus(self) -> str:
        lines: list[str] = []
        with self._lock:
            for name, series in sorted(self._counters.items()):
                lines.append(f"# TYPE {name} counter")
                for labels, value in sorted(series.items()):
                    lines.append(f"{name}{_fmt(labels)} {value:g}")
            for name, series in sorted(self._dists.items()):
                lines.append(f"# TYPE {name} summary")
                for labels, values in sorted(series.items()):
                    vals = list(values)
                    for q in (0.5, 0.95, 0.99):
                        lab = _fmt(labels + (("quantile", str(q)),))
                        lines.append(f"{name}{lab} {quantile(vals, q) or 0:g}")
                    lines.append(f"{name}_count{_fmt(labels)} {len(vals)}")
        return "\n".join(lines) + "\n"


def _fmt(labels: Labels) -> str:
    if not labels:
        return ""
    return "{" + ",".join(f'{k}="{v}"' for k, v in labels) + "}"


__all__ = ["Metrics", "quantile"]
