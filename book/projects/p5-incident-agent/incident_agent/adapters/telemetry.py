# path: book/projects/p5-incident-agent/incident_agent/adapters/telemetry.py
"""Fake observability backend: alerts, metric time series with a simple anomaly rule, deploys.

Anomaly rule (illustrative, deliberately simple): baseline is the median of the first six
points; a metric is anomalous at time t when its value is at least `ratio` times the
baseline (or at most 1/ratio for metrics where lower is worse) and past its absolute floor.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ..domain.models import Alert


@dataclass(frozen=True)
class MetricReading:
    name: str
    unit: str
    baseline: float
    value: float
    anomalous: bool
    onset: datetime | None

    @property
    def source_id(self) -> str:
        return f"metric:{self.name}"

    def describe(self, at: datetime) -> str:
        ratio = self.value / self.baseline if self.baseline else 0.0
        text = (f"baseline {self.baseline:g}{self.unit}, at {at:%H:%M} UTC {self.value:g}{self.unit} "
                f"(x{ratio:.1f})")
        return text + (f", anomalous since {self.onset:%H:%M} UTC" if self.anomalous and self.onset else ", normal")


class Telemetry:
    def __init__(self, data_dir: Path, ratio: float = 2.0) -> None:
        raw = json.loads((data_dir / "telemetry.json").read_text(encoding="utf-8"))
        self.alerts = {a["id"]: Alert.model_validate(a)
                       for a in json.loads((data_dir / "alerts.json").read_text(encoding="utf-8"))}
        self.start = datetime.fromisoformat(raw["start"].replace("Z", "+00:00"))
        self.step = timedelta(minutes=raw["step_minutes"])
        self.services: dict[str, dict[str, Any]] = raw["services"]
        self.metrics: dict[str, dict[str, Any]] = raw["metrics"]
        self.deploys: list[dict[str, Any]] = raw["deploys"]
        self.ratio = ratio
        self.queries = 0                                  # tests use this to prove replay executes nothing

    def alert(self, alert_id: str) -> Alert:
        if alert_id not in self.alerts:
            raise KeyError(f"unknown alert {alert_id!r}")
        return self.alerts[alert_id]

    def known_service(self, service: str) -> bool:
        return service in self.services

    def dependencies(self, service: str) -> list[str]:
        return list(self.services.get(service, {}).get("depends_on", []))

    def read(self, service: str, as_of: datetime) -> list[MetricReading]:
        self.queries += 1
        idx = int((as_of - self.start) / self.step)
        out = []
        for name, spec in sorted(self.metrics.items()):
            if not name.startswith(service + "."):
                continue
            values = spec["values"][: max(idx + 1, 1)]
            baseline = statistics.median(spec["values"][:6])
            down = spec.get("direction") == "down"
            floor = spec.get("floor")

            def bad(v: float) -> bool:
                over = v <= baseline / self.ratio if down else v >= baseline * self.ratio
                return over and (floor is None or (v <= floor if down else v >= floor))

            onset = next((self.start + i * self.step for i, v in enumerate(values) if bad(v)), None)
            out.append(MetricReading(name, spec.get("unit", ""), baseline, values[-1], bad(values[-1]), onset))
        return out

    def recent_deploys(self, service: str, as_of: datetime, hours: int) -> list[dict[str, Any]]:
        self.queries += 1
        lo = as_of - timedelta(hours=hours)
        found = []
        for d in self.deploys:
            at = datetime.fromisoformat(d["at"].replace("Z", "+00:00"))
            if d["service"] == service and lo <= at <= as_of:
                found.append({**d, "at_dt": at})
        return sorted(found, key=lambda d: d["at_dt"])


__all__ = ["Telemetry", "MetricReading"]
