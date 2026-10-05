# path: book/capstone/northwind-assist/northwind_assist/cost/ledger.py
"""Cost per request and per tenant (Chapter 30).

Before a request may spend, SpendGuard reserves a worst-case estimate against the tenant's daily
limit and answers allow, degrade (past the soft limit) or block. After the request, the actual
cost from the usage meter replaces the reservation. The ledger keeps one row per request, so the
daily report can show cost per *successful* answer, which is the number a budget owner cares
about (failed and abstained requests still cost money and are charged to the successes).
"""
from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from budgets import Alert, Decision, SpendGuard, SpendPolicy, utc_day  # type: ignore[import-not-found]


@dataclass
class CostRow:
    ts: float
    day: str
    tenant: str
    user_id: str
    request_id: str
    intent: str
    models: list[str]
    input_tokens: int
    output_tokens: int
    cost_usd: float
    success: bool
    cache_hit: bool
    degraded: int


@dataclass
class TenantDay:
    tenant: str
    day: str
    requests: int = 0
    successes: int = 0
    cache_hits: int = 0
    cost_usd: float = 0.0
    by_intent: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def as_dict(self, limit: float | None) -> dict[str, Any]:
        per_success = self.cost_usd / self.successes if self.successes else None
        return {"tenant": self.tenant, "day": self.day, "requests": self.requests, "successes": self.successes,
                "cache_hit_rate": round(self.cache_hits / self.requests, 3) if self.requests else 0.0,
                "cost_usd": round(self.cost_usd, 6),
                "cost_per_successful_answer_usd": round(per_success, 6) if per_success is not None else None,
                "daily_limit_usd": limit, "utilization": round(self.cost_usd / limit, 4) if limit else None,
                "by_intent": {k: round(v, 6) for k, v in sorted(self.by_intent.items())}}


class CostLedger:
    def __init__(self, daily_limits: dict[str, float], *, thresholds: tuple[float, ...] = (0.5, 0.8, 1.0),
                 path: str | None = None, clock: Callable[[], float] = time.time) -> None:
        self.alerts: list[Alert] = []
        self.limits = dict(daily_limits)
        policies = {t: SpendPolicy(daily_limit_usd=v, mode="degrade", alert_thresholds=thresholds)
                    for t, v in daily_limits.items()}
        # Unknown tenants (eval principals, a tenant added before its budget) get the smallest limit.
        default = SpendPolicy(daily_limit_usd=min(daily_limits.values(), default=1.0), mode="degrade",
                              alert_thresholds=thresholds)
        self.guard = SpendGuard(policies, default_policy=default, on_alert=self.alerts.append, clock=clock)
        self.rows: list[CostRow] = []
        self.path = Path(path) if path else None
        self._clock = clock
        self._lock = threading.Lock()

    def reserve(self, tenant: str, estimate_usd: float) -> Decision:
        return self.guard.reserve(tenant, estimate_usd)

    def settle(self, decision: Decision, row: CostRow) -> None:
        if decision.reservation is not None:
            self.guard.commit(decision.reservation, row.cost_usd)
        with self._lock:
            self.rows.append(row)
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(row)) + "\n")

    def row(self, **fields: Any) -> CostRow:
        ts = self._clock()
        return CostRow(ts=ts, day=utc_day(ts), **fields)

    def daily_report(self, day: str | None = None) -> dict[str, Any]:
        day = day or utc_day(self._clock())
        per: dict[str, TenantDay] = {}
        with self._lock:
            rows = [r for r in self.rows if r.day == day]
        for r in rows:
            t = per.setdefault(r.tenant, TenantDay(r.tenant, day))
            t.requests += 1
            t.successes += int(r.success)
            t.cache_hits += int(r.cache_hit)
            t.cost_usd += r.cost_usd
            t.by_intent[r.intent] += r.cost_usd
        alerts = [asdict(a) for a in self.alerts if a.day == day]
        return {"day": day, "tenants": [per[t].as_dict(self.limits.get(t)) for t in sorted(per)],
                "total_cost_usd": round(sum(r.cost_usd for r in rows), 6), "alerts": alerts}


__all__ = ["CostLedger", "CostRow"]
