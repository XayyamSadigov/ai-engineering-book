# path: book/capstone/northwind-assist/northwind_assist/resilience/wiring.py
"""Admission control, degraded plans and flags (Chapters 29 and 32) in one decision per request.

The plan a request runs under is the most restrictive of four inputs: the admission controller's
load level for this replica, the open circuit breakers, the tenant's spend position, and the
operator's switches (forced level, read-only mode, kill switches in the flag file). Deciding it
once, up front, means every stage reads one plan instead of re-asking "are we degraded?".
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from northwind_triage.flags import FlagEvaluator  # type: ignore[import-not-found]
from reliability import (
    AdmissionConfig,
    AdmissionController,
    AdmissionDecision,
    AdmissionRequest,
    DegradedPlan,
    DegradeLevel,
    DegradePolicy,
    Priority,
    TenantQuota,
)

from .. import _paths
from ..config import Settings

DEFAULT_FLAGS = _paths.CAPSTONE_ROOT / "northwind_assist" / "flags.json"


@dataclass
class Plan:
    plan: DegradedPlan
    admission: AdmissionDecision
    flags: dict[str, str]

    @property
    def level(self) -> int:
        return int(self.plan.level)


class Resilience:
    def __init__(self, settings: Settings, *, clock: Any = None,
                 model_dependencies: tuple[str, ...] = ("llm:primary",)) -> None:
        quota = TenantQuota(requests_per_minute=settings.tenant_requests_per_minute)
        cfg = AdmissionConfig(capacity=settings.admission_capacity, max_queue=settings.admission_max_queue,
                              default_quota=quota, initial_service_s=2.0)
        self.admission = AdmissionController(cfg, **({"clock": clock} if clock is not None else {}))
        self.degrade = DegradePolicy(model_dependencies=model_dependencies)
        path = Path(settings.flags_path) if settings.flags_path else DEFAULT_FLAGS
        self.flags = FlagEvaluator.from_dict(json.loads(path.read_text(encoding="utf-8")), settings.environment)

    def admit(self, tenant: str, est_tokens: int, deadline_s: float | None) -> AdmissionDecision:
        return self.admission.admit(AdmissionRequest(tenant_id=tenant, priority=Priority.INTERACTIVE,
                                                     est_tokens=est_tokens, deadline_s=deadline_s))

    def release(self, decision: AdmissionDecision, service_s: float) -> None:
        self.admission.release(decision, service_s)

    def plan(self, decision: AdmissionDecision, *, open_dependencies: set[str], spend_action: str,
             user_id: str) -> Plan:
        # DegradePolicy labels its first argument "admission:<level>"; pass only the admission level
        # there and record the spend position as its own reason, so lineage says why a plan shrank.
        plan = self.degrade.resolve(decision.degrade_level, open_dependencies)
        if spend_action == "degrade" and plan.level < DegradeLevel.REDUCED:
            reduced = self.degrade.resolve(int(DegradeLevel.REDUCED), open_dependencies)
            reasons = [r for r in reduced.reasons if not r.startswith("admission:")]
            plan = reduced.model_copy(update={"reasons": [*plan.reasons, *reasons, "spend:soft_limit"]})
        flags = {name: self.flags.evaluate(name, user_id).variant for name in ("rag.rerank", "agent.actions",
                                                                              "rag.answer_cache")}
        if flags["rag.rerank"] == "off":
            plan = plan.model_copy(update={"rerank": False, "reasons": [*plan.reasons, "flag:rag.rerank=off"]})
        if flags["agent.actions"] == "off":
            plan = plan.model_copy(update={"allow_agents": False, "reasons": [*plan.reasons, "flag:agent.actions=off"]})
        return Plan(plan=plan, admission=decision, flags=flags)


__all__ = ["Resilience", "Plan", "DEFAULT_FLAGS"]
