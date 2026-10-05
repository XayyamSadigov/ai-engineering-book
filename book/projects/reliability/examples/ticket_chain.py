# path: book/projects/reliability/examples/ticket_chain.py
"""Northwind Assist ticket triage: a three-step chain wired with every reliability primitive.

    classify (critical)  -> structured output, repair once, small-model fallback
    retrieve (optional)  -> runbook search behind a breaker, idempotent so retried on budget
    draft    (critical)  -> reply text from the gateway; template fallback if all models fail

The gateway wraps each model client in a CircuitBreakerClient, so an open primary circuit
sends calls straight to the backup. The degrade policy reads open circuits and the
admission level to pick a plan (retrieval depth, output cap, model tier) before the chain
starts. The chaos tests drive this module with injected faults.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from aie_core.llm.client import LLMClient
from aie_core.llm.gateway import ModelGateway, RetryPolicy
from aie_core.llm.types import CompletionRequest, Message
from pydantic import BaseModel, Field

from reliability import (
    ChainResult,
    ChainStatus,
    StepRecord,
    StepStatus,
    CircuitBreakerClient,
    CircuitBreakerRegistry,
    Deadline,
    DegradedPlan,
    DegradePolicy,
    RetryBudget,
    Step,
    complete_with_recovery,
    run_chain,
)
from reliability.clock import Clock, Sleep


class TicketClass(BaseModel):
    category: Literal["network", "access", "hardware", "billing", "other"]
    priority: int = Field(ge=1, le=4)
    summary: str


@dataclass
class TriageServices:
    gateway: LLMClient
    retrieve: Callable[[str, int], list[str]]
    breakers: CircuitBreakerRegistry
    retry_budget: RetryBudget
    policy: DegradePolicy = field(default_factory=DegradePolicy)
    small_model: LLMClient | None = None
    models: dict[str, str] = field(default_factory=lambda: {"primary": "assist-large", "small": "assist-small"})
    clock: Clock = time.monotonic
    sleep: Sleep = time.sleep


def build_gateway(primary: LLMClient, backup: LLMClient, breakers: CircuitBreakerRegistry, *,
                  clock: Clock = time.monotonic, sleep: Sleep = time.sleep) -> ModelGateway:
    """Breakers go *inside* the gateway, one per client, named after the failure domain.

    `backup` serves the same model through another provider or region, so the fallback does
    not change capabilities. Fallbacks that do (smaller context, no tools) are routes with
    compatibility checks in the Router (Chapter 7), not gateway fallbacks.
    """
    return ModelGateway(
        primary=CircuitBreakerClient(primary, breakers.get("llm:primary")),
        fallbacks=[CircuitBreakerClient(backup, breakers.get("llm:backup"))],
        retry=RetryPolicy(max_attempts=2, base_delay_s=0.2, max_delay_s=1.0),
        clock=clock, sleep=sleep, default_timeout_s=8.0,
    )


CLASSIFY_SYSTEM = "Classify the Northwind support ticket. Return category, priority 1-4, and a one-line summary."
DRAFT_SYSTEM = "Draft a short, polite reply to the employee. Use the runbook excerpts when relevant."


def _classify(svc: TriageServices, plan: DegradedPlan) -> Callable[[dict[str, Any], Deadline], Any]:
    def run(state: dict[str, Any], deadline: Deadline) -> Any:
        req = CompletionRequest(messages=[Message.system(CLASSIFY_SYSTEM), Message.user(state["ticket"]["text"])],
                                max_tokens=200, metadata={"step": "classify"})
        req = deadline.apply(plan.apply(req, svc.models))
        return complete_with_recovery(svc.gateway, req, TicketClass, repair_attempts=1, fallback_client=svc.small_model)
    return run


def _retrieve(svc: TriageServices, plan: DegradedPlan) -> Callable[[dict[str, Any], Deadline], Any]:
    def run(state: dict[str, Any], deadline: Deadline) -> list[str]:
        if plan.retrieval_k == 0:
            return []
        deadline.check("retrieve")
        cls = state["classify"].value
        return svc.retrieve(f"{cls.category} {cls.summary}", plan.retrieval_k)
    return run


def _draft(svc: TriageServices, plan: DegradedPlan) -> Callable[[dict[str, Any], Deadline], Any]:
    def run(state: dict[str, Any], deadline: Deadline) -> str:
        evidence = state.get("retrieve") or []
        cls = state["classify"].value
        user = json.dumps({"ticket": state["ticket"]["text"], "category": cls.category, "runbooks": evidence})
        req = CompletionRequest(messages=[Message.system(DRAFT_SYSTEM), Message.user(user)],
                                max_tokens=600, metadata={"step": "draft"})
        return svc.gateway.complete(deadline.apply(plan.apply(req, svc.models))).text
    return run


def _template_reply(state: dict[str, Any], exc: BaseException) -> str:
    cls = state["classify"].value
    return (f"Thanks for contacting Northwind IT. Your ticket was filed as '{cls.category}' "
            f"(priority {cls.priority}). An engineer will follow up; no action is needed from you yet.")


def triage_ticket(ticket: dict[str, Any], svc: TriageServices, deadline: Deadline,
                  admission_level: int = 0) -> ChainResult:
    plan = svc.policy.resolve(admission_level, svc.breakers.open_dependencies())
    state: dict[str, Any] = {"ticket": ticket, "plan": plan}
    if not plan.use_model:
        # Static mode: every model circuit is open. Do not knock on closed doors; answer with
        # the canned path (help articles, "we will follow up") and let probes heal the circuits.
        return ChainResult(status=ChainStatus.FAILED, state=state,
                           steps=[StepRecord(n, StepStatus.SKIPPED, error="static mode: all model circuits open")
                                  for n in ("classify", "retrieve", "draft")])
    steps = [
        Step("classify", _classify(svc, plan), critical=True, budget_s=3.0, min_s=0.5),
        Step("retrieve", _retrieve(svc, plan), critical=False, budget_s=1.5, min_s=0.2,
             dependency="retrieval", idempotent=True),
        Step("draft", _draft(svc, plan), critical=True, min_s=0.5, fallback=_template_reply),
    ]
    return run_chain(steps, state, deadline, breakers=svc.breakers, retry_budget=svc.retry_budget,
                     retry_policy=RetryPolicy(max_attempts=2, base_delay_s=0.1, max_delay_s=0.5),
                     clock=svc.clock, sleep=svc.sleep)
