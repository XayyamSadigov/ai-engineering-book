# path: book/projects/p5-incident-agent/incident_agent/service.py
"""Wiring and use cases: investigate an alert, decide on publication, read an investigation.

The CLI and the HTTP API are thin shells over this class. All state lives in three durable
places under P5_STATE_DIR: investigation records, agentkit event logs (one JSONL per run),
and the channel log. A decision can therefore be made by a different process than the one
that ran the investigation, which is the normal case for a human approval.
"""
from __future__ import annotations

import uuid
from typing import Any

from aie_core.llm.client import LLMClient
from aie_core.llm.errors import LLMError
from aie_core.observability import NoopTracer, Tracer
from aie_core.settings import Settings, make_llm_client
from agentkit import Budget, EventStore, JsonlEventStore, TerminationReason

from .adapters.channel import Channel, JsonlChannel
from .adapters.corpus import KnowledgeBase
from .adapters.metering import MeteredLLM
from .adapters.scripted import scripted_llm
from .adapters.store import FileInvestigationStore, InvestigationStore
from .adapters.telemetry import Telemetry
from .agent import AgentLimits, IncidentResearchAgent, Publisher
from .config import DIRECTORY, P5Settings
from .domain.models import Investigation, Status
from .judge import ReportJudge


class NotAllowed(PermissionError):
    """The caller may not perform this action (unknown user, wrong tenant, missing group)."""


class InvalidState(RuntimeError):
    """The investigation is not in a state that allows this action."""


def default_llm() -> LLMClient:
    """Real provider when configured; the scripted stand-in when LLM_PROVIDER=fake (the default)."""
    settings = Settings()
    return scripted_llm() if settings.llm_provider == "fake" else make_llm_client(settings)


class IncidentService:
    def __init__(self, settings: P5Settings | None = None, *, llm: LLMClient | None = None,
                 judge_llm: LLMClient | None = None, channel: Channel | None = None,
                 store: InvestigationStore | None = None, event_store: EventStore | None = None,
                 kb: KnowledgeBase | None = None, telemetry: Telemetry | None = None,
                 tracer: Tracer | None = None) -> None:
        self.settings = s = settings or P5Settings()
        self.kb = kb or KnowledgeBase(s.shared_data_dir / "docs", chunk_tokens=s.chunk_tokens)
        self.telemetry = telemetry or Telemetry(s.data_dir)
        self.channel = channel or JsonlChannel(s.state_dir / "channel.jsonl")
        self.store = store or FileInvestigationStore(s.state_dir / "investigations")
        self.event_store = event_store or JsonlEventStore(s.state_dir / "events")
        self.llm = llm or default_llm()
        self.judge_llm = judge_llm or self.llm
        self.tracer = tracer or NoopTracer()
        self.limits = AgentLimits(max_plan_steps=s.max_plan_steps, max_replans=s.max_replans,
                                  max_revisions=s.max_revisions, min_improvement=s.min_improvement,
                                  step_budget=Budget(max_steps=s.step_max_steps, max_tool_calls=s.step_max_tool_calls))
        self.publisher = Publisher(self.channel, self.store.get, self.event_store, s.channel, self.tracer)

    # ------------------------------------------------------------------ helpers
    def principal(self, user: str) -> dict[str, Any]:
        if user not in DIRECTORY:
            raise NotAllowed(f"unknown user {user!r}")
        return dict(DIRECTORY[user])

    def get(self, inv_id: str) -> Investigation:
        inv = self.store.get(inv_id)
        if inv is None:
            raise KeyError(inv_id)
        return inv

    # ---------------------------------------------------------------- use cases
    def investigate(self, alert_id: str, user: str, *, inv_id: str | None = None) -> Investigation:
        principal = self.principal(user)
        alert = self.telemetry.alert(alert_id)                       # KeyError for unknown alerts
        if principal["tenant"] != alert.tenant:
            raise NotAllowed(f"{user} works in tenant {principal['tenant']}, alert is in {alert.tenant}")
        inv = Investigation(id=inv_id or f"inv-{alert.id.lower()}-{uuid.uuid4().hex[:6]}", alert=alert,
                            requested_by=user, principal=principal)
        metered = MeteredLLM(self.llm, self.settings.max_llm_calls)
        judge_client = metered if self.judge_llm is self.llm else metered.share(self.judge_llm)   # one ceiling
        agent = IncidentResearchAgent(metered, ReportJudge(judge_client, pass_score=self.settings.judge_pass_score,
                                                           model=self.settings.judge_model),
                                      self.kb, self.telemetry, event_store=self.event_store, limits=self.limits,
                                      tracer=self.tracer)
        try:
            agent.investigate(inv)
        except LLMError as exc:                          # includes the call-budget ceiling
            inv.status, inv.detail = Status.FAILED, f"{type(exc).__name__}: {exc}"
        inv.usage = metered.snapshot()
        self.store.save(inv)                             # the publish tool reads the report from here
        if inv.status is Status.AWAITING_APPROVAL:
            run = self.publisher.request(inv)
            if run.stop_reason is not TerminationReason.APPROVAL_REQUIRED:   # never post without a human
                inv.status, inv.detail = Status.FAILED, f"publish run did not pause: {run.stop_reason}"
            inv.publish_run_id = run.run_id
            self.store.save(inv)
        return inv

    def decide(self, inv_id: str, user: str, *, approve: bool, reason: str = "") -> Investigation:
        reviewer = self.principal(user)
        inv = self.get(inv_id)
        if reviewer["tenant"] != inv.alert.tenant:   # another tenant's investigation looks like an unknown id
            raise KeyError(inv_id)
        if inv.status is not Status.AWAITING_APPROVAL or inv.publish_run_id is None:
            raise InvalidState(f"investigation {inv_id} is {inv.status.value}; only awaiting_approval can be decided")
        if "it-oncall" not in reviewer["groups"]:
            raise NotAllowed(f"{user} may not approve reports for tenant {inv.alert.tenant}")
        run = self.publisher.decide(inv.publish_run_id, approve=approve, reviewer=user, reason=reason)
        inv.decided_by = user
        if approve and run.ok and "message_id" in run.state.artifacts:
            inv.status, inv.posted_message_id = Status.PUBLISHED, str(run.state.artifacts["message_id"])
        elif not approve:
            inv.status, inv.detail = Status.REJECTED, reason or "rejected by reviewer"
        else:
            inv.status, inv.detail = Status.FAILED, f"publish failed: {run.stop_reason}: {run.detail}"
        self.store.save(inv)
        return inv


__all__ = ["IncidentService", "NotAllowed", "InvalidState", "default_llm"]
