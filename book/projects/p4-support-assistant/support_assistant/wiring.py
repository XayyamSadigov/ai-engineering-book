# path: book/projects/p4-support-assistant/support_assistant/wiring.py
"""Composition root: builds every object once, so tests and the API share one wiring."""
from __future__ import annotations

from dataclasses import dataclass

from aie_core.llm.client import LLMClient
from aie_core.observability import Tracer, get_tracer
from aie_core.settings import Settings as CoreSettings
from aie_core.settings import make_llm_client
from toolkit import (ApprovalManager, AuditSink, InMemoryAuditLog, JsonlAuditLog, SQLiteIdempotencyStore,
                     ToolExecutor, ToolLoop, ToolRegistry)

from .adapters.demo_llm import make_demo_llm
from .config import AssistantSettings
from .domain.directory import Directory
from .domain.services import DraftStore, Outbox, StatusBoard
from .domain.tickets import TicketStore, load_shared_tickets
from .tools import Backends, build_policy, build_registry, summarize_for_approval


@dataclass
class Container:
    settings: AssistantSettings
    backends: Backends
    registry: ToolRegistry
    executor: ToolExecutor
    approvals: ApprovalManager
    audit: AuditSink
    loop: ToolLoop
    llm: LLMClient


def build_container(settings: AssistantSettings | None = None, *, llm: LLMClient | None = None,
                    tracer: Tracer | None = None, sleep=None) -> Container:
    settings = settings or AssistantSettings()
    backends = Backends(
        directory=Directory(),
        tickets=TicketStore(load_shared_tickets(settings.shared_data_dir, settings.extra_tickets_path)),
        status=StatusBoard(), drafts=DraftStore(), outbox=Outbox(),
    )
    registry = build_registry(backends)
    approvals = ApprovalManager(ttl_s=settings.approval_ttl_s, allow_self_approval=not settings.four_eyes)
    audit: AuditSink = JsonlAuditLog(settings.audit_log_path) if settings.audit_log_path else InMemoryAuditLog()
    tracer = tracer or get_tracer()
    extra = {"sleep": sleep} if sleep is not None else {}
    executor = ToolExecutor(registry, build_policy(settings), approvals=approvals,
                            idempotency=SQLiteIdempotencyStore(settings.idempotency_db), audit=audit,
                            tracer=tracer, max_attempts=settings.tool_max_attempts,
                            summarize=summarize_for_approval, **extra)
    if llm is None:
        core = CoreSettings()
        llm = make_demo_llm() if core.llm_provider == "fake" and settings.demo_llm else make_llm_client(core)
    loop = ToolLoop(llm, executor, max_rounds=settings.max_rounds, tracer=tracer)
    return Container(settings, backends, registry, executor, approvals, audit, loop, llm)


__all__ = ["Container", "build_container"]
