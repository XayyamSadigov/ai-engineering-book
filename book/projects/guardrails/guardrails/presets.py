# path: book/projects/guardrails/guardrails/presets.py
"""Ready-made pipelines for the two Northwind surfaces threat-modeled in Chapter 26.

`rag_pipeline` implements the Project 3 control list (read-only knowledge assistant);
`agent_pipeline` adds the Project 4 tool controls (support agent with `send_reply`).
Order inside each stage: cheap and deterministic first, model-based last.
"""
from __future__ import annotations

from typing import Iterable

from aie_core.observability import Tracer

from .context import ContextSanitizerCheck
from .input import InjectionHeuristicCheck, LLMInjectionClassifier, SizeLimitCheck
from .moderation import KeywordModerator, ModerationCheck, Moderator
from .output import ActiveContentCheck, CanaryCheck, CitationCheck, UrlAllowlistCheck
from .pii import PIIRedactionCheck
from .pipeline import Action, Check, GuardrailPipeline, Stage
from .secrets import SecretsCheck
from .tools import ToolPolicyCheck, ToolRule, matches, max_length, one_of, recipient_domains

NORTHWIND_HOSTS: tuple[str, ...] = ("intranet.northwind.example", "docs.northwind.example")
NORTHWIND_MAIL_DOMAINS: tuple[str, ...] = ("northwind.example",)


def rag_checks(*, allowed_hosts: Iterable[str] = NORTHWIND_HOSTS, canaries: Iterable[str] = (),
               require_citations: bool = True, moderator: Moderator | None = None,
               classifier: LLMInjectionClassifier | None = None, pii_mode: str = "tokenize") -> list[Check]:
    checks: list[Check] = [
        # INPUT
        SizeLimitCheck(max_chars=8_000, max_tokens=2_000),
        SecretsCheck(on_detect=Action.REDACT, stages=frozenset({Stage.INPUT, Stage.CONTEXT}), name="secrets_in"),
        PIIRedactionCheck(mode=pii_mode, stages=frozenset({Stage.INPUT})),  # type: ignore[arg-type]
        InjectionHeuristicCheck(stages=frozenset({Stage.INPUT, Stage.CONTEXT})),
        ModerationCheck(moderator or KeywordModerator(), stages=frozenset({Stage.INPUT, Stage.OUTPUT})),
        # CONTEXT (sanitizer last so the heuristic above saw the original carriers)
        ContextSanitizerCheck(),
        # OUTPUT
        CanaryCheck(canaries, stages=frozenset({Stage.OUTPUT, Stage.TOOL})),
        SecretsCheck(on_detect=Action.BLOCK, stages=frozenset({Stage.OUTPUT}), name="secrets_out"),
        UrlAllowlistCheck(allowed_hosts),
        ActiveContentCheck(),
    ]
    if require_citations:
        checks.append(CitationCheck())
    if classifier is not None:
        checks.append(classifier)          # most expensive, so last among input checks
    return checks


def rag_pipeline(tracer: Tracer | None = None, **kwargs) -> GuardrailPipeline:
    return GuardrailPipeline(rag_checks(**kwargs), tracer=tracer)


TICKET_ID_PATTERN = r"^TCK-\d{4}-\d{4}$"
PRIORITIES = ("P1", "P2", "P3", "P4")


def support_tool_rules(mail_domains: Iterable[str] = NORTHWIND_MAIL_DOMAINS, *,
                       require_send_approval: bool = True) -> list[ToolRule]:
    """Tool rules for Project 4's six support tools, with Project 4's argument names and limits
    (`support_assistant/tools.py`); a test runs P4's real tool specs through these rules.

    E-mail arguments are re-hydrated from PII tokens at the tool boundary (`guard_tool_call`).
    Set `require_send_approval=False` when an upstream tool layer (Chapter 16's ApprovalManager)
    already binds approval to the exact arguments, so the user is not asked twice.
    """
    mail_domains = tuple(mail_domains)
    return [
        ToolRule("lookup_employee", required_args=("query",), constraints={"query": max_length(100)},
                 rehydrate_args=("query",)),
        ToolRule("search_tickets", required_args=("query",),
                 constraints={"query": max_length(200), "status": one_of("open", "closed", "any")}),
        ToolRule("get_service_status", required_args=("service",)),
        ToolRule("create_ticket", required_args=("subject", "body", "category", "priority"),
                 constraints={"subject": max_length(120), "body": max_length(2_000), "priority": one_of(*PRIORITIES)}),
        ToolRule("draft_reply", required_args=("ticket_id", "to", "subject", "body"), rehydrate_args=("to",),
                 constraints={"ticket_id": matches(TICKET_ID_PATTERN), "to": recipient_domains(mail_domains),
                              "subject": max_length(150), "body": max_length(4_000)}),
        ToolRule("send_reply", outbound=True, requires_approval=require_send_approval,
                 required_args=("ticket_id", "to", "subject", "body"), rehydrate_args=("to",),
                 constraints={"ticket_id": matches(TICKET_ID_PATTERN), "to": recipient_domains(mail_domains),
                              "subject": max_length(150), "body": max_length(4_000)}),
    ]


def agent_pipeline(tracer: Tracer | None = None, *, canaries: Iterable[str] = (),
                   max_calls_per_request: int = 8, **kwargs) -> GuardrailPipeline:
    canaries = tuple(canaries)
    checks = rag_checks(canaries=canaries, require_citations=False, **kwargs)
    checks.append(ToolPolicyCheck(support_tool_rules(), canaries=canaries,
                                  max_calls_per_request=max_calls_per_request))
    return GuardrailPipeline(checks, tracer=tracer)


__all__ = ["NORTHWIND_HOSTS", "NORTHWIND_MAIL_DOMAINS", "TICKET_ID_PATTERN", "rag_checks", "rag_pipeline", "support_tool_rules",
           "agent_pipeline"]
