# path: book/capstone/northwind-assist/northwind_assist/security/guards.py
"""Guardrails at four stages, composed from Chapter 27's presets.

The preset `rag_checks` supplies the input, context and output checks unchanged except for one
substitution: the context sanitizer runs with `wrap=False`, because ragkit's EvidencePacker
already wraps every evidence block in an <untrusted_data> tag and a second wrapper would nest
tags. Tool rules are the preset `support_tool_rules`, which uses Project 4's argument names, with
`require_send_approval=False`: toolkit's ApprovalManager already binds approval to the exact
argument hash, so the user is not asked twice. The tool stage goes through `guard_tool_call`,
which re-hydrates the rule's PII-token arguments (e-mail only) inside the tool boundary and
checks the re-hydrated call; the caller executes that call, never the model's original.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aie_core.llm.types import ToolCall
from guardrails import GuardContext, GuardrailPipeline, guard_tool_call, rehydrate_kinds
from guardrails.context import ContextSanitizerCheck
from guardrails.pipeline import Action, PipelineResult
from guardrails.presets import rag_checks, support_tool_rules
from guardrails.tools import ToolPolicyCheck

from ..config import Settings

POLICY_VERSION = "guards-2026-10"


@dataclass
class GuardOutcome:
    allowed: bool
    text: str
    action: str
    reasons: list[str]
    blocked_by: str | None = None


def _outcome(res: PipelineResult) -> GuardOutcome:
    return GuardOutcome(allowed=res.allowed, text=res.text, action=res.action.value, reasons=res.reasons(),
                        blocked_by=res.blocked_by)


class Guards:
    def __init__(self, settings: Settings, tracer: Any = None) -> None:
        checks = [ContextSanitizerCheck(wrap=False) if isinstance(c, ContextSanitizerCheck) else c
                  for c in rag_checks(allowed_hosts=settings.allowed_hosts, canaries=settings.canaries,
                                      require_citations=False)]
        checks.append(ToolPolicyCheck(support_tool_rules(settings.allowed_mail_domains, require_send_approval=False),
                                      canaries=settings.canaries,
                                      max_calls_per_request=settings.agent_max_tool_calls + 2))
        self.pipeline = GuardrailPipeline(checks, tracer=tracer)
        self.version = POLICY_VERSION
        self.rehydrate = rehydrate_kinds("email")

    def input(self, text: str, gctx: GuardContext) -> GuardOutcome:
        return _outcome(self.pipeline.check_input(text, gctx))

    def context(self, text: str, gctx: GuardContext, source: str) -> GuardOutcome:
        return _outcome(self.pipeline.check_context(text, gctx, source=source))

    def output(self, text: str, gctx: GuardContext) -> GuardOutcome:
        return _outcome(self.pipeline.check_output(text, gctx))

    def tool(self, call: ToolCall, gctx: GuardContext) -> tuple[ToolCall, GuardOutcome]:
        """Re-hydrate PII tokens in the rule's arguments, then run the tool stage on that call."""
        prepared, res = guard_tool_call(self.pipeline, call, gctx, policy=self.rehydrate)
        return prepared, _outcome(res)


__all__ = ["Guards", "GuardOutcome", "POLICY_VERSION", "Action"]
