# path: book/projects/guardrails/guardrails/pipeline.py
"""The guardrail pipeline: ordered checks per stage, explicit verdicts, explicit failure policy.

A *check* inspects one subject (a piece of text at the input, context or output stage, or a
proposed tool call at the tool stage) and returns a `Verdict`: allow, flag, redact or block,
with a reason and a score. The pipeline runs the checks of one stage in order, threads
redacted text from one check to the next, stops at the first block, and applies each check's
`FailMode` when the check itself raises. Every run emits trace spans that carry decisions,
scores and sizes, never the raw text, so the trace sink does not become a second copy of the
sensitive data the guardrails exist to protect.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Protocol, runtime_checkable

from aie_core.llm.types import ToolCall
from aie_core.observability import NoopTracer, Tracer


class Stage(str, Enum):
    INPUT = "input"        # user text before it reaches the model
    CONTEXT = "context"    # untrusted text (retrieved docs, tool results) before it enters the prompt
    OUTPUT = "output"      # model text before it reaches a user, a renderer or a downstream system
    TOOL = "tool"          # a proposed tool call before it executes


class Action(str, Enum):
    ALLOW = "allow"
    FLAG = "flag"          # let it through, record a signal (alerting, sampling for review)
    REDACT = "redact"      # let a modified payload through (masked PII, stripped URLs, wrapped data)
    BLOCK = "block"        # stop; the caller must take the fallback path

    @property
    def severity(self) -> int:
        return _SEVERITY[self]


_SEVERITY = {Action.ALLOW: 0, Action.FLAG: 1, Action.REDACT: 2, Action.BLOCK: 3}


class FailMode(str, Enum):
    CLOSED = "closed"      # if the check cannot decide, block
    OPEN = "open"          # if the check cannot decide, allow and flag the error


@dataclass(frozen=True)
class Finding:
    """One concrete thing a check found. `value` is never exported to traces."""

    kind: str
    start: int = -1
    end: int = -1
    value: str = ""
    detail: str = ""


@dataclass(frozen=True)
class Verdict:
    action: Action
    check: str = ""
    reason: str = ""
    score: float = 0.0
    redacted: str | None = None              # replacement payload when action is REDACT
    findings: tuple[Finding, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    error: bool = False                      # the check raised; action came from its FailMode

    @classmethod
    def allow(cls, reason: str = "", score: float = 0.0, **metadata: Any) -> "Verdict":
        return cls(Action.ALLOW, reason=reason, score=score, metadata=metadata)

    @classmethod
    def flag(cls, reason: str, score: float = 0.0, findings: Iterable[Finding] = (), **metadata: Any) -> "Verdict":
        return cls(Action.FLAG, reason=reason, score=score, findings=tuple(findings), metadata=metadata)

    @classmethod
    def redact(cls, text: str, reason: str, score: float = 0.0, findings: Iterable[Finding] = (),
               **metadata: Any) -> "Verdict":
        return cls(Action.REDACT, reason=reason, score=score, redacted=text, findings=tuple(findings),
                   metadata=metadata)

    @classmethod
    def block(cls, reason: str, score: float = 1.0, findings: Iterable[Finding] = (), **metadata: Any) -> "Verdict":
        return cls(Action.BLOCK, reason=reason, score=score, findings=tuple(findings), metadata=metadata)


@dataclass
class GuardContext:
    """Who the request is for. Built once from the authenticated request (Chapter 28), never
    from anything the model or a document said."""

    tenant: str
    user_id: str
    groups: frozenset[str] = frozenset({"all"})
    request_id: str = ""
    vault: Any = None                                   # a pii.PIIVault for reversible tokenization
    approvals: set[str] = field(default_factory=set)    # approval tokens bound to exact tool args
    evidence_ids: frozenset[str] = frozenset()          # ids actually shown to the model (citations)
    state: dict[str, Any] = field(default_factory=dict) # per-request counters (tool call budget...)


@dataclass(frozen=True)
class Subject:
    stage: Stage
    text: str = ""
    tool_call: ToolCall | None = None
    source: str = "user"           # provenance: "user", "retrieved:<doc-id>", "tool:<name>", "model"


@runtime_checkable
class Check(Protocol):
    name: str
    stages: frozenset[Stage]
    fail_mode: FailMode

    def evaluate(self, subject: Subject, ctx: GuardContext) -> Verdict: ...


class BaseCheck:
    """Convenience base: subclasses set `name`, `stages`, `fail_mode` and implement `evaluate`."""

    name: str = "check"
    stages: frozenset[Stage] = frozenset(Stage)
    fail_mode: FailMode = FailMode.CLOSED

    def evaluate(self, subject: Subject, ctx: GuardContext) -> Verdict:  # pragma: no cover
        raise NotImplementedError


@dataclass
class PipelineResult:
    stage: Stage
    action: Action
    text: str                           # final payload after redactions ("" for tool stage)
    verdicts: list[Verdict]
    blocked_by: str | None = None
    tool_call: ToolCall | None = None

    @property
    def allowed(self) -> bool:
        return self.action is not Action.BLOCK

    @property
    def flags(self) -> list[Verdict]:
        return [v for v in self.verdicts if v.action is Action.FLAG]

    @property
    def errors(self) -> list[Verdict]:
        return [v for v in self.verdicts if v.error]

    def reasons(self) -> list[str]:
        return [f"{v.check}: {v.reason}" for v in self.verdicts if v.action is not Action.ALLOW]


def text_fingerprint(text: str) -> str:
    """Short hash so traces can correlate payloads without storing them."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


class GuardrailPipeline:
    """Runs checks stage by stage.

    Ordering rule: cheap deterministic checks first, model-based checks last, so a request that
    is going to be blocked by a size limit never pays for a classifier call.
    """

    def __init__(self, checks: Iterable[Check] = (), tracer: Tracer | None = None) -> None:
        self._checks: list[Check] = list(checks)
        self.tracer = tracer or NoopTracer()
        for c in self._checks:
            if not isinstance(c, Check):
                raise TypeError(f"{c!r} does not implement the Check protocol")

    def add(self, check: Check) -> "GuardrailPipeline":
        self._checks.append(check)
        return self

    def checks_for(self, stage: Stage) -> list[Check]:
        return [c for c in self._checks if stage in c.stages]

    # ------------------------------------------------------------------ entry points
    def check_input(self, text: str, ctx: GuardContext) -> PipelineResult:
        return self.run(Subject(Stage.INPUT, text=text, source="user"), ctx)

    def check_context(self, text: str, ctx: GuardContext, source: str = "retrieved") -> PipelineResult:
        return self.run(Subject(Stage.CONTEXT, text=text, source=source), ctx)

    def check_output(self, text: str, ctx: GuardContext) -> PipelineResult:
        return self.run(Subject(Stage.OUTPUT, text=text, source="model"), ctx)

    def check_tool(self, call: ToolCall, ctx: GuardContext) -> PipelineResult:
        return self.run(Subject(Stage.TOOL, tool_call=call, source="model"), ctx)

    # ------------------------------------------------------------------ core loop
    def run(self, subject: Subject, ctx: GuardContext) -> PipelineResult:
        stage = subject.stage
        current = subject
        verdicts: list[Verdict] = []
        worst = Action.ALLOW
        blocked_by: str | None = None
        with self.tracer.span(
            f"guardrail.{stage.value}",
            **{
                "guardrail.stage": stage.value,
                "guardrail.source": subject.source,
                "guardrail.input_chars": len(subject.text),
                "guardrail.input_hash": text_fingerprint(subject.text) if subject.text else "",
                "guardrail.tool": subject.tool_call.name if subject.tool_call else "",
                "request.id": ctx.request_id,
                "tenant.id": ctx.tenant,
            },
        ) as span:
            for check in self.checks_for(stage):
                verdict = self._evaluate_one(check, current, ctx)
                verdicts.append(verdict)
                if verdict.action.severity > worst.severity:
                    worst = verdict.action
                if verdict.action is Action.REDACT and verdict.redacted is not None:
                    current = Subject(stage, text=verdict.redacted, tool_call=current.tool_call,
                                      source=current.source)
                if verdict.action is Action.BLOCK:
                    blocked_by = check.name
                    break
            span.set_attribute("guardrail.action", worst.value)
            span.set_attribute("guardrail.blocked_by", blocked_by or "")
            span.set_attribute("guardrail.checks_run", len(verdicts))
            span.set_attribute("guardrail.flags", [v.check for v in verdicts if v.action is Action.FLAG])
            span.set_attribute("guardrail.errors", [v.check for v in verdicts if v.error])
            span.set_attribute("guardrail.output_chars", len(current.text))
        return PipelineResult(stage=stage, action=worst, text=current.text, verdicts=verdicts,
                              blocked_by=blocked_by, tool_call=subject.tool_call)

    def _evaluate_one(self, check: Check, subject: Subject, ctx: GuardContext) -> Verdict:
        t0 = time.perf_counter()
        with self.tracer.span("guardrail.check", **{"guardrail.check": check.name,
                                                     "guardrail.stage": subject.stage.value,
                                                     "guardrail.fail_mode": check.fail_mode.value}) as span:
            try:
                verdict = check.evaluate(subject, ctx)
                if not isinstance(verdict, Verdict):
                    raise TypeError(f"check {check.name} returned {type(verdict).__name__}, not Verdict")
            except Exception as exc:  # the check failed, not the request: apply its policy
                span.set_attribute("guardrail.exception", type(exc).__name__)
                if check.fail_mode is FailMode.CLOSED:
                    verdict = Verdict(Action.BLOCK, reason=f"check failed closed: {type(exc).__name__}",
                                      score=1.0, error=True)
                else:
                    verdict = Verdict(Action.FLAG, reason=f"check failed open: {type(exc).__name__}",
                                      score=0.0, error=True)
            if not verdict.check:
                verdict = Verdict(verdict.action, check.name, verdict.reason, verdict.score, verdict.redacted,
                                  verdict.findings, verdict.metadata, verdict.error)
            span.set_attribute("guardrail.action", verdict.action.value)
            span.set_attribute("guardrail.score", round(float(verdict.score), 4))
            span.set_attribute("guardrail.reason", verdict.reason[:200])
            span.set_attribute("guardrail.findings", [f.kind for f in verdict.findings])
            span.set_attribute("guardrail.error", verdict.error)
            span.set_attribute("guardrail.latency_ms", round((time.perf_counter() - t0) * 1000, 3))
        return verdict


__all__ = [
    "Stage", "Action", "FailMode", "Finding", "Verdict", "GuardContext", "Subject", "Check", "BaseCheck",
    "PipelineResult", "GuardrailPipeline", "text_fingerprint",
]
