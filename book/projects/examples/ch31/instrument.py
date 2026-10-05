# path: book/projects/examples/ch31/instrument.py
"""Trace context, content capture policy, and instrumentation helpers for AI stages.

Builds on `aie_core.observability`: `AITracer` *is* an `aie_core` `Tracer`, so it can be
handed to `ModelGateway(tracer=...)` and every gateway attempt becomes a child span of
whatever stage is open. What it adds to the base tracer:

* trace context: every span gets `trace_id` and `parent_span_id` through a contextvar, so
  spans from different layers assemble into one tree per request;
* resource attributes (service, environment, deployment) stamped on every span;
* a content capture policy (off | hashed | redacted | full, with per-trace sampling,
  capture-on-error, and per-tenant ceilings) applied when the span ends;
* error classification into the taxonomy in `semconv`;
* stage helpers: request, retrieval, rerank, context, generation, tool, agent, guardrail.

The export target is any `aie_core` tracer (`JsonlTracer`, `InMemoryTracer`, `NoopTracer`).
`otel_setup.OTelAITracer` subclasses this class to drive a real OpenTelemetry SDK.
"""
from __future__ import annotations

import functools
import hashlib
import hmac
import json
import random
import re
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Literal, Sequence, TypeVar

from aie_core.llm.client import LLMClient
from aie_core.llm.types import Completion, CompletionRequest, Message
from aie_core.observability import NoopTracer, Span, Tracer

from semconv import VERSION_KEYS, Attr, ErrorClass, SpanName, classify_exception, genai_aliases, normalize

CaptureMode = Literal["off", "hashed", "redacted", "full"]
# Keys inherited by descendants and stamped on every provider attempt (llm.complete).
# "prompt.hash" is set by Chapter 4's traced_complete on its prompt.call span.
PROPAGATED_KEYS: tuple[str, ...] = (*VERSION_KEYS, "prompt.hash", Attr.TENANT, Attr.ROUTE)
_MODE_RANK: dict[str, int] = {"off": 0, "hashed": 1, "redacted": 2, "full": 3}
F = TypeVar("F", bound=Callable[..., Any])


# ============================================================================ spans with lineage
@dataclass
class LinkedSpan(Span):
    """An aie_core Span plus the fields that make a tree: trace id, parent id, resource."""

    trace_id: str = ""
    parent_span_id: str | None = None
    resource: dict[str, Any] = field(default_factory=dict)
    baggage: dict[str, Any] = field(default_factory=dict)   # propagated to children, not exported
    pending_content: dict[str, str] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        out = super().to_dict()
        out.update({"trace_id": self.trace_id, "parent_span_id": self.parent_span_id, "resource": self.resource})
        return out


_current: ContextVar[LinkedSpan | None] = ContextVar("ch31_current_span", default=None)


def current_span() -> LinkedSpan | None:
    return _current.get()


def mark_error(span: Span, error_class: ErrorClass | str, message: str = "") -> None:
    """Record a *semantic* failure (no exception raised): wrong evidence, loop, bad citation."""
    span.status = "error"
    span.set_attribute(Attr.ERROR_CLASS, ErrorClass(error_class).value)
    span.events.append({"type": "semantic_error", "error.class": ErrorClass(error_class).value,
                        "message": message, "time": time.time()})


# ============================================================================ capture policy
_REDACTIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[EMAIL]"),
    (re.compile(r"\b(?:sk|pk|api|key|tok)[-_][A-Za-z0-9]{12,}\b", re.IGNORECASE), "[SECRET]"),
    (re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]{12,}"), r"\1 [SECRET]"),
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"), "[CARD]"),
    (re.compile(r"\+?\d{1,3}[ .-]?\(?\d{2,4}\)?[ .-]?\d{3}[ .-]?\d{2,4}\b"), "[PHONE]"),
    (re.compile(r"\bNW-\d{6}\b"), "[EMPLOYEE_ID]"),
]


def redact(text: str) -> str:
    """Pattern redaction. A floor, not a guarantee: names and free-text PII survive regexes,
    so production adds a dedicated PII detector (Chapter 27) behind the same function."""
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


def _bucket(trace_id: str) -> float:
    """Deterministic value in [0, 1) per trace, so every service makes the same sampling choice."""
    return int(hashlib.sha256(trace_id.encode()).hexdigest()[:8], 16) / 2**32


@dataclass
class CapturePolicy:
    """Decides, per span and at span end, how much prompt/response/tool content to keep.

    mode            default for every trace
    sample_rate     fraction of traces upgraded to `sampled_mode` (head sampling by trace id)
    on_error_mode   upgrade for spans that ended in error (they are the ones you will debug)
    tenant_ceiling  hard per-tenant maximum, applied last: contracts beat debugging convenience
    """

    mode: CaptureMode = "hashed"
    sample_rate: float = 0.0
    sampled_mode: CaptureMode = "redacted"
    on_error_mode: CaptureMode | None = "redacted"
    max_chars: int = 4000
    salt: str = "change-me"
    tenant_ceiling: dict[str, CaptureMode] = field(default_factory=dict)
    redactor: Callable[[str], str] = redact

    def decide(self, trace_id: str, tenant: str | None, errored: bool) -> CaptureMode:
        mode: CaptureMode = self.mode
        if self.sample_rate > 0 and _bucket(trace_id) < self.sample_rate:
            mode = max(mode, self.sampled_mode, key=_MODE_RANK.__getitem__)
        if errored and self.on_error_mode:
            mode = max(mode, self.on_error_mode, key=_MODE_RANK.__getitem__)
        ceiling = self.tenant_ceiling.get(tenant or "")
        if ceiling is not None:
            mode = min(mode, ceiling, key=_MODE_RANK.__getitem__)
        return mode

    def digest(self, text: str) -> str:
        # keyed hash: a plain sha256 of an email address is reversible by dictionary attack
        return hmac.new(self.salt.encode(), text.encode(), hashlib.sha256).hexdigest()[:16]

    def render(self, key: str, text: str, mode: CaptureMode) -> dict[str, Any]:
        out: dict[str, Any] = {f"{key}.chars": len(text)}
        if mode == "off":
            return out
        out[f"{key}.hash"] = self.digest(text)
        if mode == "redacted":
            out[f"{key}.content"] = self.redactor(text)[: self.max_chars]
        elif mode == "full":
            out[f"{key}.content"] = text[: self.max_chars]
        return out


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


# ============================================================================ the tracer
class AITracer(Tracer):
    """aie_core Tracer with trace context, resource attributes, and content capture."""

    def __init__(
        self,
        sink: Tracer | None = None,
        *,
        resource: dict[str, Any] | None = None,
        capture: CapturePolicy | None = None,
        clock: Callable[[], float] = time.time,
        rng: random.Random | None = None,
        emit_genai_aliases: bool = False,
    ) -> None:
        self.sink = sink or NoopTracer()
        self.resource = dict(resource or {})
        self.capture_policy = capture or CapturePolicy()
        self.clock = clock
        self._rng = rng
        self.emit_genai_aliases = emit_genai_aliases

    # ---------------------------------------------------------------- ids
    def _hex(self, bits: int) -> str:
        if self._rng is not None:
            return format(self._rng.getrandbits(bits), f"0{bits // 4}x")
        return uuid.uuid4().hex[: bits // 4] if bits <= 128 else uuid.uuid4().hex

    # ---------------------------------------------------------------- backend hooks (OTel overrides)
    def _backend_start(self, span: LinkedSpan, parent: LinkedSpan | None) -> Any:
        return None

    def _backend_end(self, span: LinkedSpan, handle: Any) -> None:
        return None

    # ---------------------------------------------------------------- core
    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[LinkedSpan]:
        parent = _current.get()
        s = LinkedSpan(
            name=name,
            attributes=normalize(name, attributes),
            start=self.clock(),
            span_id=self._hex(64),
            trace_id=parent.trace_id if parent else self._hex(128),
            parent_span_id=parent.span_id if parent else None,
            resource=dict(self.resource),
            baggage=dict(parent.baggage) if parent else {},
        )
        self._propagate(s)
        handle = self._backend_start(s, parent)
        token = _current.set(s)
        try:
            yield s
        except BaseException as exc:
            s.record_exception(exc)
            s.attributes.setdefault(Attr.ERROR_TYPE, type(exc).__name__)
            s.attributes.setdefault(Attr.ERROR_CLASS, classify_exception(exc).value)
            raise
        finally:
            _current.reset(token)
            s.end = self.clock()
            self._finalize(s)
            self._backend_end(s, handle)
            self.sink.export(s)

    @staticmethod
    def _propagate(s: LinkedSpan) -> None:
        """Lineage keys flow down the tree. The gateway's llm.complete span cannot see request
        metadata, so it receives prompt identity, tenant, and the version manifest from its
        ancestors' baggage. Shared by `span()` and `export()`, so a streamed attempt adopted at
        export time is stamped exactly like a non-streamed one."""
        for key in PROPAGATED_KEYS:
            if key in s.attributes:
                s.baggage[key] = s.attributes[key]
        if s.name == SpanName.LLM_ATTEMPT:
            for key in PROPAGATED_KEYS:
                if key in s.baggage and key not in s.attributes:
                    s.attributes[key] = s.baggage[key]

    def export(self, span: Span) -> None:
        """Adopt spans built by hand elsewhere (the gateway's streaming path) into the tree."""
        if not isinstance(span, LinkedSpan):
            parent = _current.get()
            span = LinkedSpan(
                name=span.name, attributes=normalize(span.name, span.attributes), start=span.start,
                end=span.end, span_id=span.span_id, status=span.status, events=list(span.events),
                trace_id=parent.trace_id if parent else self._hex(128),
                parent_span_id=parent.span_id if parent else None,
                resource=dict(self.resource), baggage=dict(parent.baggage) if parent else {},
            )
            if span.status == "error":
                span.attributes.setdefault(Attr.ERROR_CLASS, ErrorClass.PROVIDER_ERROR.value)
            self._propagate(span)
            self._finalize(span)
            self._backend_end(span, self._backend_start(span, None))
        self.sink.export(span)

    def _finalize(self, s: LinkedSpan) -> None:
        # the gateway sets its legacy keys after the span opens, so normalize again at the end
        s.attributes = normalize(s.name, s.attributes)
        if s.pending_content:
            mode = self.capture_policy.decide(s.trace_id, s.baggage.get(Attr.TENANT), s.status == "error")
            for key, text in s.pending_content.items():
                s.attributes.update(self.capture_policy.render(key, text, mode))
            s.attributes[Attr.CAPTURE_MODE] = mode
            s.pending_content.clear()
        if self.emit_genai_aliases:
            s.attributes.update(genai_aliases(s.attributes))

    def capture(self, span: LinkedSpan, key: str, value: Any) -> None:
        """Offer content to the policy. Nothing is written until the span ends."""
        span.pending_content[key] = _as_text(value)


# ============================================================================ stage helpers
@dataclass
class StageHandle:
    tracer: AITracer
    span: LinkedSpan

    def set(self, **attributes: Any) -> None:
        for k, v in attributes.items():
            self.span.set_attribute(k, v)


def _user_hash(policy: CapturePolicy, user_id: str | None) -> str | None:
    return policy.digest(f"user:{user_id}") if user_id else None


@contextmanager
def trace_request(
    tracer: AITracer,
    *,
    route: str,
    tenant: str,
    user_id: str | None = None,
    traffic: str = "user",
    response_id: str | None = None,
    versions: dict[str, Any] | None = None,
    **attributes: Any,
) -> Iterator[LinkedSpan]:
    """The root span. Everything that can explain a bad answer later is stamped here."""
    attrs: dict[str, Any] = {Attr.ROUTE: route, Attr.TENANT: tenant, Attr.TRAFFIC: traffic, **(versions or {}), **attributes}
    if user_id:
        attrs[Attr.USER_HASH] = _user_hash(tracer.capture_policy, user_id)
    if response_id:
        attrs[Attr.RESPONSE_ID] = response_id
    with tracer.span(SpanName.REQUEST, **attrs) as root:
        root.baggage[Attr.TENANT] = tenant
        yield root


class RetrievalHandle(StageHandle):
    def results(
        self,
        ids: Sequence[str],
        scores: Sequence[float],
        tenants: Sequence[str] | None = None,
        acl_filtered: int = 0,
    ) -> None:
        self.set(**{Attr.RETRIEVAL_IDS: list(ids), Attr.RETRIEVAL_SCORES: [round(float(x), 4) for x in scores],
                    Attr.RETRIEVAL_FILTERED: acl_filtered})
        if tenants is not None:
            self.set(**{Attr.RETRIEVAL_TENANTS: list(tenants)})
            request_tenant = self.span.baggage.get(Attr.TENANT)
            foreign = sorted({t for t in tenants if t not in (request_tenant, "shared")})
            if request_tenant and foreign:
                # zero cross-tenant leakage is a hard requirement: make it an error class
                mark_error(self.span, ErrorClass.RETRIEVAL_CONTAMINATION, f"foreign tenants: {foreign}")


@contextmanager
def retrieval_span(tracer: AITracer, query: str, *, index_version: str, top_k: int) -> Iterator[RetrievalHandle]:
    with tracer.span(SpanName.RETRIEVAL, **{Attr.INDEX_VERSION: index_version, Attr.RETRIEVAL_TOP_K: top_k}) as s:
        tracer.capture(s, Attr.RETRIEVAL_QUERY, query)
        yield RetrievalHandle(tracer, s)


class RerankHandle(StageHandle):
    def kept(self, ids: Sequence[str]) -> None:
        self.set(**{Attr.RERANK_IDS: list(ids)})


@contextmanager
def rerank_span(tracer: AITracer, *, model: str, candidates: int) -> Iterator[RerankHandle]:
    with tracer.span(SpanName.RERANK, **{"rerank.model": model, "rerank.candidates": candidates}) as s:
        yield RerankHandle(tracer, s)


class ContextHandle(StageHandle):
    def packed(self, included: Sequence[str], dropped: Sequence[str], tokens: int) -> None:
        self.set(**{Attr.CONTEXT_IDS: list(included), Attr.CONTEXT_DROPPED: list(dropped),
                    Attr.CONTEXT_TOKENS: tokens, Attr.CONTEXT_TRUNCATED: bool(dropped)})


@contextmanager
def context_span(tracer: AITracer, *, budget_tokens: int) -> Iterator[ContextHandle]:
    with tracer.span(SpanName.CONTEXT, **{Attr.CONTEXT_BUDGET: budget_tokens}) as s:
        yield ContextHandle(tracer, s)


def _render_messages(messages: Sequence[Message]) -> str:
    return "\n".join(f"[{m.role.value}] {m.text}" for m in messages)


def traced_generation(
    tracer: AITracer,
    client: LLMClient,
    req: CompletionRequest,
    *,
    prompt_id: str,
    prompt_version: str,
) -> Completion:
    """One logical generation. Usually `client` is a ModelGateway built with the same tracer,
    so each provider attempt (`llm.complete`) nests under this `llm.generate` span."""
    attrs = {Attr.PROMPT_ID: prompt_id, Attr.PROMPT_VERSION: prompt_version, Attr.LLM_TEMPERATURE: req.temperature}
    if req.model:
        attrs[Attr.LLM_MODEL] = req.model
    with tracer.span(SpanName.GENERATE, **attrs) as s:
        tracer.capture(s, Attr.LLM_PROMPT, _render_messages(req.messages))
        completion = client.complete(req)
        s.set_attribute(Attr.LLM_PROVIDER, completion.provider)
        s.set_attribute(Attr.LLM_MODEL, completion.model)
        s.set_attribute(Attr.LLM_INPUT_TOKENS, completion.usage.input_tokens)
        s.set_attribute(Attr.LLM_OUTPUT_TOKENS, completion.usage.output_tokens)
        s.set_attribute(Attr.LLM_CACHED_TOKENS, completion.usage.cached_input_tokens)
        s.set_attribute(Attr.LLM_FINISH, completion.finish_reason)
        s.set_attribute(Attr.LLM_COST, float((completion.raw or {}).get("cost_usd", 0.0)))
        tracer.capture(s, Attr.LLM_COMPLETION, completion.text or [tc.model_dump() for tc in completion.tool_calls])
        if completion.finish_reason == "length":
            mark_error(s, ErrorClass.FORMAT_ERROR, "output truncated at max_tokens")
        return completion


class ToolHandle(StageHandle):
    def result(self, value: Any, status: str = "ok") -> None:
        text = _as_text(value)
        self.set(**{Attr.TOOL_STATUS: status, Attr.TOOL_RESULT_CHARS: len(text)})
        self.tracer.capture(self.span, Attr.TOOL_RESULT, text)
        if status == "denied":
            mark_error(self.span, ErrorClass.PERMISSION_DENIED, "tool call denied by policy")


@contextmanager
def tool_span(
    tracer: AITracer,
    name: str,
    arguments: dict[str, Any],
    *,
    side_effect: bool = False,
    approval: str = "not_required",
    call_id: str | None = None,
    idempotency_key: str | None = None,
) -> Iterator[ToolHandle]:
    attrs: dict[str, Any] = {Attr.TOOL_NAME: name, Attr.TOOL_SIDE_EFFECT: side_effect, Attr.TOOL_APPROVAL: approval,
                             # keyed, so low-entropy arguments cannot be recovered by guessing
                             "tool.args_fingerprint": tracer.capture_policy.digest(_as_text(arguments))[:12]}
    if call_id:
        attrs[Attr.TOOL_CALL_ID] = call_id
    if idempotency_key:
        attrs[Attr.TOOL_IDEMPOTENCY] = idempotency_key
    with tracer.span(SpanName.TOOL, **attrs) as s:
        tracer.capture(s, Attr.TOOL_ARGS, arguments)
        handle = ToolHandle(tracer, s)
        try:
            yield handle
        except PermissionError:
            s.set_attribute(Attr.TOOL_STATUS, "denied")
            raise
        except BaseException:
            s.set_attribute(Attr.TOOL_STATUS, "error")
            s.set_attribute(Attr.ERROR_CLASS, ErrorClass.TOOL_FAILURE.value)
            raise


def traced_tool(tracer: AITracer, name: str | None = None, *, side_effect: bool = False) -> Callable[[F], F]:
    """Decorator for keyword-argument tool functions: `@traced_tool(tracer, "search_tickets")`."""

    def wrap(fn: F) -> F:
        tool_name = name or fn.__name__

        @functools.wraps(fn)
        def inner(**kwargs: Any) -> Any:
            with tool_span(tracer, tool_name, kwargs, side_effect=side_effect) as t:
                value = fn(**kwargs)
                t.result(value)
                return value

        return inner  # type: ignore[return-value]

    return wrap


def traced(tracer: AITracer, span_name: str, **static_attributes: Any) -> Callable[[F], F]:
    """Generic decorator: run the function inside a span with fixed attributes."""

    def wrap(fn: F) -> F:
        @functools.wraps(fn)
        def inner(*args: Any, **kwargs: Any) -> Any:
            with tracer.span(span_name, **static_attributes):
                return fn(*args, **kwargs)

        return inner  # type: ignore[return-value]

    return wrap


class AgentRunHandle(StageHandle):
    def stop(self, reason: str, *, error_class: ErrorClass | None = None) -> None:
        self.set(**{Attr.AGENT_STOP: reason})
        if error_class is not None:
            mark_error(self.span, error_class, f"agent stopped: {reason}")


@contextmanager
def agent_run(tracer: AITracer, agent_name: str, *, max_steps: int) -> Iterator[AgentRunHandle]:
    with tracer.span(SpanName.AGENT_RUN, **{Attr.AGENT_NAME: agent_name, Attr.AGENT_MAX_STEPS: max_steps}) as s:
        yield AgentRunHandle(tracer, s)


@contextmanager
def agent_step(tracer: AITracer, step: int, action: str, *, tokens_cumulative: int | None = None) -> Iterator[StageHandle]:
    attrs: dict[str, Any] = {Attr.AGENT_STEP: step, Attr.AGENT_ACTION: action}
    if tokens_cumulative is not None:
        attrs[Attr.AGENT_TOKENS] = tokens_cumulative
    with tracer.span(SpanName.AGENT_STEP, **attrs) as s:
        yield StageHandle(tracer, s)


class GuardrailHandle(StageHandle):
    def decide(self, decision: str, reason: str = "", *, error_class: ErrorClass | None = None) -> None:
        self.set(**{Attr.GUARD_DECISION: decision, Attr.GUARD_REASON: reason})
        if error_class is not None:
            mark_error(self.span, error_class, reason)


@contextmanager
def guardrail_span(tracer: AITracer, name: str, *, stage: str, policy_version: str | None = None) -> Iterator[GuardrailHandle]:
    attrs: dict[str, Any] = {Attr.GUARD_NAME: name, Attr.GUARD_STAGE: stage}
    if policy_version:
        attrs[Attr.POLICY_VERSION] = policy_version
    with tracer.span(SpanName.GUARDRAIL, **attrs) as s:
        yield GuardrailHandle(tracer, s)


__all__ = [
    "LinkedSpan", "AITracer", "PROPAGATED_KEYS", "CapturePolicy", "CaptureMode", "redact", "mark_error", "current_span",
    "trace_request", "retrieval_span", "rerank_span", "context_span", "traced_generation",
    "tool_span", "traced_tool", "traced", "agent_run", "agent_step", "guardrail_span",
]
