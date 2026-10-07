# path: book/projects/examples/ch31/semconv.py
"""The book's semantic conventions for AI spans: span names, attribute keys, error classes.

One module owns every name so that instrumentation, the trace store, the analysis queries,
the metric definitions, and the alert rules cannot drift apart. The keys are deliberately
close in spirit to the OpenTelemetry GenAI semantic conventions (`gen_ai.*`), but those
conventions were still marked experimental while this book was written and have renamed
keys more than once. Check current conventions before you standardize on either set;
`GENAI_ALIASES` is the single place to update the mapping, and `AITracer` can dual-write.
"""
from __future__ import annotations

from enum import Enum
from typing import Any


# ----------------------------------------------------------------------------- span names
class SpanName:
    REQUEST = "request"                  # root: one user-visible request or task
    ROUTER = "router.decide"             # model/route selection (Chapter 7)
    RETRIEVAL = "retrieval.search"       # first-stage retrieval (Chapters 10, 12)
    RERANK = "retrieval.rerank"          # second-stage ranking
    CONTEXT = "context.build"            # evidence packing under a token budget (Chapters 5, 13)
    GENERATE = "llm.generate"            # one logical generation: prompt identity + content
    LLM_ATTEMPT = "llm.complete"         # one provider attempt, emitted by aie_core's ModelGateway
    TOOL = "tool.call"                   # one tool execution (Chapter 16)
    AGENT_RUN = "agent.run"              # one agent loop (Chapter 19)
    AGENT_STEP = "agent.step"            # one iteration of that loop
    GUARDRAIL = "guardrail.check"        # one policy/validator decision (Chapter 27)
    EVAL = "eval.score"                  # an evaluation attached to a trace (offline or online)


STAGE_ORDER: tuple[str, ...] = (
    SpanName.ROUTER,
    SpanName.RETRIEVAL,
    SpanName.RERANK,
    SpanName.CONTEXT,
    SpanName.GENERATE,
    SpanName.GUARDRAIL,
)


# ----------------------------------------------------------------------------- attribute keys
class Attr:
    # identity and lineage, set on the root span or as resource attributes
    TRACE_ID = "trace.id"
    TENANT = "tenant.id"
    USER_HASH = "user.hash"              # salted hash, never the raw user id
    SESSION = "session.id"
    ROUTE = "app.route"                  # e.g. "rag.answer", "agent.incident"
    RESPONSE_ID = "response.id"          # id shown to the client; feedback references it
    TRAFFIC = "traffic.source"           # "user" | "probe" | "replay" | "eval"

    # versions: every artifact that can change behavior without a code deploy
    APP_VERSION = "app.version"
    PROMPT_ID = "prompt.id"
    PROMPT_VERSION = "prompt.version"
    INDEX_VERSION = "index.version"
    POLICY_VERSION = "policy.version"
    CHUNKER_VERSION = "chunker.version"
    FLAG_SET = "flags.active"

    # model calls (llm.*)
    LLM_PROVIDER = "llm.provider"
    LLM_MODEL = "llm.model"
    LLM_TEMPERATURE = "llm.temperature"
    LLM_INPUT_TOKENS = "llm.usage.input_tokens"
    LLM_OUTPUT_TOKENS = "llm.usage.output_tokens"
    LLM_CACHED_TOKENS = "llm.usage.cached_input_tokens"
    LLM_COST = "llm.cost_usd"
    LLM_AVOIDED_COST = "llm.avoided_cost_usd"   # cost a cache hit saved; never counted as spend
    LLM_CACHE_HIT = "llm.cache_hit"
    LLM_FINISH = "llm.finish_reason"
    LLM_ATTEMPT = "llm.attempt"
    LLM_LATENCY = "llm.latency_ms"
    LLM_QUEUE = "llm.queue_ms"
    LLM_TTFT = "llm.ttft_ms"
    LLM_STREAM = "llm.stream"
    LLM_PROMPT = "llm.prompt"            # content key; capture policy decides the suffix
    LLM_COMPLETION = "llm.completion"

    # retrieval and context (retrieval.*, context.*)
    RETRIEVAL_QUERY = "retrieval.query"
    RETRIEVAL_TOP_K = "retrieval.top_k"
    RETRIEVAL_IDS = "retrieval.returned_ids"
    RETRIEVAL_SCORES = "retrieval.scores"
    RETRIEVAL_TENANTS = "retrieval.returned_tenants"
    RETRIEVAL_FILTERED = "retrieval.acl_filtered"
    RERANK_IDS = "retrieval.rerank.kept_ids"
    CONTEXT_IDS = "context.evidence_ids"
    CONTEXT_DROPPED = "context.dropped_ids"
    CONTEXT_TOKENS = "context.tokens"
    CONTEXT_BUDGET = "context.budget_tokens"
    CONTEXT_TRUNCATED = "context.truncated"
    CITATION_IDS = "response.citation_ids"
    ABSTAINED = "response.abstained"

    # tools (tool.*)
    TOOL_NAME = "tool.name"
    TOOL_CALL_ID = "tool.call_id"
    TOOL_ARGS = "tool.arguments"
    TOOL_STATUS = "tool.status"          # ok | error | denied | timeout
    TOOL_SIDE_EFFECT = "tool.side_effect"
    TOOL_APPROVAL = "tool.approval"      # not_required | granted | denied | pending
    TOOL_IDEMPOTENCY = "tool.idempotency_key"
    TOOL_RESULT = "tool.result"
    TOOL_RESULT_CHARS = "tool.result.chars"

    # agents (agent.*)
    AGENT_NAME = "agent.name"
    AGENT_STEP = "agent.step"
    AGENT_ACTION = "agent.action"
    AGENT_MAX_STEPS = "agent.max_steps"
    AGENT_STOP = "agent.stop_reason"     # done | max_steps | budget | error | escalated
    AGENT_TOKENS = "agent.tokens_cumulative"

    # guardrails (guardrail.*)
    GUARD_NAME = "guardrail.name"
    GUARD_STAGE = "guardrail.stage"      # input | retrieval | output | tool
    GUARD_DECISION = "guardrail.decision"  # allow | block | redact | flag
    GUARD_REASON = "guardrail.reason"

    # evaluation and feedback (eval.*, feedback.*)
    EVAL_NAME = "eval.name"
    EVAL_SCORE = "eval.score"
    EVAL_PASSED = "eval.passed"
    EVAL_SOURCE = "eval.source"          # offline | judge | human | probe
    EVAL_GOLD_IDS = "eval.gold_ids"
    FEEDBACK_VALUE = "feedback.value"    # +1 / -1
    FEEDBACK_REASON = "feedback.reason"

    # errors
    ERROR_TYPE = "error.type"            # exception class name
    ERROR_CLASS = "error.class"          # taxonomy class below

    # content capture bookkeeping
    CAPTURE_MODE = "capture.mode"


VERSION_KEYS: tuple[str, ...] = (
    Attr.APP_VERSION,
    Attr.PROMPT_ID,
    Attr.PROMPT_VERSION,
    Attr.INDEX_VERSION,
    Attr.POLICY_VERSION,
    Attr.CHUNKER_VERSION,
    Attr.LLM_MODEL,
)


# ----------------------------------------------------------------------------- error taxonomy
class ErrorClass(str, Enum):
    """Failure classes that map to engineering work. A metric suite should cover each."""

    RETRIEVAL_MISS = "retrieval_miss"              # gold evidence not retrieved
    RETRIEVAL_CONTAMINATION = "retrieval_contamination"  # wrong tenant / stale / injected content
    PERMISSION_DENIED = "permission_denied"
    CONTEXT_TRUNCATION = "context_truncation"      # evidence retrieved but dropped by the packer
    UNSUPPORTED_CLAIM = "unsupported_claim"        # hallucination despite evidence
    CITATION_MISMATCH = "citation_mismatch"
    FORMAT_ERROR = "format_error"
    BAD_REFUSAL = "bad_refusal"
    TOOL_SELECTION = "tool_selection"
    TOOL_ARGUMENT = "tool_argument"
    TOOL_FAILURE = "tool_failure"
    LOOP = "loop"
    PREMATURE_STOP = "premature_stop"
    BUDGET_EXCEEDED = "budget_exceeded"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    PROVIDER_ERROR = "provider_error"
    CONTENT_FILTER = "content_filter"
    UNSAFE_CONTENT = "unsafe_content"
    UNKNOWN = "unknown"


_EXCEPTION_CLASSES: dict[str, ErrorClass] = {
    # aie_core.llm.errors names; matched by class name so this module has no import cycle
    "RateLimitError": ErrorClass.RATE_LIMITED,
    "TimeoutError": ErrorClass.TIMEOUT,
    "ProviderUnavailableError": ErrorClass.PROVIDER_ERROR,
    "InvalidRequestError": ErrorClass.PROVIDER_ERROR,
    "ContentFilterError": ErrorClass.CONTENT_FILTER,
    "MalformedResponseError": ErrorClass.FORMAT_ERROR,
    "ValidationError": ErrorClass.FORMAT_ERROR,
    "PermissionError": ErrorClass.PERMISSION_DENIED,
}


def classify_exception(exc: BaseException) -> ErrorClass:
    """Map an exception to the taxonomy by walking its MRO (subclasses inherit the class)."""
    for cls in type(exc).__mro__:
        if cls.__name__ in _EXCEPTION_CLASSES:
            return _EXCEPTION_CLASSES[cls.__name__]
    return ErrorClass.UNKNOWN


# ----------------------------------------------------------------------------- normalization
# aie_core's ModelGateway predates this chapter and emits unprefixed keys on `llm.complete`.
# Normalizing at export and again at load means old JSONL files stay queryable.
LEGACY_GATEWAY_KEYS: dict[str, str] = {
    "provider": Attr.LLM_PROVIDER,
    "model": Attr.LLM_MODEL,
    "input_tokens": Attr.LLM_INPUT_TOKENS,
    "output_tokens": Attr.LLM_OUTPUT_TOKENS,
    "cached_input_tokens": Attr.LLM_CACHED_TOKENS,
    "latency_ms": Attr.LLM_LATENCY,
    "queue_ms": Attr.LLM_QUEUE,
    "cache_hit": Attr.LLM_CACHE_HIT,
    "cost_usd": Attr.LLM_COST,
    "avoided_cost_usd": Attr.LLM_AVOIDED_COST,  # newer aie_core emits this natively on cache hits
    "finish_reason": Attr.LLM_FINISH,
    "attempt": Attr.LLM_ATTEMPT,
    "stream": Attr.LLM_STREAM,
}


def normalize(span_name: str, attributes: dict[str, Any]) -> dict[str, Any]:
    """Return attributes with legacy gateway keys renamed. Idempotent; unknown keys pass through."""
    if span_name not in (SpanName.LLM_ATTEMPT, "prompt.call"):
        return dict(attributes)
    out: dict[str, Any] = {}
    for key, value in attributes.items():
        out[LEGACY_GATEWAY_KEYS.get(key, key)] = value
    return account_cache_hit(out)


def account_cache_hit(attributes: dict[str, Any]) -> dict[str, Any]:
    """Older aie_core gateways served a response-cache hit with the *original* call's cost_usd
    on the span; current ones write cost 0 plus avoided_cost_usd. For legacy logs, move the
    cost to llm.avoided_cost_usd. Idempotent and a no-op when the avoided key is present."""
    if attributes.get(Attr.LLM_CACHE_HIT) and Attr.LLM_AVOIDED_COST not in attributes:
        attributes[Attr.LLM_AVOIDED_COST] = float(attributes.get(Attr.LLM_COST) or 0.0)
        attributes[Attr.LLM_COST] = 0.0
    return attributes


# ----------------------------------------------------------------------------- OTel GenAI aliases
# Conceptual equivalents in the OpenTelemetry GenAI semantic conventions. CHECK CURRENT
# CONVENTIONS: the GenAI semantic conventions are still evolving; pin the version you emit.
GENAI_ALIASES: dict[str, str] = {
    Attr.LLM_PROVIDER: "gen_ai.provider.name",
    Attr.LLM_MODEL: "gen_ai.request.model",
    Attr.LLM_TEMPERATURE: "gen_ai.request.temperature",
    Attr.LLM_INPUT_TOKENS: "gen_ai.usage.input_tokens",
    Attr.LLM_OUTPUT_TOKENS: "gen_ai.usage.output_tokens",
    Attr.LLM_FINISH: "gen_ai.response.finish_reasons",
    Attr.TOOL_NAME: "gen_ai.tool.name",
    Attr.TOOL_CALL_ID: "gen_ai.tool.call.id",
    Attr.AGENT_NAME: "gen_ai.agent.name",
    Attr.ERROR_TYPE: "error.type",
}


def genai_aliases(attributes: dict[str, Any]) -> dict[str, Any]:
    """The subset of attributes re-keyed to their GenAI-convention equivalents."""
    return {GENAI_ALIASES[k]: v for k, v in attributes.items() if k in GENAI_ALIASES and GENAI_ALIASES[k] != k}


__all__ = [
    "SpanName",
    "Attr",
    "ErrorClass",
    "STAGE_ORDER",
    "VERSION_KEYS",
    "LEGACY_GATEWAY_KEYS",
    "GENAI_ALIASES",
    "classify_exception",
    "normalize",
    "account_cache_hit",
    "genai_aliases",
]
