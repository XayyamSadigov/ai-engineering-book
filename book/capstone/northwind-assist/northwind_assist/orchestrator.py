# path: book/capstone/northwind-assist/northwind_assist/orchestrator.py
"""The request path of one chat turn.

    prepare:  admission (Ch 29) -> spend reservation (Ch 30) -> degraded plan + flags (Ch 29, 32)
    run:      input guard (Ch 27) -> memory observation (Ch 21) -> intent rules -> model route (Ch 7)
              -> one of: grounded answer (Ch 12-15) | agent with tools (Ch 16, 19) |
                 extraction (Ch 6) | memory command | small talk
              -> done event with lineage, cost and usage -> ledger, memory, trace

`prepare` runs before the HTTP response starts, so rejections are real 429/503 responses with
Retry-After instead of an error event inside a 200 stream. `run` pushes typed events through
`emit` as they happen; the API turns them into SSE frames, tests collect them in a list. One
AITracer root span covers the turn; every stage span (router, retrieval, rerank, guardrails,
model attempts, tools, agent steps) nests under it.
"""
from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from aie_core.llm.errors import LLMError
from aie_core.llm.types import CompletionRequest, Message
from context import ContextItem, RequestScope, Trust  # type: ignore[import-not-found]
from instrument import trace_request  # type: ignore[import-not-found]
from pydantic import BaseModel, Field
from reliability import DEFAULT_PLANS, CircuitOpenError, Deadline, DeadlineExceeded, DegradeLevel

from .domain.context import RequestContext
from .domain.events import ServerEvent
from .domain.intents import Intent, IntentDecision, route_intent
from .llm.models import UsageMeter
from .rag.service import RAG_PROMPT, RagResult
from .resilience.wiring import Plan

Emit = Callable[[ServerEvent], None]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    # 8,000 characters is the input guard's size limit; a longer document fails here as 422 instead of
    # passing validation and then being refused by policy.
    document: str | None = Field(default=None, max_length=8_000, description="text to extract fields from")
    doc_type: str | None = Field(default=None, pattern=r"^(invoice|support_ticket)$")


class Rejected(Exception):
    def __init__(self, status: int, reason: str, retry_after_s: float | None = None) -> None:
        super().__init__(reason)
        self.status, self.reason, self.retry_after_s = status, reason, retry_after_s


@dataclass
class Prepared:
    ctx: RequestContext
    req: ChatRequest
    plan: Plan
    spend: Any
    started: float
    intent: IntentDecision


@dataclass
class TurnResult:
    request_id: str
    intent: str
    status: str = "error"
    answer: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    rag: RagResult | None = None
    agent: Any = None
    approvals: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    memory: list[dict[str, Any]] = field(default_factory=list)
    structured: dict[str, Any] | None = None
    events: list[ServerEvent] = field(default_factory=list)
    cost_usd: float = 0.0
    lineage: dict[str, Any] = field(default_factory=dict)
    trace_id: str | None = None


class _Emitter:
    def __init__(self, emit: Emit, result: TurnResult) -> None:
        self._emit, self._result, self._n = emit, result, 0

    def __call__(self, event: str, data: dict[str, Any]) -> None:
        self._n += 1
        ev = ServerEvent(event=event, data=data, id=self._n)  # type: ignore[arg-type]
        self._result.events.append(ev)
        self._emit(ev)


class Orchestrator:
    def __init__(self, c: Any) -> None:   # c: Container (composition root)
        self.c = c

    # ================================================================== prepare
    def prepare(self, ctx: RequestContext, req: ChatRequest, *, idempotency_key: str | None = None,
                force_intent: Intent | None = None) -> Prepared:
        s = self.c.settings
        session = req.session_id or f"ses_{uuid.uuid4().hex[:10]}"
        ctx = ctx.with_request(session_id=session, idempotency_key=idempotency_key,
                               deadline=Deadline.after(s.request_deadline_s))
        est_tokens = 1500 + len(req.message) // 3
        admission = self.c.resilience.admit(ctx.tenant, est_tokens, s.request_deadline_s)
        if not admission.admitted:
            raise Rejected(admission.http_status(), admission.reason, admission.retry_after_s)
        estimate = self.c.models.pricing.cost_usd(self.c.models.catalog.get("nw-general").model_id,
                                                  _usage(est_tokens, 600)) * 4   # worst case: a few calls
        spend = self.c.ledger.reserve(ctx.tenant, estimate)
        if spend.action == "block":
            self.c.resilience.release(admission, 0.0)
            raise Rejected(429, "tenant_budget_exhausted", 3600.0)
        plan = self.c.resilience.plan(admission, open_dependencies=self.c.models.open_dependencies(),
                                      spend_action=spend.action, user_id=ctx.user_id)
        intent = route_intent(req.message, has_document=bool(req.document))
        if force_intent is not None:   # evaluation of one workflow, never set from a request
            intent = IntentDecision(force_intent, "forced", intent.task)
        return Prepared(ctx, req, plan, spend, time.monotonic(), intent)

    # ================================================================== run
    def run(self, p: Prepared, emit: Emit = lambda e: None) -> TurnResult:
        ctx, plan = p.ctx, p.plan
        result = TurnResult(request_id=ctx.request_id, intent=p.intent.intent.value)
        out = _Emitter(emit, result)
        meter = UsageMeter()
        manifest = self.c.manifest_for(plan.flags)
        versions = manifest.as_span_attributes()
        success = False
        try:
            prompt_id, prompt_version = self._prompt_identity(p.intent.intent)
            # Chapter 31's lineage keys (tenant, route, prompt.version, index.version) on the root, so its
            # completeness metric and the version-diff queries work on capstone traces unchanged.
            with trace_request(self.c.tracer, route=p.intent.intent.value, tenant=ctx.tenant, user_id=ctx.user_id,
                               response_id=ctx.request_id, versions=versions,
                               **{"session.id": ctx.session_id, "degrade.level": plan.level,
                                  "intent.rule": p.intent.rule, "index.version": self.c.kb.index_version,
                                  "prompt.id": prompt_id, "prompt.version": prompt_version}) as root:
                result.trace_id = root.trace_id
                success = self._run(p, out, meter, manifest, result, root)
                root.set_attribute("cost.usd", meter.cost_usd)
                root.set_attribute("tokens.input", meter.input_tokens)
                root.set_attribute("tokens.output", meter.output_tokens)
                root.set_attribute("answer.status", result.status)
                root.set_attribute("evidence.ids", [c.get("doc_id") for c in result.citations])
                root.set_attribute("cache.hit", bool(result.rag and result.rag.answer_cache_hit))
                if result.rag is not None:
                    root.set_attribute("response.abstained", result.status == "insufficient_evidence")
                    violations = (result.rag.retrieval.trace.get("acl_violations") or []) if result.rag.retrieval else []
                    if violations:   # the final ACL check dropped hits: a security event (Chapter 31 alert)
                        root.set_attribute("acl.violations", len(violations))
                        root.set_attribute("error.class", "retrieval_contamination")
        finally:
            result.cost_usd = meter.cost_usd
            cache_hit = bool(result.rag and result.rag.answer_cache_hit)
            self.c.ledger.settle(p.spend, self.c.ledger.row(
                tenant=ctx.tenant, user_id=ctx.user_id, request_id=ctx.request_id, intent=result.intent,
                models=meter.models, input_tokens=meter.input_tokens, output_tokens=meter.output_tokens,
                cost_usd=meter.cost_usd, success=success, cache_hit=cache_hit, degraded=plan.level))
            self.c.resilience.release(plan.admission, time.monotonic() - p.started)
        return result

    def _prompt_identity(self, intent: Intent) -> tuple[str, str]:
        """The registry prompt a workflow runs under; workflows without a model prompt say so explicitly."""
        if intent is Intent.QUESTION:
            ref = RAG_PROMPT
        elif intent is Intent.ACTION:
            ref = self.c.agent_prompt_ref
        else:
            return "none", "none"
        prompt_id, _, version = ref.partition("@")
        return prompt_id, version.split("#", 1)[0] or "unversioned"

    def _run(self, p: Prepared, out: _Emitter, meter: UsageMeter, manifest: Any, result: TurnResult, root: Any) -> bool:
        ctx, req, plan = p.ctx, p.req, p.plan
        gctx = ctx.guard_context()
        intent = p.intent

        # 1. input guard: PII tokenized before any model or index sees it; injection scored and flagged
        g = self.c.guards.input(req.message, gctx)
        root.set_attribute("guardrail.input", g.action)
        if not g.allowed:
            out("error", {"stage": "input_guard", "message": "The request was blocked by policy.",
                          "reasons": g.reasons})
            result.status = "blocked"
            out("done", self._done(result, meter, manifest, plan, None))
            return False
        text = g.text
        decision_task = {"rag.answer": "rag_answer", "agent.action": "support_action",
                         "extract.document": "extract_invoice", "memory.command": "route_intent",
                         "assist.smalltalk": "route_intent"}[intent.intent.value]
        needs_tools = intent.intent is Intent.ACTION
        with self.c.tracer.span("router.decide", **{"intent": intent.intent.value, "intent.rule": intent.rule}) as rs:
            route = self.c.models.decide(task=decision_task, needs_tools=needs_tools, plan=plan.plan)
            rs.set_attribute("route", route.route)
            rs.set_attribute("route.model", route.model)
            rs.set_attribute("route.stage", route.stage)
        client = self.c.models.client_for(route, meter=meter, deadline=ctx.deadline, plan=plan.plan,
                                          purpose=intent.intent.value)
        out("meta", {"request_id": ctx.request_id, "session_id": ctx.session_id, "intent": intent.intent.value,
                     "intent_rule": intent.rule, "route": route.route, "model_alias": route.model,
                     "model": client.model_id, "prompt_version": RAG_PROMPT if intent.intent is Intent.QUESTION
                     else self.c.agent_prompt_ref, "index_version": self.c.kb.index_version,
                     "manifest": manifest.fingerprint(), "degrade_level": plan.level,
                     "degrade_reasons": plan.plan.reasons, "flags": plan.flags, "trace_id": root.trace_id})
        if plan.plan.user_notice:
            out("notice", {"kind": "degraded", "message": plan.plan.user_notice, "level": plan.level})
        if input_flags := [r for r in g.reasons if r.startswith("injection")]:
            out("notice", {"kind": "input_flagged", "reasons": input_flags})

        # 2. memory: only the user's own words can write the profile; inferences wait for confirmation
        mem_events = self.c.memory.observe_user_turn(ctx.owner(), text, turn_ref=f"turn:{ctx.session_id}:{ctx.request_id}")
        for ev in mem_events:
            result.memory.append(ev.as_dict())
            out("memory", ev.as_dict())
        conversation = self.c.memory.conversation(ctx.owner(), ctx.session_id or ctx.request_id, client)

        # 3. the workflow chosen by the intent rules
        try:
            if intent.intent is Intent.QUESTION:
                ok = self._answer(ctx, text, client, plan, gctx, out, result, conversation)
            elif intent.intent is Intent.ACTION:
                ok = self._act(ctx, text, client, plan, gctx, out, result, conversation, intent.task)
            elif intent.intent is Intent.EXTRACT:
                ok = self._extract(ctx, req, text, gctx, client, out, result)
            elif intent.intent is Intent.MEMORY:
                ok = self._memory(ctx, text, out, result)
            else:
                ok = self._smalltalk(client, text, out, result)
        except DeadlineExceeded as exc:
            out("error", {"stage": getattr(exc, "stage", None) or "deadline", "message": str(exc)})
            result.status, ok = "timeout", False
        except (CircuitOpenError, LLMError) as exc:
            # A provider failure mid-request: fall back to the static plan for questions.
            out("notice", {"kind": "degraded", "message": "The model provider is unavailable; showing matching "
                           "documents instead.", "error": type(exc).__name__})
            root.set_attribute("degraded.fallback", type(exc).__name__)
            if intent.intent is Intent.QUESTION:
                static = Plan(plan=DEFAULT_PLANS[DegradeLevel.STATIC], admission=plan.admission, flags=plan.flags)
                ok = self._answer(ctx, text, None, static, gctx, out, result, conversation)
            else:
                result.status, ok = "unavailable", False
        conversation.add("user", text)
        if result.answer:
            conversation.add("assistant", result.answer[:2000])
        out("done", self._done(result, meter, manifest, plan, route))
        return ok

    # ================================================================== workflows
    def _answer(self, ctx: RequestContext, text: str, client: Any, plan: Plan, gctx: Any, out: _Emitter,
                result: TurnResult, conversation: Any) -> bool:
        rag = RagResult()
        result.rag = rag
        use_cache = self.c.settings.answer_cache and plan.flags.get("rag.answer_cache") == "on"
        for event, data in self.c.rag.answer_stream(ctx, text, client=client, plan=plan.plan, gctx=gctx, result=rag,
                                                    history=conversation.window(), use_answer_cache=use_cache):
            out(event, data)
        result.status, result.answer, result.citations = rag.status, rag.answer, rag.citations
        if rag.static:
            result.status = "degraded"
        return rag.status in ("answered", "partial", "conflict")

    def _agent_system_prompt(self, ctx: RequestContext, conversation: Any) -> str:
        rendered = self.c.prompts.get("assist.agent", "prod").render({"tenant": ctx.tenant, "state": ""})
        items = [ContextItem(kind="instructions", content=rendered.messages[0].text, source_id="prompt:assist.agent",
                             trust=Trust.TRUSTED, pinned=True)]
        profile = self.c.memory.render(ctx.owner())
        if profile:
            items.append(ContextItem(kind="memory", content=profile, source_id="memory:profile", priority=0.8,
                                     metadata={"tenant": ctx.tenant}))
        state = conversation.state_block()
        recent = "\n".join(f"{m.role.value}: {m.text[:400]}" for m in conversation.window()[-6:])
        if state or recent:
            items.append(ContextItem(kind="summary", content="\n".join(x for x in (state, recent) if x),
                                     source_id=f"session:{ctx.session_id}", priority=0.6,
                                     metadata={"tenant": ctx.tenant}))
        built = self.c.context_builder.build(items, RequestScope(user_id=ctx.user_id, tenant=ctx.tenant,
                                                                 groups=sorted(ctx.groups)))
        return built.messages[0].text

    def _act(self, ctx: RequestContext, text: str, client: Any, plan: Plan, gctx: Any, out: _Emitter,
             result: TurnResult, conversation: Any, task: str) -> bool:
        if not plan.plan.allow_agents:
            msg = plan.plan.user_notice or "Actions are temporarily disabled; questions still work."
            out("notice", {"kind": "agents_disabled", "message": msg})
            result.status, result.answer = "degraded", msg
            out("delta", {"text": msg})
            return False
        run_id = f"{ctx.request_id}.agent"
        run, activity = self.c.tools.run_agent(ctx, text, llm=client, system_prompt=self._agent_system_prompt(ctx, conversation),
                                               gctx=gctx, allow_side_effects=plan.plan.allow_side_effects,
                                               task=task, run_id=run_id)
        result.agent = run
        for call in activity.calls:
            result.tools.append(call)
            out("tool", call)
        for appr in activity.approvals:
            result.approvals.append(appr)
            out("approval", appr)
        answer = run.final_answer or f"I stopped before finishing ({run.stop_reason.value if run.stop_reason else 'unknown'})."
        g = self.c.guards.output(answer, gctx)
        answer = g.text if g.allowed else "The answer was withheld by policy."
        result.answer = answer
        out("delta", {"text": answer})
        reason = run.stop_reason.value if run.stop_reason else "unknown"
        result.status = "awaiting_approval" if activity.approvals else ("completed" if run.ok else f"stopped:{reason}")
        return run.ok

    def _extract(self, ctx: RequestContext, req: ChatRequest, text: str, gctx: Any, client: Any, out: _Emitter,
                 result: TurnResult) -> bool:
        if req.document:
            # An attached document crosses the provider boundary too, so it gets the same input guard
            # as the message: personal data is tokenized and injection text is scanned first.
            g = self.c.guards.input(req.document, gctx)
            if not g.allowed:
                msg = "The document was withheld by policy."
                result.answer, result.status = msg, "blocked"
                out("delta", {"text": msg})
                return False
            document = g.text
        else:
            document = text.split(":", 1)[1]           # the message itself, already guarded above
        res = self.c.extraction.extract(document, tenant=ctx.tenant, request_id=ctx.request_id, client=client,
                                        doc_type=req.doc_type)
        data = res.model_dump(mode="json")
        result.structured = data
        out("structured", data)
        msg = (f"Extracted a {res.doc_type.value if hasattr(res.doc_type, 'value') else res.doc_type}; "
               f"route: {res.route.value if hasattr(res.route, 'value') else res.route}.")
        result.answer, result.status = msg, "extracted"
        out("delta", {"text": msg})
        return True

    def _memory(self, ctx: RequestContext, text: str, out: _Emitter, result: TurnResult) -> bool:
        low = text.strip().lower()
        if low.startswith("forget"):
            key = low.replace("forget", "", 1).strip().removeprefix("my ").strip().replace(" ", "_")
            n = self.c.memory.forget(ctx.owner(), key)
            msg = f"Forgotten: {key} ({n} records deleted)." if n else f"I had nothing stored for {key}."
        elif result.memory:
            parts = [f"{m['key']} = {m['value']} ({m['kind']})" for m in result.memory]
            msg = "Noted: " + "; ".join(parts) + "."
        else:
            msg = "Tell me what to remember as 'remember that my <thing> is <value>'."
        result.answer, result.status = msg, "memory"
        out("delta", {"text": msg})
        return True

    def _smalltalk(self, client: Any, text: str, out: _Emitter, result: TurnResult) -> bool:
        completion = client.complete(CompletionRequest(messages=[Message.system(
            "You are Northwind Assist. Briefly say what you can help with."), Message.user(text)], max_tokens=120))
        result.answer, result.status = completion.text, "answered"
        out("delta", {"text": completion.text})
        return True

    # ================================================================== done
    def _done(self, result: TurnResult, meter: UsageMeter, manifest: Any, plan: Plan, route: Any) -> dict[str, Any]:
        rag = result.rag
        result.lineage = {
            "manifest": manifest.fingerprint(), "index_version": self.c.kb.index_version,
            "prompt": RAG_PROMPT if result.intent == Intent.QUESTION.value else self.c.agent_prompt_ref,
            "route": getattr(route, "route", None), "model_alias": getattr(route, "model", None),
            "policy_versions": {"tools": self.c.tools.policy.version, "guards": self.c.guards.version},
            "flags": plan.flags, "degrade_level": plan.level,
            "evidence_chunk_ids": [cid for c in result.citations for cid in c.get("chunk_ids", [])],
            "retrieval_cache_hit": bool(rag and rag.retrieval_cache_hit),
            "answer_cache_hit": bool(rag and rag.answer_cache_hit),
            "agent_run_id": getattr(result.agent, "run_id", None), "trace_id": result.trace_id,
        }
        return {"request_id": result.request_id, "status": result.status, "answer": result.answer,
                "citations": result.citations, "approvals": result.approvals, "memory": result.memory,
                "usage": {"input_tokens": meter.input_tokens, "output_tokens": meter.output_tokens,
                          "models": meter.models}, "cost_usd": meter.cost_usd, "lineage": result.lineage}


def _usage(input_tokens: int, output_tokens: int) -> Any:
    from aie_core.llm.types import Usage  # noqa: PLC0415

    return Usage(input_tokens=input_tokens, output_tokens=output_tokens)


__all__ = ["Orchestrator", "ChatRequest", "Prepared", "TurnResult", "Rejected"]
