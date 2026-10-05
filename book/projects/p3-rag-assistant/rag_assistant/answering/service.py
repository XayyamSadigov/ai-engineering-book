# path: book/projects/p3-rag-assistant/rag_assistant/answering/service.py
"""AnswerService: the request path, from a verified principal to a cited answer or a safe fallback.

    admit -> guard input -> refresh index view -> answer cache? -> retrieval cache? -> retrieve
          -> guard context -> pack -> generate -> validate -> abstain? -> guard output -> cache -> respond

The composition is the whole point of this module; each step is someone else's code:
retrieval is ragkit.retrieval.RetrievalPipeline (Chapter 12), packing, generation, validation and
abstention are ragkit.generation.GroundedQA (Chapter 13), guardrails are guardrails.rag_pipeline
(Chapter 27), budgets and breakers are reliability (Chapter 29). What lives here is the order,
the keys, and the degraded modes:

    dense retriever down   -> lexical only            (RetrievalPipeline skips it; degraded "retrieve:dense")
    reranker down          -> fused order             (degraded "rerank:fallback")
    LLM down / breaker open / budget spent -> sources only, no generated text
    every retriever down   -> mode "unavailable" (HTTP 503)

Authorization is enforced before anything reaches a model: retrieval pre-filters by tenant and
groups, the packer re-checks, and nothing is ever filtered after generation.
"""
from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Iterator

from aie_core.llm.errors import LLMError
from aie_core.observability import NoopTracer, Tracer
from guardrails import Action, GuardrailPipeline
from ragkit.generation import (
    ABSTAIN_MESSAGE,
    AnswerEnvelope,
    GroundedQA,
    GroundedStreamer,
    PackedEvidence,
    QAResult,
    pre_generation,
)
from ragkit.generation.generator import PROMPT_VERSION
from ragkit.retrieval import Principal, RetrievalError, RetrievalQuery, RetrievalResult, ScoredChunk
from reliability import AdmissionController, AdmissionRequest, CircuitBreakerRegistry, CircuitState, Deadline

from ..adapters.llm import DeadlineBoundLLM
from ..caching.caches import CacheSet, guard_context
from ..config import AssistantSettings
from ..domain.keys import normalize_query, principal_scopes
from ..domain.models import AskResponse, SourceSnippet
from ..ingestion.registry import DocumentRegistry
from ..observability.metrics import Metrics
from ..retrieval.index import IndexSet
from ..retrieval.wiring import build_pipeline


@dataclass
class AskOutcome:
    """The response plus the internals evaluation and tests need (never serialized to clients)."""

    response: AskResponse
    retrieval: RetrievalResult | None = None
    qa: QAResult | None = None
    security_events: list[str] = field(default_factory=list)


@dataclass
class _Prepared:
    request_id: str
    principal: Principal
    question: str
    version: str
    gens: dict[str, int]
    degrade_level: int
    deadline: Deadline
    timings: dict[str, float]
    retrieval: RetrievalResult | None = None
    hits: list[ScoredChunk] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)
    security: list[str] = field(default_factory=list)
    cache: str = "miss"
    early: AskOutcome | None = None


class _Clock:
    def __init__(self, timings: dict[str, float], stage: str) -> None:
        self.timings, self.stage = timings, stage

    def __enter__(self) -> "_Clock":
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.timings[self.stage] = round((time.perf_counter() - self.t0) * 1000, 3)


class AnswerService:
    def __init__(self, settings: AssistantSettings, *, registry: DocumentRegistry, index_set: IndexSet,
                 qa: GroundedQA, streamer: GroundedStreamer, guards: GuardrailPipeline, caches: CacheSet,
                 breakers: CircuitBreakerRegistry, metrics: Metrics, tracer: Tracer | None = None,
                 admission: AdmissionController | None = None, model_id: str = "default") -> None:
        self.s = settings
        self.registry = registry
        self.index_set = index_set
        self.qa = qa
        self.streamer = streamer
        self.guards = guards
        self.caches = caches
        self.breakers = breakers
        self.metrics = metrics
        self.tracer = tracer or NoopTracer()
        self.admission = admission
        self.model_id = model_id
        self._seen_view: tuple[str, tuple[tuple[str, int], ...]] | None = None
        self._executor = ThreadPoolExecutor(max_workers=16, thread_name_prefix="rag-retrieve")

    # ------------------------------------------------------------------ public
    def ask(self, question: str, principal: Principal, *, request_id: str | None = None) -> AskOutcome:
        rid = request_id or uuid.uuid4().hex[:16]
        decision = self._admit(principal)
        t0 = time.perf_counter()
        try:
            with self.tracer.span("rag.request", **{"request.id": rid, "tenant.id": principal.tenant,
                                                    "user.id": principal.user_id, "stream": False}) as span:
                outcome = self._ask(question, principal, rid, decision.degrade_level if decision else 0)
                r = outcome.response
                span.set_attribute("mode", r.mode)
                span.set_attribute("cache", r.cache)
                span.set_attribute("index.version", r.index_version)
                span.set_attribute("degraded", r.degraded)
                span.set_attribute("action", r.answer.action if r.answer else "")
        finally:
            if decision is not None and self.admission is not None:
                self.admission.release(decision, time.perf_counter() - t0)
        self._record(outcome, (time.perf_counter() - t0) * 1000)
        return outcome

    def stream(self, question: str, principal: Principal, *, request_id: str | None = None) -> Iterator[tuple[str, dict[str, Any]]]:
        """Server-sent events as (event, data) pairs: meta, sources?, citation*, text*, done | error."""
        rid = request_id or uuid.uuid4().hex[:16]
        decision = self._admit(principal)
        t0 = time.perf_counter()
        try:
            prep = self._prepare(question, principal, rid, decision.degrade_level if decision else 0, use_answer_cache=False)
            meta = {"request_id": rid, "index_version": prep.version, "degraded": prep.degraded, "cache": prep.cache}
            if prep.early is not None:
                yield "meta", meta
                yield "done", prep.early.response.model_dump(mode="json")
                self._record(prep.early, (time.perf_counter() - t0) * 1000)
                return
            yield "meta", meta
            packed = self.qa.packer.pack(prep.hits, principal)
            early = pre_generation(prep.hits, packed, self.qa.policy)
            if early is not None:
                yield "done", {"request_id": rid, "mode": "answer", "status": "insufficient_evidence",
                               "action": "abstain", "text": early.user_message}
                return
            if not self._can_generate(prep):
                out = self._sources_only(prep, packed, reason="generate:budget")
                yield "done", out.response.model_dump(mode="json")
                return
            ctx = guard_context(principal, rid)
            ctx.evidence_ids = frozenset(packed.eids)
            withheld = 0
            final: dict[str, Any] | None = None
            # A generator outlives any `with deadline.scope()` (the server resumes it in other
            # contexts), so the budget is bound to the LLM client explicitly instead.
            streamer = GroundedStreamer(DeadlineBoundLLM(self.streamer.llm, prep.deadline),
                                        generator=self.qa.generator)
            try:
                with _Clock(prep.timings, "generate"):
                    for ev in streamer.stream(prep.question, packed):
                        if ev.type == "citation" and ev.citation is not None:
                            yield "citation", ev.citation.model_dump(mode="json")
                        elif ev.type == "text" and ev.text:
                            verdict = self.guards.check_output(ev.text, ctx)
                            if not verdict.allowed:
                                withheld += 1
                                self.metrics.inc("rag_security_events_total", kind=f"output:{verdict.blocked_by}")
                                continue
                            yield "text", {"text": verdict.text, "eids": ev.eids}
                        elif ev.type == "withheld":
                            withheld += 1  # logged and counted, never shown
                        elif ev.type == "error":
                            raise LLMError(ev.text or "stream error")
                        elif ev.type == "done" and ev.answer is not None:
                            status = ev.status or ev.answer.status
                            final = {"request_id": rid, "mode": "answer", "status": status,
                                     "action": "abstain" if status == "insufficient_evidence" else "answer",
                                     "withheld": withheld, "prompt_version": PROMPT_VERSION}
                            if status == "insufficient_evidence":
                                final["text"] = f"{ABSTAIN_MESSAGE} You can ask {self.qa.policy.fallback_contact}."
            except LLMError as exc:
                out = self._sources_only(prep, packed, reason=f"generate:{type(exc).__name__}")
                yield "error", {"message": "The answer could not be generated; showing sources instead."}
                yield "done", out.response.model_dump(mode="json")
                return
            yield "done", final or {"request_id": rid, "mode": "answer", "status": "insufficient_evidence",
                                    "action": "abstain"}
        finally:
            if decision is not None and self.admission is not None:
                self.admission.release(decision, time.perf_counter() - t0)

    # ------------------------------------------------------------------ the non-streaming path
    def _ask(self, question: str, principal: Principal, rid: str, level: int) -> AskOutcome:
        prep = self._prepare(question, principal, rid, level, use_answer_cache=True)
        if prep.early is not None:
            return prep.early
        packed: PackedEvidence | None = None
        if not self._can_generate(prep):
            packed = self.qa.packer.pack(prep.hits, principal)
            return self._sources_only(prep, packed, reason="generate:budget")
        try:
            with prep.deadline.scope(), _Clock(prep.timings, "generate"):
                qa = self.qa.answer(prep.question, prep.hits, principal, request_id=rid)
        except LLMError as exc:  # provider down, breaker open, deadline spent, malformed output
            packed = self.qa.packer.pack(prep.hits, principal)
            return self._sources_only(prep, packed, reason=f"generate:{type(exc).__name__}")
        envelope = self._guard_output(qa, prep)
        security = prep.security + list(qa.decision.security_events)
        response = AskResponse(request_id=rid, mode="answer", answer=envelope, citations=envelope.citations,
                               sources=_snippets(prep.hits) if envelope.action == "abstain" else [],
                               degraded=prep.degraded, cache=prep.cache, index_version=prep.version,
                               timings_ms=prep.timings)
        outcome = AskOutcome(response=response, retrieval=prep.retrieval, qa=qa, security_events=security)
        if response.answer is not None and response.answer.action == "abstain":
            response.sources = []  # an abstention lists nothing: the visible hits were not good enough
        if self.s.answer_cache and not prep.degraded and envelope.action in ("answer", "answer_with_caveat"):
            ctx = guard_context(principal)
            docs = {c.doc_id for c in envelope.citations} | {b.doc_id for b in qa.packed.blocks}
            self.caches.answers.put(ctx, outcome, *self._answer_key(prep), doc_ids=docs,
                                    stamp=self._stamp(prep))
        return outcome

    def _prepare(self, question: str, principal: Principal, rid: str, level: int, *,
                 use_answer_cache: bool) -> _Prepared:
        timings: dict[str, float] = {}
        deadline = Deadline.after(self.s.request_deadline_s, name=f"ask:{rid}")
        ctx = guard_context(principal, rid)
        with _Clock(timings, "guard_input"):
            verdict = self.guards.check_input(question, ctx)
        prep = _Prepared(request_id=rid, principal=principal, question=verdict.text, version="", gens={},
                         degrade_level=level, deadline=deadline, timings=timings)
        if level > 0:
            prep.degraded.append(f"admission:level{level}")
        if verdict.flags:
            prep.security.extend(f"input:{v.check}" for v in verdict.flags)
        if not verdict.allowed:
            prep.early = AskOutcome(AskResponse(request_id=rid, mode="blocked", timings_ms=timings,
                                                message="This request cannot be processed."),
                                    security_events=[f"input_blocked:{verdict.blocked_by}"])
            return prep

        with _Clock(timings, "refresh"):
            self._refresh_view()
        prep.version = self.index_set.active_version
        prep.gens = self.registry.generations(principal_scopes(principal))

        if use_answer_cache and self.s.answer_cache and level == 0:
            with _Clock(timings, "answer_cache"):
                cached = self.caches.answers.get(guard_context(principal), *self._answer_key(prep))
            self.metrics.inc("rag_cache_lookups_total", cache="answer", result="hit" if cached else "miss")
            if cached is not None:
                resp = cached.response.model_copy(update={"request_id": rid, "cache": "hit", "timings_ms": timings})
                prep.early = AskOutcome(resp, retrieval=cached.retrieval, qa=cached.qa)
                return prep

        result: RetrievalResult | None = None
        rkey = ("v1", normalize_query(prep.question), prep.version, tuple(sorted(prep.gens.items())), level,
                self.s.final_k, self.s.reranker)
        if self.s.retrieval_cache:
            result = self.caches.retrieval.get(guard_context(principal), *rkey)
            self.metrics.inc("rag_cache_lookups_total", cache="retrieval", result="hit" if result else "miss")
            prep.cache = "hit" if result is not None else "miss"
        if result is None:
            hidden = frozenset(r.doc_id for r in self.registry.all("deleting"))
            pipeline = build_pipeline(self.index_set, principal, self.s, breakers=self.breakers, hidden=hidden,
                                      degrade_level=level, tracer=self.tracer, version=prep.version,
                                      executor=self._executor)
            try:
                with _Clock(timings, "retrieve"):
                    result = pipeline.retrieve(RetrievalQuery(text=prep.question, principal=principal,
                                                              k=self.s.final_k))
            except RetrievalError:
                prep.degraded.append("retrieve:all")
                prep.early = AskOutcome(AskResponse(request_id=rid, mode="unavailable", degraded=prep.degraded,
                                                    index_version=prep.version, timings_ms=timings,
                                                    message="Search is temporarily unavailable."))
                return prep
            if self.s.retrieval_cache and not result.trace.get("degraded") and level == 0:
                # never cache a degraded result: it would pin lower quality after recovery
                self.caches.retrieval.put(guard_context(principal), result, *rkey, doc_ids=result.doc_ids,
                                          stamp=self._stamp(prep))
        for stage, ms in (result.trace.get("latency_ms") or {}).items():
            self.metrics.observe("rag_stage_latency_ms", ms, stage=f"retrieval.{stage}")
        prep.degraded.extend(result.trace.get("degraded") or [])
        if result.trace.get("acl_violations"):
            prep.security.append(f"acl_violation:{len(result.trace['acl_violations'])}")
        prep.retrieval = result

        with _Clock(timings, "guard_context"):
            prep.hits = self._guard_context(result.hits, prep)
        return prep

    # ------------------------------------------------------------------ steps
    def _admit(self, principal: Principal):  # type: ignore[no-untyped-def]
        if self.admission is None:
            return None
        decision = self.admission.admit(AdmissionRequest(
            tenant_id=principal.tenant, deadline_s=self.s.request_deadline_s,
            est_tokens=self.s.evidence_token_budget + self.s.generation_max_tokens))
        if not decision.admitted:
            from reliability import AdmissionRejected

            raise AdmissionRejected(f"{decision.action.value}: {decision.reason}", reason=decision.reason,
                                    retry_after_s=decision.retry_after_s)
        return decision

    def _refresh_view(self) -> None:
        """Pick up worker snapshots; when generations moved, sweep entries that became unreachable."""
        self.index_set.refresh()
        view = (self.index_set.active_version, tuple(sorted(self.registry.generations().items())))
        if view != self._seen_view:
            gens = dict(view[1])

            def stale(stamp: Any) -> bool:
                if not stamp:
                    return False
                version, stamped = stamp
                return version != view[0] or any(gens.get(s, 0) != g for s, g in stamped)

            self.caches.sweep(stale)
            self._seen_view = view

    def _guard_context(self, hits: list[ScoredChunk], prep: _Prepared) -> list[ScoredChunk]:
        ctx = guard_context(prep.principal, prep.request_id)
        kept: list[ScoredChunk] = []
        for h in hits:
            res = self.guards.check_context(h.chunk.text, ctx, source=f"retrieved:{h.chunk.doc_id}")
            if not res.allowed:
                prep.security.append(f"context_blocked:{h.chunk.doc_id}")
                continue
            if res.action in (Action.FLAG, Action.REDACT) and res.flags:
                prep.security.append(f"context_flagged:{h.chunk.doc_id}")  # the packer flags the span too
            kept.append(h)
        return kept

    def _can_generate(self, prep: _Prepared) -> bool:
        if prep.degrade_level >= 2:
            prep.degraded.append("generate:shed")
            return False
        if not prep.deadline.fits(self.s.min_generation_s):
            prep.degraded.append("generate:deadline")
            return False
        if self.breakers.get("llm").state is CircuitState.OPEN:  # fail fast: do not wait for a timeout
            prep.degraded.append("generate:breaker_open")
            return False
        return True

    def _guard_output(self, qa: QAResult, prep: _Prepared) -> AnswerEnvelope:
        env = qa.envelope
        if env.action not in ("answer", "answer_with_caveat"):
            return env
        ctx = guard_context(prep.principal, prep.request_id)
        ctx.evidence_ids = frozenset(qa.packed.eids)
        res = self.guards.check_output(env.text, ctx)
        if not res.allowed:
            prep.security.append(f"output_blocked:{res.blocked_by}")
            return AnswerEnvelope(request_id=env.request_id, status="insufficient_evidence", action="abstain",
                                  text=f"{ABSTAIN_MESSAGE} You can ask {self.qa.policy.fallback_contact}.",
                                  prompt_version=env.prompt_version)
        if res.text != env.text:
            prep.security.append("output_redacted")
            return env.model_copy(update={"text": res.text})
        return env

    def _sources_only(self, prep: _Prepared, packed: PackedEvidence, *, reason: str) -> AskOutcome:
        if reason not in prep.degraded:
            prep.degraded.append(reason)
        blocks = {b.doc_id: b for b in packed.blocks}
        sources = [SourceSnippet(doc_id=b.doc_id, title=b.title, version=b.version, section=b.section,
                                 source_uri=b.source_uri, snippet=_clip(b.clean_text())) for b in blocks.values()]
        response = AskResponse(request_id=prep.request_id, mode="sources_only", sources=sources,
                               degraded=prep.degraded, cache=prep.cache, index_version=prep.version,
                               timings_ms=prep.timings,
                               message="The assistant cannot write an answer right now. These documents matched your question.")
        return AskOutcome(response, retrieval=prep.retrieval, security_events=prep.security)

    # ------------------------------------------------------------------ keys and telemetry
    def _answer_key(self, prep: _Prepared) -> tuple[Any, ...]:
        return ("v1", normalize_query(prep.question), prep.version, tuple(sorted(prep.gens.items())),
                PROMPT_VERSION, self.model_id, self.s.guard_policy_version, self.s.final_k, self.s.reranker)

    @staticmethod
    def _stamp(prep: _Prepared) -> tuple[str, tuple[tuple[str, int], ...]]:
        return prep.version, tuple(sorted(prep.gens.items()))

    def _record(self, outcome: AskOutcome, total_ms: float) -> None:
        r = outcome.response
        self.metrics.inc("rag_requests_total", mode=r.mode, cache=r.cache)
        self.metrics.observe("rag_stage_latency_ms", total_ms, stage="total")
        for stage, ms in r.timings_ms.items():
            self.metrics.observe("rag_stage_latency_ms", ms, stage=stage)
        for reason in r.degraded:
            self.metrics.inc("rag_degraded_total", reason=reason.split(":")[0] + ":" + reason.split(":")[-1])
        for ev in outcome.security_events:
            self.metrics.inc("rag_security_events_total", kind=ev.split(":")[0])
        if outcome.qa is not None:
            self.metrics.observe("rag_evidence_tokens", outcome.qa.packed.token_count)


def _clip(text: str, n: int = 280) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3].rstrip() + "..."


def _snippets(hits: list[ScoredChunk]) -> list[SourceSnippet]:
    seen: dict[str, SourceSnippet] = {}
    for h in hits:
        c = h.chunk
        if c.doc_id not in seen:
            seen[c.doc_id] = SourceSnippet(doc_id=c.doc_id, title=str(c.metadata.get("title", "")), version=c.version,
                                           section=" > ".join(c.section_path), source_uri=c.metadata.get("source_uri"),
                                           snippet=_clip(c.text))
    return list(seen.values())


__all__ = ["AnswerService", "AskOutcome"]
