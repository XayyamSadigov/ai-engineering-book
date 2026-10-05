# path: book/capstone/northwind-assist/northwind_assist/rag/service.py
"""Grounded answers: retrieve under ACL -> sanitize -> pack -> stream with validation -> guard.

Composition, stage by stage:
  retrieval      ragkit RetrievalPipeline (BM25 + dense, RRF, lexical rerank), Ch 12, with the
                 Ch 30 RetrievalCache in front (ids only, re-hydrated and re-checked);
  context guard  guardrails context stage (Ch 27): hidden carriers removed, injection flagged;
  packing        ragkit EvidencePacker (Ch 13): budget, supersession, conflicts, instruction flags;
  generation     ragkit GroundedStreamer (Ch 13): sentence-level validation of [E#] citations,
                 driven by the routed, metered, deadline-bound client (Ch 3, 7, 29);
  output guard   guardrails output stage on every sentence (URL allowlist, canaries, secrets);
  answer cache   whole validated answers, scoped by tenant, ACL scope and versions (Ch 30).
Degraded plans (Ch 29) shrink k, drop the reranker, or skip the model entirely and return the
matching documents ("static" mode), which still respects the ACL because retrieval does.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from aie_core.llm.types import Message
from ragkit import Chunk
from ragkit.eval.rag_dataset import RagOutput
from ragkit.generation import EvidencePacker, GroundedStreamer, PackedEvidence, PackerConfig
from ragkit.generation.generator import PROMPT_VERSION
from ragkit.generation.stream import STREAM_PROMPT_ID
from ragkit.retrieval import RetrievalQuery, RetrievalResult, ScoredChunk
from reliability import DegradedPlan

from ..domain.context import RequestContext
from ..llm.models import RoutedClient
from ..security.guards import Guards
from .caches import CacheLayer
from .knowledge import Knowledge

RAG_PROMPT = f"{STREAM_PROMPT_ID}@{PROMPT_VERSION}"


@dataclass
class RagResult:
    status: str = "insufficient_evidence"
    answer: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    retrieval: RetrievalResult | None = None
    packed: PackedEvidence | None = None
    withheld: int = 0
    redactions: list[str] = field(default_factory=list)
    context_flags: list[str] = field(default_factory=list)
    retrieval_cache_hit: bool = False
    answer_cache_hit: bool = False
    static: bool = False

    @property
    def evidence_ids(self) -> list[str]:
        return [c["doc_id"] for c in self.citations]

    def to_rag_output(self) -> RagOutput:
        packed_chunks: list[Chunk] = []
        if self.packed is not None and self.retrieval is not None:
            by_id = {h.chunk.id: h.chunk for h in self.retrieval.hits}
            for b in self.packed.blocks:
                packed_chunks.extend(by_id[c] for c in b.chunk_ids if c in by_id)
        cited = [cid for c in self.citations for cid in c.get("chunk_ids", [])]
        return RagOutput(answer=self.answer, abstained=self.status == "insufficient_evidence",
                         cited_chunk_ids=cited, packed_chunks=packed_chunks, retrieval=self.retrieval,
                         metadata={"status": self.status, "withheld": self.withheld,
                                   "answer_cache_hit": self.answer_cache_hit})


class RagService:
    def __init__(self, kb: Knowledge, caches: CacheLayer, guards: Guards, *, tracer: Any = None,
                 packer_budget: int = 2400) -> None:
        self.kb = kb
        self.caches = caches
        self.guards = guards
        self.tracer = tracer
        self.packer = EvidencePacker(PackerConfig(token_budget=packer_budget, max_blocks=6))

    # ------------------------------------------------------------------ retrieval
    def retrieve(self, ctx: RequestContext, question: str, *, plan: DegradedPlan,
                 history: list[Message] | None = None) -> tuple[RetrievalResult, bool]:
        principal = ctx.principal()
        k = max(1, plan.retrieval_k or self.kb.settings.final_k)
        k = min(k, self.kb.settings.final_k)
        rerank = plan.rerank
        query = RetrievalQuery(text=question, principal=principal, k=k)
        scope = self.caches.scope(ctx.tenant, ctx.groups)
        holder: dict[str, RetrievalResult] = {}

        def compute() -> list[str]:
            result = self.kb.pipeline(rerank=rerank, principal=principal).retrieve(query, history)
            holder["r"] = result
            return [h.chunk.id for h in result.hits]

        ids, hit = self.caches.retrieval.get_or_compute(
            question, scope, compute, index_version=self.kb.index_version,
            retriever_config=f"rerank={rerank}:k={k}")
        if not hit:
            return holder["r"], False
        # Re-hydrate from the live chunk table and re-check the ACL: deletions and permission
        # changes after caching must win over the cache.
        live = self.kb.get_chunks(ids, principal)       # ACL-checked; deleted or forbidden ids drop out
        hits = [ScoredChunk(chunk=c, score=1.0 / (i + 1), stage="cache", rank=i + 1) for i, c in enumerate(live)]
        return RetrievalResult(query=query, hits=hits, trace={"cache": "hit", "stages": []}), True

    def _sanitize(self, ctx: RequestContext, hits: list[ScoredChunk], gctx: Any) -> tuple[list[ScoredChunk], list[str]]:
        out, flags = [], []
        for h in hits:
            g = self.guards.context(h.chunk.text, gctx, source=f"retrieved:{h.chunk.doc_id}")
            if not g.allowed:
                flags.append(f"{h.chunk.doc_id}: dropped ({g.blocked_by})")
                continue
            flags.extend(f"{h.chunk.doc_id}: {r}" for r in g.reasons if not r.startswith("context_sanitizer: wrapped"))
            chunk = h.chunk if g.text == h.chunk.text else h.chunk.model_copy(update={"text": g.text})
            out.append(h.model_copy(update={"chunk": chunk}))
        return out, flags

    # ------------------------------------------------------------------ answer
    def answer_stream(self, ctx: RequestContext, question: str, *, client: RoutedClient | None, plan: DegradedPlan,
                      gctx: Any, result: RagResult, history: list[Message] | None = None,
                      use_answer_cache: bool = True) -> Iterator[tuple[str, dict]]:
        """Yield (event, data) pairs; fill `result` as a side record for persistence and eval."""
        model = client.model_id if client is not None else "none"
        scope = self.caches.scope(ctx.tenant, ctx.groups)
        key = self.caches.answer_key(question, scope, prompt_version=RAG_PROMPT, index_version=self.kb.index_version,
                                     model=model, plan_level=int(plan.level))
        cached = self.caches.get_answer(key) if plan.use_model and use_answer_cache else None
        if cached is not None:
            result.answer_cache_hit = True
            result.status, result.answer, result.citations = cached["status"], cached["answer"], cached["citations"]
            for c in result.citations:
                yield "citation", c
            if result.answer:
                yield "delta", {"text": result.answer}
            return

        retrieval, result.retrieval_cache_hit = self.retrieve(ctx, question, plan=plan, history=history)
        hits, result.context_flags = self._sanitize(ctx, retrieval.hits, gctx)
        result.retrieval = retrieval.model_copy(update={"hits": hits})
        packed = self.packer.pack(hits, ctx.principal())
        result.packed = packed
        if self.tracer is not None:
            with self.tracer.span("context.build", **{"evidence.ids": packed.eids, "pack.tokens": packed.token_count,
                                                      "context.flags": result.context_flags[:10]}):
                pass

        if not plan.use_model or client is None:   # static degraded mode: documents, no model
            result.static = True
            seen: list[str] = []
            for b in packed.blocks:
                if b.doc_id not in seen:
                    seen.append(b.doc_id)
                    cit = b.to_citation().model_dump(mode="json")
                    result.citations.append(cit)
                    yield "citation", cit
            result.status = "partial" if seen else "insufficient_evidence"
            result.answer = ("The assistant is in reduced mode. These documents match your question: " +
                             "; ".join(c["title"] or c["doc_id"] for c in result.citations)) if seen else ""
            if result.answer:
                yield "delta", {"text": result.answer}
            return

        streamer = GroundedStreamer(client)
        sentences: list[str] = []
        with (self.tracer.span("llm.generate", **{"prompt.id": STREAM_PROMPT_ID, "prompt.version": PROMPT_VERSION,
                                                  "route.alias": client.alias, "model": client.model_id})
              if self.tracer is not None else _null()) as span:
            for ev in streamer.stream(question, packed):
                if ev.type == "citation" and ev.citation is not None:
                    cit = ev.citation.model_dump(mode="json")
                    result.citations.append(cit)
                    yield "citation", cit
                elif ev.type == "text" and ev.text:
                    g = self.guards.output(ev.text, gctx)
                    if not g.allowed:
                        result.withheld += 1
                        result.redactions.extend(g.reasons)
                        yield "notice", {"kind": "withheld", "reason": g.blocked_by}
                        continue
                    if g.action == "redact":
                        result.redactions.extend(g.reasons)
                    sentences.append(g.text)
                    yield "delta", {"text": g.text}
                elif ev.type == "withheld":
                    result.withheld += 1
                    yield "notice", {"kind": "withheld", "reason": ev.issue.code if ev.issue else "validation"}
                elif ev.type == "error":
                    raise RuntimeError(ev.text or "generation stream failed")
                elif ev.type == "done" and ev.answer is not None:
                    result.status = ev.status or ev.answer.status
            if span is not None:
                span.set_attribute("answer.status", result.status)
                span.set_attribute("answer.withheld", result.withheld)
                span.set_attribute("answer.citations", [c["eid"] for c in result.citations])
        result.answer = "".join(sentences).strip()
        if result.status == "insufficient_evidence":
            result.citations = []
        if use_answer_cache and result.status in ("answered", "insufficient_evidence") and not result.withheld \
                and not result.redactions:
            self.caches.put_answer(key, {"status": result.status, "answer": result.answer,
                                         "citations": result.citations})


class _null:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None


__all__ = ["RagService", "RagResult", "RAG_PROMPT"]
