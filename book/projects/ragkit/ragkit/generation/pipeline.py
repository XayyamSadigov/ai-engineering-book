# path: book/projects/ragkit/ragkit/generation/pipeline.py
"""GroundedQA: pack -> (abstain early?) -> generate -> validate -> decide -> AnswerEnvelope.

This is the generation half of the RAG request path. Retrieval (Chapter 12) produces the
hits; Project 3 (Chapter 15) wraps this in an API with ACL-aware retrieval, caching and
tracing. Every intermediate object is returned in `QAResult` so tests and traces can see why
the envelope looks the way it does.
"""
from __future__ import annotations

import uuid

from aie_core.observability import NoopTracer, Tracer
from pydantic import BaseModel, ConfigDict

from ..retrieval.types import Principal, ScoredChunk
from .abstain import AbstentionDecision, AbstentionPolicy, decide, pre_generation
from .generator import PROMPT_VERSION, GenerationResult, GroundedGenerator
from .packer import EvidencePacker, PackedEvidence
from .schema import AnswerEnvelope, GroundedAnswer
from .validator import CitationValidator, ValidationReport


class QAResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    envelope: AnswerEnvelope
    packed: PackedEvidence
    decision: AbstentionDecision
    generation: GenerationResult | None = None
    report: ValidationReport | None = None


class GroundedQA:
    def __init__(self, generator: GroundedGenerator, packer: EvidencePacker | None = None,
                 validator: CitationValidator | None = None, policy: AbstentionPolicy | None = None,
                 tracer: Tracer | None = None) -> None:
        self.generator = generator
        self.packer = packer or EvidencePacker()
        self.validator = validator or CitationValidator()
        self.policy = policy or AbstentionPolicy()
        self.tracer = tracer or NoopTracer()

    def answer(self, question: str, hits: list[ScoredChunk], principal: Principal | None = None,
               request_id: str | None = None) -> QAResult:
        rid = request_id or uuid.uuid4().hex[:12]
        with self.tracer.span("rag.answer", request_id=rid, hits=len(hits)) as span:
            packed = self.packer.pack(hits, principal)
            span.set_attribute("pack.blocks", len(packed.blocks))
            span.set_attribute("pack.tokens", packed.token_count)
            early = pre_generation(hits, packed, self.policy)
            if early is not None:
                span.set_attribute("decision", early.action)
                return QAResult(envelope=_abstain_envelope(rid, early), packed=packed, decision=early)

            generation = self.generator.generate(question, packed)
            report = self.validator.validate(generation.answer, packed)
            decision = decide(packed, report, self.policy)
            span.set_attribute("answer.status", report.repaired.status)
            span.set_attribute("validation.errors", len(report.errors))
            span.set_attribute("decision", decision.action)
            return QAResult(envelope=build_envelope(rid, report, decision), packed=packed,
                            decision=decision, generation=generation, report=report)


def _abstain_envelope(rid: str, decision: AbstentionDecision) -> AnswerEnvelope:
    return AnswerEnvelope(request_id=rid, status="insufficient_evidence", action=decision.action,
                          text=decision.user_message or GroundedAnswer.insufficient("").answer,
                          prompt_version=PROMPT_VERSION)


def build_envelope(rid: str, report: ValidationReport, decision: AbstentionDecision) -> AnswerEnvelope:
    ans = report.repaired
    showing_answer = decision.action in ("answer", "answer_with_caveat")
    return AnswerEnvelope(
        request_id=rid,
        status=ans.status,
        action=decision.action,
        text=ans.answer if showing_answer else (decision.user_message or ans.answer),
        citations=report.citations if showing_answer else [],
        missing_info=ans.missing_info,
        conflicts=ans.conflicts,
        notices=decision.notices,
        confidence=ans.confidence,
        issues=report.issues,
        prompt_version=PROMPT_VERSION,
    )


__all__ = ["GroundedQA", "QAResult", "build_envelope"]
