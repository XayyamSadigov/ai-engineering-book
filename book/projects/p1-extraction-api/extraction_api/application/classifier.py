# path: book/projects/p1-extraction-api/extraction_api/application/classifier.py
"""Document-type classification with abstention.

Two confidence signals are available without provider-specific features: the model's
verbalized confidence, and agreement across several samples (self-consistency). Agreement
is the better-calibrated of the two but multiplies cost by the number of samples, so it is
off by default and worth enabling only where misrouting is expensive.
"""
from __future__ import annotations

from collections import Counter

from aie_core import CompletionRequest, LLMClient, MalformedResponseError, Message
from aie_core.llm.structured import complete_structured
from pydantic import BaseModel, ConfigDict, Field

from ..domain.common import DocumentType
from .prompts import CLASSIFY_SYSTEM, render_document


class DocumentClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    doc_type: DocumentType
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(max_length=300)


class ClassificationResult(BaseModel):
    doc_type: DocumentType
    confidence: float
    abstained: bool
    reason: str


class DocumentClassifier:
    def __init__(
        self,
        threshold: float = 0.70,
        samples: int = 1,
        sample_temperature: float = 0.7,
        max_chars: int = 4000,
    ) -> None:
        self.threshold = threshold
        self.samples = max(1, samples)
        self.sample_temperature = sample_temperature
        self.max_chars = max_chars  # the head of a document is enough to know its type

    def classify(self, client: LLMClient, text: str, *, timeout_s: float | None = None) -> ClassificationResult:
        req = CompletionRequest(
            messages=[Message.system(CLASSIFY_SYSTEM), Message.user(render_document(text[: self.max_chars]))],
            temperature=0.0 if self.samples == 1 else self.sample_temperature,
            max_tokens=200,
            metadata={"task": "classify"},
            timeout_s=timeout_s,
        )
        votes: list[DocumentClassification] = []
        for _ in range(self.samples):
            try:
                result, _ = complete_structured(client, req, DocumentClassification, max_repair_attempts=1)
            except MalformedResponseError:
                continue
            votes.append(result)  # type: ignore[arg-type]
        if not votes:
            return ClassificationResult(doc_type=DocumentType.OTHER, confidence=0.0, abstained=True,
                                        reason="classifier output unusable")
        winner, count = Counter(v.doc_type for v in votes).most_common(1)[0]
        if self.samples == 1:
            confidence = votes[0].confidence
        else:
            confidence = count / self.samples  # agreement, not self-report
        reason = next(v.reason for v in votes if v.doc_type == winner)
        abstained = winner is DocumentType.OTHER or confidence < self.threshold
        return ClassificationResult(doc_type=winner, confidence=confidence, abstained=abstained, reason=reason)


__all__ = ["DocumentClassification", "ClassificationResult", "DocumentClassifier"]
