# path: book/projects/ragkit/ragkit/generation/__init__.py
"""Grounded generation and citations (Chapter 13).

Input: the ranked `list[ScoredChunk]` produced by ragkit.retrieval (Chapter 12).
Output: a validated `GroundedAnswer` and an `AnswerEnvelope` a UI can render.
"""
from .abstain import ABSTAIN_MESSAGE, AbstentionDecision, AbstentionPolicy, Escalation, decide, pre_generation
from .generator import (
    CONTRACT_RULES,
    GROUNDED_SYSTEM_PROMPT,
    PROMPT_ID,
    PROMPT_VERSION,
    GenerationResult,
    GeneratorConfig,
    GroundedGenerator,
)
from .packer import EvidenceBlock, EvidencePacker, PackedEvidence, PackerConfig, PackNote, version_key
from .pipeline import GroundedQA, QAResult, build_envelope
from .schema import (
    AnswerEnvelope,
    AnswerStatus,
    Claim,
    GroundedAnswer,
    ResolvedCitation,
    ValidationIssue,
)
from .stream import AnswerEvent, GroundedStreamer, SentenceBuffer
from .support import lexical_support, markers, split_sentences, strip_markers
from .validator import (
    CitationValidator,
    JudgeVerdict,
    LLMGroundednessJudge,
    ValidationReport,
    ValidatorConfig,
    render_claims,
)

__all__ = [
    "ABSTAIN_MESSAGE",
    "AbstentionDecision",
    "AbstentionPolicy",
    "AnswerEnvelope",
    "AnswerEvent",
    "AnswerStatus",
    "CONTRACT_RULES",
    "CitationValidator",
    "Claim",
    "Escalation",
    "EvidenceBlock",
    "EvidencePacker",
    "GROUNDED_SYSTEM_PROMPT",
    "GenerationResult",
    "GeneratorConfig",
    "GroundedAnswer",
    "GroundedGenerator",
    "GroundedQA",
    "GroundedStreamer",
    "JudgeVerdict",
    "LLMGroundednessJudge",
    "PROMPT_ID",
    "PROMPT_VERSION",
    "PackNote",
    "PackedEvidence",
    "PackerConfig",
    "QAResult",
    "ResolvedCitation",
    "SentenceBuffer",
    "ValidationIssue",
    "ValidationReport",
    "ValidatorConfig",
    "build_envelope",
    "decide",
    "lexical_support",
    "markers",
    "pre_generation",
    "render_claims",
    "split_sentences",
    "strip_markers",
    "version_key",
]
