# path: book/projects/memorykit/memorykit/__init__.py
"""memorykit: memory stores for AI applications (Chapter 21).

Typed memory records with provenance and expiry, tenant-scoped stores with hard delete and
tombstones, a write policy that blocks poisoning, consolidation, and four memory types:
conversation, semantic, episodic, and user profile.
"""
from .conversation import ConversationMemory, ExtractedFact, FactExtraction, SessionFact, Turn
from .episodic import Episode, EpisodicStore, ScoredEpisode
from .evaluation import (
    RecallCase,
    RecallReport,
    WriteCase,
    WritePolicyReport,
    evaluate_recall,
    evaluate_write_policy,
)
from .models import (
    SOURCE_PRECEDENCE,
    UNTRUSTED_SOURCES,
    MemoryKind,
    MemoryRecord,
    MemoryStatus,
    Owner,
    Sensitivity,
    Source,
    Tombstone,
)
from .policy import (
    ConsolidationAction,
    ConsolidationResult,
    Decision,
    WriteDecision,
    WriteOutcome,
    WritePolicy,
    consolidate,
    write,
)
from .profile import ProfileError, UserProfileMemory
from .semantic import RecallResult, ScoredMemory, ScoringWeights, SemanticMemory, render_memories
from .store import InMemoryStore, MemoryStore, SQLiteStore, VersionConflict

__version__ = "0.1.0"

__all__ = [
    "MemoryKind", "MemoryRecord", "MemoryStatus", "Owner", "Sensitivity", "Source", "Tombstone",
    "SOURCE_PRECEDENCE", "UNTRUSTED_SOURCES",
    "MemoryStore", "InMemoryStore", "SQLiteStore", "VersionConflict",
    "WritePolicy", "WriteDecision", "WriteOutcome", "Decision", "write",
    "consolidate", "ConsolidationAction", "ConsolidationResult",
    "ConversationMemory", "Turn", "SessionFact", "ExtractedFact", "FactExtraction",
    "SemanticMemory", "ScoringWeights", "ScoredMemory", "RecallResult", "render_memories",
    "EpisodicStore", "Episode", "ScoredEpisode",
    "UserProfileMemory", "ProfileError",
    "RecallCase", "RecallReport", "evaluate_recall", "WriteCase", "WritePolicyReport", "evaluate_write_policy",
]
