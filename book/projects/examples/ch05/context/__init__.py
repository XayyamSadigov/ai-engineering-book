# path: book/projects/examples/ch05/context/__init__.py
"""Chapter 5: context engineering. Build the smallest high-value prompt that fits a budget."""
from .builder import (
    BudgetPolicy,
    BuildResult,
    ContextBuilder,
    ContextOverflowError,
    ManifestEntry,
    SectionLimits,
    edge_order,
)
from .filters import RequestScope, acl_filter, min_score_filter
from .items import ContextItem, Section, Trust
from .labels import UNTRUSTED_NOTICE, neutralize, render_item
from .layout import PrefixStabilityTracker, lint_stable_items, shared_prefix_tokens
from .state import (
    CompactionReport,
    ConversationState,
    Fact,
    InMemoryStateStore,
    LLMSummarizer,
    StaleStateError,
    StateSnapshot,
    Summarizer,
    Turn,
)

__all__ = [
    "BudgetPolicy", "BuildResult", "ContextBuilder", "ContextOverflowError", "ManifestEntry",
    "SectionLimits", "edge_order", "RequestScope", "acl_filter", "min_score_filter",
    "ContextItem", "Section", "Trust", "UNTRUSTED_NOTICE", "neutralize", "render_item",
    "PrefixStabilityTracker", "lint_stable_items", "shared_prefix_tokens",
    "CompactionReport", "ConversationState", "Fact", "LLMSummarizer", "Summarizer", "Turn",
    "StateSnapshot", "StaleStateError", "InMemoryStateStore",
]
