# path: book/projects/ragkit/ragkit/retrieval/__init__.py
"""Retrieval stages (Chapter 12). Shared contract lives in `types`.

Lexical (bm25), dense (dense), fusion (hybrid), rerankers (rerank), query transformation (query),
diversity-aware selection (diversity), parent-document retrieval (parent), contextual retrieval (contextual), and the funnel that
composes them (pipeline). Heavy optional dependencies (semsearch, sentence-transformers) are
imported lazily, so importing this package never requires them.
"""
from .bm25 import BM25Index, BM25Tokenizer
from .common import FILTER_KEYS, UnknownFilterError, indexed_text, matches_filters
from .contextual import ContextualEnricher, JsonFileCache
from .dense import DenseRetriever
from .diversity import MMRDiversifier
from .mmr import mmr_select
from .hybrid import HybridRetriever, min_max, reciprocal_rank_fusion, weighted_score_fusion
from .parent import ParentDocumentRetriever, parent_child_index
from .pipeline import RetrievalError, RetrievalPipeline, stage_candidates
from .query import (
    ChainedTransformer,
    HyDEGenerator,
    IdentityTransformer,
    MultiQueryExpander,
    QueryDecomposer,
    QueryPlan,
    QueryRewriter,
    QueryTransformer,
)
from .rerank import CrossEncoderReranker, LexicalOverlapReranker, LLMReranker
from .settings import RetrievalSettings
from .types import Principal, Reranker, RetrievalQuery, RetrievalResult, Retriever, ScoredChunk, visible

__all__ = [
    # contract
    "Principal", "RetrievalQuery", "ScoredChunk", "RetrievalResult", "Retriever", "Reranker", "visible",
    # first stage
    "BM25Index", "BM25Tokenizer", "DenseRetriever", "ParentDocumentRetriever", "parent_child_index",
    # fusion
    "HybridRetriever", "reciprocal_rank_fusion", "weighted_score_fusion", "min_max",
    # rerank
    "LexicalOverlapReranker", "CrossEncoderReranker", "LLMReranker",
    # diversity
    "MMRDiversifier", "mmr_select",
    # query transformation
    "QueryPlan", "QueryTransformer", "IdentityTransformer", "QueryRewriter", "MultiQueryExpander",
    "QueryDecomposer", "HyDEGenerator", "ChainedTransformer",
    # enrichment and composition
    "ContextualEnricher", "JsonFileCache", "RetrievalPipeline", "RetrievalError", "stage_candidates",
    "RetrievalSettings",
    # helpers
    "FILTER_KEYS", "UnknownFilterError", "indexed_text", "matches_filters",
]
