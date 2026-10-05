# path: book/projects/ragkit/ragkit/chunking/__init__.py
"""Chunking strategies. All share BaseChunker.finalize, so ids and metadata behave identically."""
from .base import BaseChunker, Chunker, Piece, make_chunk_id
from .fixed import FixedTokenChunker
from .parent_child import ParentChildChunker, expand_to_parents, split_roles
from .recursive import DEFAULT_SEPARATORS, RecursiveChunker
from .section import MarkdownSectionChunker
from .semantic import SemanticChunker
from .sentence import SentenceChunker
from .sentences import pack_spans, split_sentences, token_windows

__all__ = [
    "BaseChunker",
    "Chunker",
    "DEFAULT_SEPARATORS",
    "FixedTokenChunker",
    "MarkdownSectionChunker",
    "ParentChildChunker",
    "Piece",
    "RecursiveChunker",
    "SemanticChunker",
    "SentenceChunker",
    "expand_to_parents",
    "make_chunk_id",
    "pack_spans",
    "split_roles",
    "split_sentences",
    "token_windows",
]
