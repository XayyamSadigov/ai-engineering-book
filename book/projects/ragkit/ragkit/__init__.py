# path: book/projects/ragkit/ragkit/__init__.py
"""ragkit: ingestion and chunking for the RAG chapters (11-15) and Project 3."""
from .chunking import (
    BaseChunker,
    Chunker,
    FixedTokenChunker,
    MarkdownSectionChunker,
    ParentChildChunker,
    RecursiveChunker,
    SemanticChunker,
    SentenceChunker,
    expand_to_parents,
    split_roles,
)
from .documents import Block, Chunk, Document, PageInfo, SourceType
from .normalize import NearDuplicateIndex, dedupe_documents, normalize_document
from .parsers import (
    DocDefaults,
    HtmlParser,
    JsonlParser,
    MarkdownParser,
    OcrEngine,
    OcrResult,
    ParseError,
    PdfParser,
    TextParser,
    parse_file,
)
from .pipeline import ChunkDiff, LoadReport, chunk_documents, diff_chunks, load_documents
from .tokenizers import RegexTokenizer, Tokenizer

__version__ = "0.1.0"

__all__ = [
    "BaseChunker",
    "Block",
    "Chunk",
    "ChunkDiff",
    "Chunker",
    "DocDefaults",
    "Document",
    "FixedTokenChunker",
    "HtmlParser",
    "JsonlParser",
    "LoadReport",
    "MarkdownParser",
    "MarkdownSectionChunker",
    "NearDuplicateIndex",
    "OcrEngine",
    "OcrResult",
    "PageInfo",
    "ParentChildChunker",
    "ParseError",
    "PdfParser",
    "RecursiveChunker",
    "RegexTokenizer",
    "SemanticChunker",
    "SentenceChunker",
    "SourceType",
    "TextParser",
    "Tokenizer",
    "chunk_documents",
    "dedupe_documents",
    "diff_chunks",
    "expand_to_parents",
    "load_documents",
    "normalize_document",
    "parse_file",
    "split_roles",
]
