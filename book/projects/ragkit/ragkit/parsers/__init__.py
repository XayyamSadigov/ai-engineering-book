# path: book/projects/ragkit/ragkit/parsers/__init__.py
"""Format parsers and a by-extension dispatcher."""
from __future__ import annotations

from pathlib import Path

from ..documents import Document
from .base import DocDefaults, DocumentBuilder, ParseError, Parser, render_table
from .html import HtmlParser
from .jsonl import JsonlParser
from .markdown import MarkdownParser, parse_front_matter, split_front_matter
from .pdf import OcrEngine, OcrResult, PdfParser
from .text import TextParser

EXTENSIONS: dict[str, type] = {
    ".md": MarkdownParser,
    ".markdown": MarkdownParser,
    ".html": HtmlParser,
    ".htm": HtmlParser,
    ".pdf": PdfParser,
    ".txt": TextParser,
    ".jsonl": JsonlParser,
}


def parser_for(path: str | Path) -> Parser:
    suffix = Path(path).suffix.lower()
    try:
        return EXTENSIONS[suffix]()
    except KeyError:
        raise ParseError(f"no parser registered for extension {suffix!r}", source_uri=str(path)) from None


def parse_file(
    path: str | Path,
    *,
    parser: Parser | None = None,
    defaults: DocDefaults | None = None,
    source_uri: str | None = None,
) -> list[Document]:
    """Parse one file. `source_uri` should be stable across machines (a repo-relative path or URL)."""
    p = Path(path)
    chosen = parser or parser_for(p)
    return chosen.parse(p.read_bytes(), source_uri=source_uri or p.as_posix(), defaults=defaults)


__all__ = [
    "DocDefaults",
    "DocumentBuilder",
    "EXTENSIONS",
    "HtmlParser",
    "JsonlParser",
    "MarkdownParser",
    "OcrEngine",
    "OcrResult",
    "ParseError",
    "Parser",
    "PdfParser",
    "TextParser",
    "parse_file",
    "parse_front_matter",
    "parser_for",
    "render_table",
    "split_front_matter",
]
