# path: book/projects/ragkit/ragkit/tokenizers.py
"""Tokenizers that report character spans, so chunk boundaries can slice the original text.

Chunk sizes are budgets in tokens, but slicing a document by re-joining decoded tokens alters
whitespace and breaks citations. Every tokenizer here returns (start, end) character spans and
chunkers cut `Document.text` at those offsets.

`RegexTokenizer` is the deterministic default: words and punctuation marks. It undercounts
subword tokenizers on rare words and code, so production systems should measure with the
tokenizer of the embedding model they index with (`TiktokenTokenizer` is one example).
"""
from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

Span = tuple[int, int]


@runtime_checkable
class Tokenizer(Protocol):
    name: str

    def spans(self, text: str) -> list[Span]: ...

    def count(self, text: str) -> int: ...


class RegexTokenizer:
    name = "regex-v1"
    _PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def spans(self, text: str) -> list[Span]:
        return [m.span() for m in self._PATTERN.finditer(text)]

    def count(self, text: str) -> int:
        return len(self._PATTERN.findall(text))


class TiktokenTokenizer:
    """BPE spans via tiktoken. Needs the encoding file locally (cached) or network on first use."""

    def __init__(self, encoding: str = "cl100k_base") -> None:
        import tiktoken  # optional dependency

        self._enc = tiktoken.get_encoding(encoding)
        self.name = f"tiktoken-{encoding}"

    def spans(self, text: str) -> list[Span]:
        tokens = self._enc.encode(text, disallowed_special=())
        if not tokens:
            return []
        decoded, offsets = self._enc.decode_with_offsets(tokens)
        if decoded != text:  # invalid UTF-8 round trip; fall back to coarse spans
            return RegexTokenizer().spans(text)
        ends = offsets[1:] + [len(text)]
        return [(s, e) for s, e in zip(offsets, ends) if e > s]

    def count(self, text: str) -> int:
        return len(self._enc.encode(text, disallowed_special=()))


def default_tokenizer() -> Tokenizer:
    from .settings import RagkitSettings

    settings = RagkitSettings()
    if settings.tokenizer == "tiktoken":
        return TiktokenTokenizer(settings.tiktoken_encoding)
    return RegexTokenizer()


__all__ = ["Tokenizer", "RegexTokenizer", "TiktokenTokenizer", "default_tokenizer", "Span"]
