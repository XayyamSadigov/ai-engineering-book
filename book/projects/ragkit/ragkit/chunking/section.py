# path: book/projects/ragkit/ragkit/chunking/section.py
"""Document-aware chunking over parsed blocks: sections, code blocks, and tables as units.

Works on any Document whose parser produced blocks (Markdown, HTML, JSONL, transcripts), not
only Markdown; the name reflects where the heading model comes from.

Rules:
1. A heading starts a new section. Chunks never span two sections.
2. Within a section, consecutive blocks are packed up to `max_tokens`. The heading line is
   included at the top of the section's first chunk.
3. A code block is never split. If it alone exceeds the budget it becomes its own chunk with
   `metadata["oversize"] = True`, so you can see and decide about it.
4. A table that fits is kept whole. A larger table is split by rows, and every part repeats the
   header row (and separator) so each part is self-describing.
5. An oversized paragraph or list falls back to sentence packing inside that block.
6. A section with a heading and no body produces no chunk; its title survives in the section
   path of its subsections.
"""
from __future__ import annotations

from typing import Any

from ..documents import Block, Document
from ..parsers.base import render_table
from ..tokenizers import Tokenizer
from .base import BaseChunker, Piece
from .recursive import RecursiveChunker
from .sentences import pack_spans, split_sentences


class MarkdownSectionChunker(BaseChunker):
    name = "section"

    def __init__(
        self,
        max_tokens: int = 400,
        *,
        repeat_table_header: bool = True,
        include_headings: bool = True,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        super().__init__(tokenizer)
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.max_tokens = max_tokens
        self.repeat_table_header = repeat_table_header
        self.include_headings = include_headings

    def config(self) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "repeat_table_header": self.repeat_table_header,
            "include_headings": self.include_headings,
        }

    def split(self, doc: Document) -> list[Piece]:
        if not doc.blocks:  # unstructured input: degrade to recursive splitting, same budget
            return RecursiveChunker(self.max_tokens, tokenizer=self.tokenizer).split(doc)
        pieces: list[Piece] = []
        for section in _sections(doc.blocks):
            heading = section[0] if section[0].kind == "heading" else None
            body = section[1:] if heading else section
            if not body:
                continue
            pieces.extend(self._pack(heading, body))
        return pieces

    def _pack(self, heading: Block | None, body: list[Block]) -> list[Piece]:
        path = body[0].section_path
        out: list[Piece] = []
        pending: list[Block] = [heading] if heading is not None and self.include_headings else []
        pending_tokens = sum(self.count(b.text) for b in pending)

        def has_content() -> bool:
            return any(b.kind != "heading" for b in pending)

        def emit() -> None:
            nonlocal pending, pending_tokens
            if has_content():
                out.append(Piece(pending[0].char_start, pending[-1].char_end, section_path=path))
            pending, pending_tokens = [], 0

        for blk in body:
            t = self.count(blk.text)
            if t > self.max_tokens:
                emit()
                if blk.kind == "table":
                    out.extend(self._split_table(blk, path))
                elif blk.kind == "code":
                    out.append(Piece(blk.char_start, blk.char_end, section_path=path, kind="code", meta={"oversize": True}))
                else:
                    spans = pack_spans(blk.text, split_sentences(blk.text), self.max_tokens, self.tokenizer)
                    out.extend(Piece(blk.char_start + s, blk.char_start + e, section_path=path) for s, e in spans)
                continue
            if has_content() and pending_tokens + t > self.max_tokens:
                emit()
            pending.append(blk)
            pending_tokens += t
        emit()
        return out

    def _split_table(self, blk: Block, path: list[str]) -> list[Piece]:
        header: list[str] = blk.meta.get("header", [])
        rows: list[list[str]] = blk.meta.get("rows", [])
        header_text = render_table(header, [])
        budget = max(1, self.max_tokens - self.count(header_text))
        lines = blk.text.split("\n")  # header, separator, then one line per row
        offsets = []
        cursor = 0
        for line in lines:
            offsets.append((cursor, cursor + len(line)))
            cursor += len(line) + 1

        groups: list[list[int]] = []
        current: list[int] = []
        used = 0
        for r, row in enumerate(rows):
            t = self.count(render_table(header, [row]).split("\n", 2)[2])
            if current and used + t > budget:
                groups.append(current)
                current, used = [], 0
            current.append(r)
            used += t
        if current:
            groups.append(current)

        pieces = []
        for k, group in enumerate(groups):
            first_line, last_line = group[0] + 2, group[-1] + 2
            start = blk.char_start + (0 if k == 0 else offsets[first_line][0])
            end = blk.char_start + offsets[last_line][1]
            selected = [rows[r] for r in group]
            if k == 0 or self.repeat_table_header:
                text = render_table(header, selected)
            else:
                text = "\n".join(lines[first_line : last_line + 1])
            pieces.append(
                Piece(
                    start,
                    end,
                    text=text,
                    kind="table",
                    section_path=path,
                    meta={
                        "table_part": k + 1,
                        "table_parts": len(groups),
                        "table_rows": [group[0], group[-1]],
                        "header_repeated": k > 0 and self.repeat_table_header,
                    },
                )
            )
        return pieces


def _sections(blocks: list[Block]) -> list[list[Block]]:
    sections: list[list[Block]] = []
    for b in blocks:
        if b.kind == "heading" or not sections:
            sections.append([b])
        else:
            sections[-1].append(b)
    return sections


__all__ = ["MarkdownSectionChunker"]
