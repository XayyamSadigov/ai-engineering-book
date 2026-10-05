# path: book/projects/ragkit/ragkit/parsers/text.py
"""Plain text, with transcript detection.

Paragraphs are separated by blank lines. If most non-blank lines look like speaker turns
(`[00:12:03] Dana: ...` or `Dana: ...`), each turn becomes its own block with `speaker` and
`timestamp` in its metadata, so chunkers can keep turns whole and retrieval can filter by
speaker or time.
"""
from __future__ import annotations

import re
from pathlib import PurePosixPath

from ..documents import Document, SourceType
from .base import DocDefaults, DocumentBuilder, decode

_TURN = re.compile(r"^\s*(?:\[?(\d{1,2}:\d{2}(?::\d{2})?)\]?\s+)?([A-Z][\w.' -]{0,40}?):\s+(.+)$")


class TextParser:
    source_type = SourceType.TEXT
    version = "text/1"

    def __init__(self, *, detect_transcript: bool = True, transcript_threshold: float = 0.6) -> None:
        self.detect_transcript = detect_transcript
        self.transcript_threshold = transcript_threshold

    def parse(self, raw: bytes | str, *, source_uri: str, defaults: DocDefaults | None = None) -> list[Document]:
        text = decode(raw).replace("\r\n", "\n").replace("\r", "\n")
        builder = DocumentBuilder()
        lines = [ln for ln in text.split("\n") if ln.strip()]
        turns = [_TURN.match(ln) for ln in lines]
        is_transcript = (
            self.detect_transcript and lines and sum(1 for t in turns if t) / len(lines) >= self.transcript_threshold
        )
        if is_transcript:
            current = None
            for ln, m in zip(lines, turns):
                if m:
                    if current:
                        builder.paragraph(current[2], speaker=current[1], timestamp=current[0])
                    current = [m.group(1), m.group(2).strip(), f"{m.group(2).strip()}: {m.group(3).strip()}"]
                elif current:  # continuation of the previous turn
                    current[2] = f"{current[2]} {ln.strip()}"
                else:
                    builder.paragraph(ln)
            if current:
                builder.paragraph(current[2], speaker=current[1], timestamp=current[0])
        else:
            for block in re.split(r"\n\s*\n", text):
                joined = " ".join(s.strip() for s in block.split("\n") if s.strip())
                builder.paragraph(joined)
        doc = builder.build(
            source_uri=source_uri,
            source_type=self.source_type,
            parser=self.version,
            defaults=defaults,
            fallback_title=PurePosixPath(source_uri).stem,
            metadata={"transcript": True} if is_transcript else {},
        )
        return [doc]


__all__ = ["TextParser"]
