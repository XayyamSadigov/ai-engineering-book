# path: book/projects/ragkit/ragkit/parsers/markdown.py
"""Markdown parser: YAML-subset front matter, heading tree, fenced code, pipe tables, lists.

The front matter reader handles the flat subset our corpora use (scalars, quoted strings, inline
lists). A nested mapping raises ParseError instead of being half-read: silently losing an
`acl_groups` line is a permission bug, not a formatting issue.
"""
from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any

from ..documents import Document, SourceType
from .base import DocDefaults, DocumentBuilder, ParseError, decode

_FRONT_MATTER = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)", re.DOTALL)
_KEY_VALUE = re.compile(r"^([A-Za-z_][\w-]*)\s*:\s*(.*)$")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^(\s*)(`{3,}|~{3,})\s*([\w+#.-]*)")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
_LIST_ITEM = re.compile(r"^\s*([-*+]|\d+[.)])\s+")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_FENCED_SEGMENT = re.compile(r"^(`{3,}|~{3,}).*?^\1[ \t]*$", re.DOTALL | re.MULTILINE)


def _scalar(value: str) -> Any:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_scalar(v) for v in re.findall(r'"[^"]*"|\'[^\']*\'|[^,]+', inner) if v.strip()]
    return value


def parse_front_matter(block: str, source_uri: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for n, line in enumerate(block.splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line[0].isspace():
            raise ParseError(f"front matter line {n}: nested values are not supported", source_uri=source_uri)
        m = _KEY_VALUE.match(line)
        if not m:
            raise ParseError(f"front matter line {n}: expected 'key: value'", source_uri=source_uri)
        key, value = m.group(1), m.group(2)
        if value.strip() == "":
            raise ParseError(f"front matter line {n}: empty or block value for '{key}'", source_uri=source_uri)
        out[key] = _scalar(value)
    return out


def split_front_matter(text: str, source_uri: str = "") -> tuple[dict[str, Any], str]:
    m = _FRONT_MATTER.match(text)
    if not m:
        return {}, text
    return parse_front_matter(m.group(1), source_uri), text[m.end():]


def strip_html_comments(body: str) -> tuple[str, int]:
    """Remove HTML comments outside fenced code. Comments are invisible to readers but not to indexers."""
    count = 0
    out: list[str] = []
    last = 0
    for m in _FENCED_SEGMENT.finditer(body):
        segment, n = _HTML_COMMENT.subn("", body[last:m.start()])
        out.append(segment)
        out.append(m.group(0))
        count += n
        last = m.end()
    segment, n = _HTML_COMMENT.subn("", body[last:])
    out.append(segment)
    return "".join(out), count + n


def _cells(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    return [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", s)]


class MarkdownParser:
    source_type = SourceType.MARKDOWN
    version = "markdown/1"

    def __init__(self, *, strip_comments: bool = True) -> None:
        self.strip_comments = strip_comments

    def parse(self, raw: bytes | str, *, source_uri: str, defaults: DocDefaults | None = None) -> list[Document]:
        text = decode(raw).replace("\r\n", "\n").replace("\r", "\n")
        declared, body = split_front_matter(text, source_uri)
        removed = 0
        if self.strip_comments:
            body, removed = strip_html_comments(body)
        builder = DocumentBuilder()
        self._parse_body(body, builder)
        meta = {"html_comments_removed": removed} if removed else {}
        doc = builder.build(
            source_uri=source_uri,
            source_type=self.source_type,
            parser=self.version,
            defaults=defaults,
            declared=declared,
            fallback_title=PurePosixPath(source_uri).stem,
            metadata=meta,
        )
        return [doc]

    def _parse_body(self, body: str, b: DocumentBuilder) -> None:
        lines = body.split("\n")
        i, n = 0, len(lines)
        para: list[str] = []

        def flush() -> None:
            if para:
                b.paragraph(" ".join(s.strip() for s in para))
                para.clear()

        while i < n:
            line = lines[i]
            stripped = line.strip()
            fence = _FENCE.match(line)
            if fence:
                flush()
                marker, lang = fence.group(2), fence.group(3)
                code: list[str] = []
                i += 1
                closed = False
                while i < n:
                    if lines[i].strip().startswith(marker[0] * len(marker)) and lines[i].strip().strip(marker[0]) == "":
                        closed = True
                        i += 1
                        break
                    code.append(lines[i])
                    i += 1
                b.code("\n".join(code), lang, **({} if closed else {"unclosed": True}))
                continue
            heading = _HEADING.match(line)
            if heading:
                flush()
                b.heading(heading.group(2), len(heading.group(1)))
                i += 1
                continue
            if stripped.startswith("|") and i + 1 < n and _TABLE_SEP.match(lines[i + 1]):
                flush()
                header = _cells(line)
                rows: list[list[str]] = []
                i += 2
                while i < n and lines[i].strip().startswith("|"):
                    rows.append(_cells(lines[i]))
                    i += 1
                b.table(header, rows)
                continue
            if _LIST_ITEM.match(line):
                flush()
                items: list[str] = []
                while i < n and lines[i].strip():
                    if _LIST_ITEM.match(lines[i]) or not items:
                        items.append(lines[i].rstrip())
                    else:  # continuation line of the previous item
                        items[-1] = f"{items[-1]} {lines[i].strip()}"
                    i += 1
                b.list_block("\n".join(items))
                continue
            if stripped.startswith(">"):
                flush()
                quote: list[str] = []
                while i < n and lines[i].strip().startswith(">"):
                    quote.append(lines[i].strip()[1:].strip())
                    i += 1
                b.quote(" ".join(q for q in quote if q))
                continue
            if not stripped:
                flush()
            else:
                para.append(line)
            i += 1
        flush()


__all__ = ["MarkdownParser", "parse_front_matter", "split_front_matter", "strip_html_comments"]
