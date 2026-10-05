# path: book/projects/ragkit/ragkit/parsers/html.py
"""HTML parser on the standard library's html.parser.

It keeps what a reader of the rendered page would treat as content (headings, paragraphs, lists,
preformatted code, tables) and drops chrome (navigation, footers, scripts, forms). It also drops
content a human never sees: elements marked `hidden`, `aria-hidden="true"`, or styled
`display:none`. Hidden text is a classic carrier for indirect prompt injection (Chapter 26).
"""
from __future__ import annotations

import re
from html.parser import HTMLParser as _StdHTMLParser
from pathlib import PurePosixPath

from ..documents import Document, SourceType
from .base import DocDefaults, DocumentBuilder, decode

DEFAULT_DROP_TAGS = frozenset(
    {"script", "style", "noscript", "template", "svg", "nav", "footer", "aside", "form", "button", "iframe", "header"}
)
_VOID = frozenset({"br", "hr", "img", "input", "meta", "link", "source", "wbr", "area", "base", "col", "embed"})
_BLOCK = frozenset({"p", "div", "section", "article", "main", "body", "dd", "dt", "figcaption", "blockquote"})
_HIDDEN_STYLE = re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden", re.I)


def _collapse(text: str) -> str:
    return " ".join(text.split())


class _Extractor(_StdHTMLParser):
    def __init__(self, builder: DocumentBuilder, drop_tags: frozenset[str]) -> None:
        super().__init__(convert_charrefs=True)
        self.b = builder
        self.drop_tags = drop_tags
        self.stack: list[str] = []
        self.skip_depth = 0  # >0 while inside a dropped or hidden element
        self.buf: list[str] = []
        self.title_parts: list[str] = []
        self.in_title = False
        self.heading_level: int | None = None
        self.in_pre = False
        self.pre_lang = ""
        self.list_depth = 0
        self.in_li = False
        self.table: list[list[str]] | None = None
        self.table_header_seen = False
        self.row: list[str] | None = None
        self.cell: list[str] | None = None
        self.meta: dict[str, str] = {}
        self.hidden_dropped = 0
        self.comments_dropped = 0

    # -- helpers
    def _flush_paragraph(self) -> None:
        text = _collapse("".join(self.buf))
        self.buf.clear()
        if not text:
            return
        if self.in_li:
            self.b.list_item(text)
        else:
            self.b.paragraph(text)

    def _is_hidden(self, attrs: dict[str, str | None]) -> bool:
        if "hidden" in attrs:
            return True
        if (attrs.get("aria-hidden") or "").lower() == "true":
            return True
        return bool(_HIDDEN_STYLE.search(attrs.get("style") or ""))

    # -- parser callbacks
    def handle_starttag(self, tag: str, attrs_list: list[tuple[str, str | None]]) -> None:
        attrs = dict(attrs_list)
        if tag == "meta" and attrs.get("name") and attrs.get("content"):
            self.meta[str(attrs["name"]).lower()] = str(attrs["content"])
        if tag == "html" and attrs.get("lang"):
            self.meta["lang"] = str(attrs["lang"])
        if tag in _VOID:
            if tag == "br" and not self.skip_depth:
                (self.cell if self.cell is not None else self.buf).append("\n" if self.in_pre else " ")
            return
        self.stack.append(tag)
        if self.skip_depth:
            self.skip_depth += 1
            return
        if tag in self.drop_tags or self._is_hidden(attrs):
            if tag not in self.drop_tags:
                self.hidden_dropped += 1
            self.skip_depth = 1
            return
        if tag == "title":
            self.in_title = True
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6") and self.table is None:
            self._flush_paragraph()
            self.heading_level = int(tag[1])
        elif tag == "pre":
            self._flush_paragraph()
            self.in_pre = True
            self.pre_lang = ""
        elif tag == "code" and self.in_pre:
            cls = attrs.get("class") or ""
            m = re.search(r"language-([\w+#-]+)", cls)
            if m:
                self.pre_lang = m.group(1)
        elif tag in ("ul", "ol"):
            self._flush_paragraph()
            self.list_depth += 1
        elif tag == "li":
            self._flush_paragraph()
            self.in_li = True
        elif tag == "table":
            self._flush_paragraph()
            if self.table is None:
                self.table = []
                self.table_header_seen = False
        elif tag == "tr" and self.table is not None:
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = []
            if tag == "th" and not self.table:
                self.table_header_seen = True
        elif tag in _BLOCK and self.table is None and not self.in_pre:
            self._flush_paragraph()

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID:
            return
        # pop to the matching tag; tolerate unclosed children (real-world HTML)
        if tag not in self.stack:
            return
        while self.stack:
            top = self.stack.pop()
            if self.skip_depth:
                self.skip_depth -= 1
                if top == tag:
                    return
                continue
            self._close(top)
            if top == tag:
                return

    def _close(self, tag: str) -> None:
        if tag == "title":
            self.in_title = False
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6") and self.heading_level is not None:
            self.b.heading(_collapse("".join(self.buf)), self.heading_level)
            self.buf.clear()
            self.heading_level = None
        elif tag == "pre":
            self.b.code("".join(self.buf).strip("\n"), self.pre_lang)
            self.buf.clear()
            self.in_pre = False
        elif tag == "li":
            self._flush_paragraph()
            self.in_li = False
        elif tag in ("ul", "ol"):
            self._flush_paragraph()
            self.list_depth = max(0, self.list_depth - 1)
        elif tag in ("td", "th") and self.cell is not None and self.row is not None:
            self.row.append(_collapse("".join(self.cell)))
            self.cell = None
        elif tag == "tr" and self.row is not None and self.table is not None:
            if any(c for c in self.row):
                self.table.append(self.row)
            self.row = None
        elif tag == "table" and self.table is not None:
            rows = self.table
            self.table = None
            if rows:
                header, body = rows[0], rows[1:]
                self.b.table(header, body, **({} if self.table_header_seen else {"header_inferred": True}))
        elif tag in _BLOCK and self.table is None and not self.in_pre:
            self._flush_paragraph()

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return
        if self.in_title:
            self.title_parts.append(data)
        elif self.cell is not None:
            self.cell.append(data)
        elif self.table is not None:
            return  # stray text between table tags
        else:
            self.buf.append(data)

    def handle_comment(self, data: str) -> None:
        self.comments_dropped += 1

    def close(self) -> None:
        super().close()
        while self.stack:
            self.handle_endtag(self.stack[-1])
        self._flush_paragraph()


class HtmlParser:
    source_type = SourceType.HTML
    version = "html/1"

    def __init__(self, *, drop_tags: frozenset[str] = DEFAULT_DROP_TAGS) -> None:
        self.drop_tags = drop_tags

    def parse(self, raw: bytes | str, *, source_uri: str, defaults: DocDefaults | None = None) -> list[Document]:
        builder = DocumentBuilder()
        ex = _Extractor(builder, self.drop_tags)
        ex.feed(decode(raw))
        ex.close()
        title = _collapse("".join(ex.title_parts))
        meta: dict[str, object] = {f"html_{k}": v for k, v in ex.meta.items() if k in ("description", "keywords", "lang", "author")}
        if ex.hidden_dropped:
            meta["hidden_elements_removed"] = ex.hidden_dropped
        if ex.comments_dropped:
            meta["html_comments_removed"] = ex.comments_dropped
        declared = {"title": title} if title else {}
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


__all__ = ["HtmlParser", "DEFAULT_DROP_TAGS"]
