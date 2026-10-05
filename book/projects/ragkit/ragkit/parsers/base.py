# path: book/projects/ragkit/ragkit/parsers/base.py
"""Parser protocol, per-source defaults, and the builder that turns structure into a Document."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ..documents import Block, BlockKind, Document, PageInfo, SourceType, assemble, doc_id_from_uri, sha256_text


class ParseError(ValueError):
    """The source could not be parsed. Carries the source URI so ingestion reports can name it."""

    def __init__(self, message: str, *, source_uri: str = "") -> None:
        super().__init__(f"{source_uri}: {message}" if source_uri else message)
        self.source_uri = source_uri


class DocDefaults(BaseModel):
    """Values applied when the source itself does not declare them (front matter wins)."""

    id: str | None = None
    version: str | None = None
    title: str | None = None
    tenant: str | None = None
    acl_groups: list[str] | None = None
    updated_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class Parser(Protocol):
    source_type: SourceType
    version: str

    def parse(self, raw: bytes | str, *, source_uri: str, defaults: DocDefaults | None = None) -> list[Document]: ...


def decode(raw: bytes | str) -> str:
    if isinstance(raw, str):
        return raw
    return raw.decode("utf-8-sig", errors="replace")


def render_table(header: list[str], rows: list[list[str]]) -> str:
    """Canonical pipe table. Padding is dropped: alignment spaces cost tokens and carry no meaning."""

    def line(cells: list[str]) -> str:
        return "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"

    width = max([len(header), *(len(r) for r in rows)] or [0])
    header = header + [""] * (width - len(header))
    out = [line(header), "|" + "|".join(["---"] * width) + "|"]
    out.extend(line(r + [""] * (width - len(r))) for r in rows)
    return "\n".join(out)


class _Section:
    __slots__ = ("level", "title")

    def __init__(self, level: int, title: str) -> None:
        self.level = level
        self.title = title


class DocumentBuilder:
    """Accumulates blocks while tracking the heading stack, then assembles a Document."""

    def __init__(self) -> None:
        self.blocks: list[Block] = []
        self._stack: list[_Section] = []
        self.page: int | None = None
        self.first_heading: str | None = None

    @property
    def section_path(self) -> list[str]:
        return [s.title for s in self._stack]

    def _add(self, kind: BlockKind, text: str, **kw: Any) -> None:
        if not text.strip():
            return
        self.blocks.append(Block(kind=kind, text=text, section_path=self.section_path, page=self.page, **kw))

    def heading(self, title: str, level: int) -> None:
        title = " ".join(title.split())
        if not title:
            return
        while self._stack and self._stack[-1].level >= level:
            self._stack.pop()
        self._stack.append(_Section(level, title))
        if self.first_heading is None:
            self.first_heading = title
        self._add("heading", f"{'#' * level} {title}", level=level, meta={"heading": title})

    def paragraph(self, text: str, **meta: Any) -> None:
        self._add("paragraph", text.strip(), meta=meta)

    def quote(self, text: str) -> None:
        self._add("quote", text.strip())

    def list_item(self, text: str, marker: str = "-") -> None:
        text = text.strip()
        if not text:
            return
        item = f"{marker} {text}"
        last = self.blocks[-1] if self.blocks else None
        if last is not None and last.kind == "list" and last.section_path == self.section_path:
            last.text = f"{last.text}\n{item}"
            last.meta["items"] = last.meta.get("items", 1) + 1
        else:
            self._add("list", item, meta={"items": 1})

    def list_block(self, text: str) -> None:
        self._add("list", text.rstrip())

    def code(self, code: str, language: str = "", **meta: Any) -> None:
        body = code.rstrip("\n")
        if not body.strip():
            return
        fence = "````" if "```" in body else "```"
        self._add("code", f"{fence}{language}\n{body}\n{fence}", meta={"language": language, **meta})

    def table(self, header: list[str], rows: list[list[str]], **meta: Any) -> None:
        if not header and not rows:
            return
        self._add("table", render_table(header, rows), meta={"header": header, "rows": rows, **meta})

    def build(
        self,
        *,
        source_uri: str,
        source_type: SourceType,
        parser: str,
        defaults: DocDefaults | None,
        declared: dict[str, Any] | None = None,
        pages: list[PageInfo] | None = None,
        fallback_title: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Document:
        """Resolve identity and permissions: declared (front matter) > defaults > derived."""
        d = defaults or DocDefaults()
        declared = declared or {}
        text, blocks = assemble(self.blocks)

        def pick(key: str) -> Any:
            value = declared.get(key)
            return value if value not in (None, "", []) else getattr(d, key)

        acl = pick("acl_groups")
        if isinstance(acl, str):
            acl = [acl]
        meta = {**d.metadata, **(metadata or {})}
        meta.update({k: v for k, v in declared.items() if k not in _IDENTITY_KEYS})
        return Document(
            id=str(pick("id") or doc_id_from_uri(source_uri)),
            version=str(pick("version") or sha256_text(text)[:12]),
            source_uri=source_uri,
            source_type=source_type,
            title=str(pick("title") or self.first_heading or fallback_title),
            tenant=pick("tenant"),
            acl_groups=[str(g) for g in (acl or [])],
            updated_at=None if pick("updated_at") is None else str(pick("updated_at")),
            metadata=meta,
            parser=parser,
            text=text,
            blocks=blocks,
            pages=pages or [],
        )


_IDENTITY_KEYS = {"id", "version", "title", "tenant", "acl_groups", "updated_at"}

__all__ = ["DocDefaults", "DocumentBuilder", "ParseError", "Parser", "decode", "render_table"]
