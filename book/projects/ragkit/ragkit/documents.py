# path: book/projects/ragkit/ragkit/documents.py
"""The document model every later RAG chapter builds on.

A `Document` is one logical source (a policy file, a web page, a PDF, one JSONL record) after
parsing. It carries identity (id, version, content_hash), provenance (source_uri, parser),
permissions (tenant, acl_groups), and structure (blocks with section paths and character
offsets into `text`). A `Chunk` is the retrievable unit derived from a document. It copies the
permission fields so that retrieval can filter without a join, and it records where it came
from (doc_id, version, section_path, char offsets, pages) so that answers can be traced back
to a source and stale chunks can be found and deleted.
"""
from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field, model_validator

BlockKind = Literal["heading", "paragraph", "list", "code", "table", "quote"]
ChunkKind = Literal["text", "table", "code", "mixed"]
ChunkRole = Literal["leaf", "parent", "child"]

BLOCK_SEPARATOR = "\n\n"


class SourceType(str, Enum):
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    TEXT = "text"
    JSONL = "jsonl"


# ----------------------------------------------------------------------------- hashing
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short_hash(*parts: object, length: int = 16) -> str:
    """Stable hash of several values. Never use Python's hash(): it is salted per process."""
    joined = "\x1f".join(str(p) for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:length]


def doc_id_from_uri(source_uri: str) -> str:
    """Fallback document id when the source does not declare one."""
    return f"doc-{short_hash(source_uri)}"


# ----------------------------------------------------------------------------- structure
class Block(BaseModel):
    """One structural unit of a parsed document, rendered as it appears in `Document.text`."""

    kind: BlockKind
    text: str
    level: int | None = None  # heading level 1-6
    section_path: list[str] = Field(default_factory=list)
    page: int | None = None  # 1-based page for paginated sources
    char_start: int = 0
    char_end: int = 0
    meta: dict[str, Any] = Field(default_factory=dict)


class PageInfo(BaseModel):
    number: int  # 1-based
    char_count: int  # characters in the extracted text layer
    needs_ocr: bool  # the text layer was empty or too thin to trust
    ocr_applied: bool = False
    ocr_confidence: float | None = None


def assemble(blocks: Iterable[Block]) -> tuple[str, list[Block]]:
    """Join blocks into document text and recompute every block's character offsets."""
    out: list[Block] = []
    parts: list[str] = []
    cursor = 0
    for b in blocks:
        if not b.text:
            continue
        if parts:
            parts.append(BLOCK_SEPARATOR)
            cursor += len(BLOCK_SEPARATOR)
        parts.append(b.text)
        out.append(b.model_copy(update={"char_start": cursor, "char_end": cursor + len(b.text)}))
        cursor += len(b.text)
    return "".join(parts), out


class Document(BaseModel):
    id: str
    version: str
    source_uri: str
    source_type: SourceType
    title: str = ""
    tenant: str | None = None
    acl_groups: list[str] = Field(default_factory=list)  # empty means nobody: fail closed
    updated_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    parser: str  # name/version of the parser that produced this document
    text: str
    blocks: list[Block] = Field(default_factory=list)
    pages: list[PageInfo] = Field(default_factory=list)
    content_hash: str = ""

    @model_validator(mode="after")
    def _fill_hash(self) -> "Document":
        if not self.content_hash:
            self.content_hash = sha256_text(self.text)
        return self

    @property
    def has_acl(self) -> bool:
        return self.tenant is not None and len(self.acl_groups) > 0

    @property
    def needs_ocr(self) -> bool:
        return any(p.needs_ocr and not p.ocr_applied for p in self.pages)

    def with_blocks(self, blocks: Iterable[Block], **updates: Any) -> "Document":
        """Return a copy rebuilt from `blocks`, with fresh text, offsets, and content hash."""
        text, rebuilt = assemble(blocks)
        data = self.model_dump()
        data.update(updates)
        data.update({"text": text, "blocks": [b.model_dump() for b in rebuilt], "content_hash": ""})
        return Document.model_validate(data)

    def visible_to(self, groups: Iterable[str], tenant: str | None = None) -> bool:
        """Same rule as the shared dataset: tenant must match or be `shared`, and a group must overlap."""
        if tenant is not None and self.tenant not in ("shared", tenant):
            return False
        return bool(set(self.acl_groups) & set(groups))


class Chunk(BaseModel):
    id: str
    doc_id: str
    version: str  # version of the document this chunk was cut from
    text: str
    content_hash: str
    tenant: str | None
    acl_groups: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)
    section_path: list[str] = Field(default_factory=list)
    parent_id: str | None = None
    position: int  # order within the chunker's output for this document
    char_start: int  # span in Document.text this chunk covers
    char_end: int
    token_count: int
    kind: ChunkKind = "text"
    role: ChunkRole = "leaf"
    page_start: int | None = None
    page_end: int | None = None
    chunker: str  # fingerprint of the chunker and its configuration

    def context_header(self) -> str:
        title = str(self.metadata.get("title", "")).strip()
        crumbs = [s for s in self.section_path if s and s != title]
        parts = [p for p in [title, *crumbs] if p]
        return " > ".join(parts)

    def embedding_text(self) -> str:
        """Text to embed or index: a breadcrumb of title and section, then the chunk body.

        The breadcrumb restores context the chunk lost when it was cut out of its document.
        Keep `text` itself clean so that citations quote the source exactly.
        """
        header = self.context_header()
        return f"{header}\n\n{self.text}" if header else self.text


__all__ = [
    "Block",
    "BlockKind",
    "Chunk",
    "ChunkKind",
    "ChunkRole",
    "Document",
    "PageInfo",
    "SourceType",
    "assemble",
    "doc_id_from_uri",
    "sha256_text",
    "short_hash",
    "BLOCK_SEPARATOR",
]
