# path: book/projects/p2-semantic-search/semsearch/ingest/loader.py
"""Load Markdown documents with YAML-style front matter and split them into paragraph chunks.

The front matter parser handles the subset the Northwind corpus uses (scalars, quoted strings,
inline lists) so the project has no YAML dependency. Chunking here is intentionally simple:
paragraphs grouped under their nearest heading up to a size budget. Chapter 11 builds real
structure-aware chunkers; this one exists so Chapter 9 can focus on the store.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

_FRONT_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


class SourceDocument(BaseModel):
    doc_id: str
    title: str
    version: str
    tenant: str
    acl_groups: tuple[str, ...]
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)
    body: str
    path: str
    content_hash: str

    @property
    def index_version(self) -> str:
        """Declared version plus a content hash: an edit without a version bump still reindexes."""
        return f"{self.version}+{self.content_hash[:8]}"


class Chunk(BaseModel):
    ordinal: int
    section: str
    text: str


def _parse_scalar(raw: str) -> Any:
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part) for part in _split_list(inner)]
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        return raw[1:-1]
    return raw


def _split_list(inner: str) -> list[str]:
    parts, buf, quote = [], [], ""
    for ch in inner:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
            buf.append(ch)
        elif ch == ",":
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


def parse_front_matter(text: str) -> tuple[dict[str, Any], str]:
    m = _FRONT_RE.match(text)
    if not m:
        return {}, text
    meta: dict[str, Any] = {}
    for line in m.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise ValueError(f"malformed front matter line: {line!r}")
        meta[key.strip()] = _parse_scalar(value)
    return meta, text[m.end():]


def load_document(path: str | Path, default_tenant: str = "shared") -> SourceDocument:
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    meta, body = parse_front_matter(raw)
    doc_id = str(meta.pop("id", path.stem))
    acl = meta.pop("acl_groups", None)
    if not acl:
        # Fail closed: a document without an ACL is not silently public.
        raise ValueError(f"{path.name}: missing acl_groups in front matter")
    tags = meta.pop("tags", []) or []
    return SourceDocument(
        doc_id=doc_id,
        title=str(meta.pop("title", doc_id)),
        version=str(meta.pop("version", "0")),
        tenant=str(meta.pop("tenant", default_tenant)),
        acl_groups=tuple(str(a) for a in (acl if isinstance(acl, list) else [acl])),
        tags=tuple(str(t) for t in tags),
        metadata={k: v for k, v in meta.items()},
        body=body,
        path=str(path),
        content_hash=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    )


def load_corpus(directory: str | Path) -> list[SourceDocument]:
    docs = [load_document(p) for p in sorted(Path(directory).glob("*.md"))]
    seen: dict[str, str] = {}
    for d in docs:
        if d.doc_id in seen:
            raise ValueError(f"duplicate doc_id {d.doc_id!r} in {seen[d.doc_id]} and {d.path}")
        seen[d.doc_id] = d.path
    return docs


def chunk_paragraphs(body: str, max_chars: int = 1200) -> list[Chunk]:
    """Group blank-line-separated paragraphs under their nearest heading, up to max_chars.

    A heading starts a new chunk. A single paragraph longer than max_chars becomes its own
    chunk rather than being cut mid-sentence (Chapter 11 handles splitting properly).
    """
    chunks: list[Chunk] = []
    section = ""
    buf: list[str] = []

    def flush() -> None:
        text = "\n\n".join(buf).strip()
        if text:
            chunks.append(Chunk(ordinal=len(chunks), section=section, text=text))
        buf.clear()

    for para in re.split(r"\n\s*\n", body):
        para = para.strip()
        if not para:
            continue
        first = para.splitlines()[0]
        hm = _HEADING_RE.match(first)
        if hm:
            flush()
            section = hm.group(2)
            rest = para[len(first):].strip()
            buf.append(first)
            if rest:
                buf.append(rest)
            continue
        if buf and sum(len(b) for b in buf) + len(para) > max_chars:
            flush()
        buf.append(para)
    flush()
    return chunks


def embedding_text(doc: SourceDocument, chunk: Chunk) -> str:
    """What gets embedded: the chunk plus the context a reader would need (title, section)."""
    header = doc.title if not chunk.section or chunk.section == doc.title else f"{doc.title} > {chunk.section}"
    return f"{header}\n\n{chunk.text}"


__all__ = [
    "SourceDocument",
    "Chunk",
    "parse_front_matter",
    "load_document",
    "load_corpus",
    "chunk_paragraphs",
    "embedding_text",
]
