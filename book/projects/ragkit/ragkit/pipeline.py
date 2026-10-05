# path: book/projects/ragkit/ragkit/pipeline.py
"""Load a corpus (parse, normalize, gate on ACL, dedupe) and chunk it; diff chunk sets.

This is the batch, in-process version. Chapter 15 turns the same steps into an idempotent
indexing worker with queues, versions, and deletion propagation; `diff_chunks` is the piece of
that logic that depends only on chunk identity, so it lives here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, Field

from .chunking.base import Chunker
from .documents import Chunk, Document
from .normalize import DuplicateRecord, NormalizationConfig, dedupe_documents, normalize_document
from .parsers import EXTENSIONS, DocDefaults, ParseError, Parser, parse_file


class Rejected(BaseModel):
    source_uri: str
    reason: str


class LoadReport(BaseModel):
    documents: list[Document] = Field(default_factory=list)
    rejected: list[Rejected] = Field(default_factory=list)
    duplicates: list[DuplicateRecord] = Field(default_factory=list)
    needs_ocr: list[str] = Field(default_factory=list)  # doc ids with pages lacking a text layer

    def summary(self) -> dict[str, int]:
        return {
            "documents": len(self.documents),
            "rejected": len(self.rejected),
            "duplicates": len(self.duplicates),
            "needs_ocr": len(self.needs_ocr),
        }


def _iter_paths(sources: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(sources, (str, Path)):
        root = Path(sources)
        if root.is_dir():
            return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
        return [root]
    return [Path(s) for s in sources]


def load_documents(
    sources: str | Path | Iterable[str | Path],
    *,
    root: str | Path | None = None,
    defaults: DocDefaults | None = None,
    parsers: dict[str, Parser] | None = None,
    normalize: bool = True,
    normalization: NormalizationConfig | None = None,
    dedupe: bool = True,
    near_duplicate_threshold: float | None = None,
    require_acl: bool | None = None,
) -> LoadReport:
    """Parse every source; never let one bad file stop the batch, never admit a document without ACL.

    `root` makes source URIs relative (stable across machines). `parsers` overrides the parser per
    extension, for example a configured JsonlParser for tickets.
    """
    from .settings import RagkitSettings

    settings = RagkitSettings()
    require_acl = settings.require_acl if require_acl is None else require_acl
    threshold = settings.near_duplicate_threshold if near_duplicate_threshold is None else near_duplicate_threshold
    report = LoadReport()
    docs: list[Document] = []
    for path in _iter_paths(sources):
        uri = path.relative_to(root).as_posix() if root else path.as_posix()
        parser = (parsers or {}).get(path.suffix.lower())
        try:
            parsed = parse_file(path, parser=parser, defaults=defaults, source_uri=uri)
        except (ParseError, OSError) as exc:
            report.rejected.append(Rejected(source_uri=uri, reason=str(exc)))
            continue
        for d in parsed:
            if normalize:
                d = normalize_document(d, normalization)
            if require_acl and not d.has_acl:
                report.rejected.append(Rejected(source_uri=d.source_uri, reason="missing tenant or acl_groups"))
                continue
            if not d.text.strip() and not d.needs_ocr:
                report.rejected.append(Rejected(source_uri=d.source_uri, reason="no extractable text"))
                continue
            if d.needs_ocr:
                report.needs_ocr.append(d.id)
            docs.append(d)
    if dedupe:
        result = dedupe_documents(docs, threshold=threshold)
        report.documents, report.duplicates = result.kept, result.duplicates
    else:
        report.documents = docs
    return report


def chunk_documents(docs: Iterable[Document], chunker: Chunker) -> list[Chunk]:
    out: list[Chunk] = []
    for d in docs:
        out.extend(chunker.chunk(d))
    return out


class ChunkDiff(BaseModel):
    added: list[Chunk] = Field(default_factory=list)  # embed and index these
    unchanged: list[str] = Field(default_factory=list)  # ids already indexed; refresh metadata only
    removed: list[str] = Field(default_factory=list)  # delete from every index and cache


def diff_chunks(previous_ids: Iterable[str], new_chunks: Iterable[Chunk]) -> ChunkDiff:
    """Compare the chunk ids currently indexed for a document with a fresh chunking of it."""
    prev = set(previous_ids)
    diff = ChunkDiff()
    current: set[str] = set()
    for c in new_chunks:
        current.add(c.id)
        if c.id in prev:
            diff.unchanged.append(c.id)
        else:
            diff.added.append(c)
    diff.removed = sorted(prev - current)
    return diff


__all__ = ["ChunkDiff", "LoadReport", "Rejected", "chunk_documents", "diff_chunks", "load_documents"]
