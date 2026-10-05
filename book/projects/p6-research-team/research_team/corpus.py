# path: book/projects/p6-research-team/research_team/corpus.py
"""The read-only knowledge source: shared-data policy documents split into section passages,
searchable with BM25, filtered by the caller's ACL on every call."""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from rank_bm25 import BM25Okapi

from .text import tokens

DEFAULT_SHARED_DATA = Path(__file__).resolve().parents[2] / "shared-data"


@dataclass(frozen=True)
class Passage:
    passage_id: str           # "<doc_id>#<section-slug>"
    doc_id: str
    title: str
    heading: str
    text: str
    version: str
    updated_at: date
    tenant: str
    acl_groups: tuple[str, ...]
    tags: tuple[str, ...]

    def visible_to(self, principal: dict[str, Any]) -> bool:
        groups = set(principal.get("groups", [])) | {"all"}
        tenant = principal.get("tenant")
        if tenant is not None and self.tenant not in ("shared", tenant):
            return False
        return bool(groups & set(self.acl_groups))

    @property
    def label(self) -> str:
        return f"{self.title} > {self.heading} (v{self.version}, {self.updated_at.isoformat()})"


def _slug(heading: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")
    return s[:48] or "intro"


def split_sections(doc_id: str, body: str) -> list[tuple[str, str, str]]:
    """Split a Markdown body on level-2 headings. Returns (passage_id, heading, text)."""
    parts: list[tuple[str, str, str]] = []
    heading, buf = "Overview", []
    for line in body.splitlines():
        if line.startswith("## "):
            if "".join(buf).strip():
                parts.append((heading, "\n".join(buf).strip()))
            heading, buf = line[3:].strip(), []
        elif not line.startswith("# "):
            buf.append(line)
    if "".join(buf).strip():
        parts.append((heading, "\n".join(buf).strip()))
    seen: dict[str, int] = {}
    out = []
    for h, text in parts:
        slug = _slug(re.sub(r"^\d+\.\s*", "", h))
        seen[slug] = seen.get(slug, 0) + 1
        if seen[slug] > 1:
            slug = f"{slug}-{seen[slug]}"
        out.append((f"{doc_id}#{slug}", h, text))
    return out


class Corpus:
    def __init__(self, passages: Iterable[Passage]) -> None:
        self.passages: list[Passage] = list(passages)
        self.by_id: dict[str, Passage] = {p.passage_id: p for p in self.passages}
        self._bm25 = BM25Okapi([tokens(f"{p.title} {p.heading} {' '.join(p.tags)} {p.text}") for p in self.passages])

    @classmethod
    def from_shared_data(cls, directory: str | Path | None = None) -> "Corpus":
        root = Path(directory or os.environ.get("P6_SHARED_DATA_DIR") or DEFAULT_SHARED_DATA).resolve()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from shared_data import load_docs  # type: ignore[import-not-found]

        passages = []
        for doc in load_docs(root / "docs"):
            for pid, heading, text in split_sections(doc.id, doc.body):
                passages.append(Passage(pid, doc.id, doc.title, heading, text, doc.version, doc.updated_at,
                                        doc.tenant, tuple(doc.acl_groups), tuple(doc.tags)))
        return cls(passages)

    def search(self, query: str, principal: dict[str, Any], k: int = 4) -> list[tuple[Passage, float]]:
        q = tokens(query)
        if not q:
            return []
        scores = self._bm25.get_scores(q)
        ranked = sorted(range(len(self.passages)), key=lambda i: (-scores[i], self.passages[i].passage_id))
        hits = [(self.passages[i], float(scores[i])) for i in ranked
                if scores[i] > 0 and self.passages[i].visible_to(principal)]
        return hits[:k]

    def get(self, passage_id: str, principal: dict[str, Any]) -> Passage | None:
        p = self.by_id.get(passage_id)
        return p if p is not None and p.visible_to(principal) else None

    def catalog(self, principal: dict[str, Any]) -> list[dict[str, Any]]:
        """One entry per visible document: what a planner may know without reading anything."""
        seen: dict[str, dict[str, Any]] = {}
        for p in self.passages:
            if p.doc_id not in seen and p.visible_to(principal):
                seen[p.doc_id] = {"doc_id": p.doc_id, "title": p.title, "tags": list(p.tags)}
        return list(seen.values())

    def doc_updated(self, doc_id: str) -> date | None:
        return next((p.updated_at for p in self.passages if p.doc_id == doc_id), None)


__all__ = ["Corpus", "Passage", "split_sections", "DEFAULT_SHARED_DATA"]
