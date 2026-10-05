# path: book/projects/p3-rag-assistant/rag_assistant/domain/authority.py
"""Authority, effective date and supersession metadata, attached at ingestion.

The FAQ-over-policy miss (Chapters 10, 12, 14) is not a ranking bug that a better embedding
fixes: the FAQ really is a closer textual match for "how many days can I carry over". What
the ranker lacks is a fact about the documents themselves: the policy is authoritative and
newer, and it explicitly replaces the FAQ's time-off section. That fact belongs to the content
owners, so it lives in a reviewed rules file and is written onto every chunk at ingestion:

    authority       int, higher wins (external 0, faq 1, runbook/product/incident 2, policy 3)
    effective_date  ISO date the content takes effect (front matter, then body, then updated_at)
    supersedes      doc ids this document replaces (read by ragkit's EvidencePacker)
    superseded_by   on the older document's affected chunks only (read by AuthorityReranker)

Chunk ids do not depend on metadata, so changing a rule re-annotates chunks without
re-embedding them (the embedding cache keys on text).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from ragkit.documents import Chunk, Document

_EFFECTIVE_RE = re.compile(r"effective(?:\s+date)?\s*[:\-]?\s*\**\s*(\d{4}-\d{2}-\d{2})", re.IGNORECASE)


class LevelRule(BaseModel):
    match: dict[str, list[str]]
    level: int
    kind: str


class Supersession(BaseModel):
    newer: str
    older: str
    older_sections: list[str] = Field(default_factory=list)  # empty: the whole older document
    reason: str = ""


class AuthorityRules(BaseModel):
    levels: list[LevelRule] = Field(default_factory=list)
    default_level: int = 1
    supersessions: list[Supersession] = Field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> "AuthorityRules":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data.pop("comment", None)
        return cls.model_validate(data)

    @property
    def fingerprint(self) -> str:
        """Part of every document fingerprint: a rule change re-annotates the affected documents."""
        from ragkit.documents import short_hash

        return short_hash(self.model_dump_json())

    # ------------------------------------------------------------------ document level
    def level_for(self, doc: Document) -> tuple[int, str]:
        declared = doc.metadata.get("authority")
        if isinstance(declared, int):
            return declared, "declared"
        tags = {str(t).lower() for t in doc.metadata.get("tags") or []}
        for rule in self.levels:  # first match wins: order rules from most to least specific
            if tags & {t.lower() for t in rule.match.get("tags_any", [])}:
                return rule.level, rule.kind
            if any(doc.id.startswith(p) for p in rule.match.get("id_prefix", [])):
                return rule.level, rule.kind
        return self.default_level, "default"

    def annotate_document(self, doc: Document) -> Document:
        level, kind = self.level_for(doc)
        supersedes = sorted({*_as_list(doc.metadata.get("supersedes")),
                             *(s.older for s in self.supersessions if s.newer == doc.id)})
        meta: dict[str, Any] = {
            **doc.metadata,
            "authority": level,
            "authority_kind": kind,
            "effective_date": effective_date(doc),
            "supersedes": supersedes,
        }
        return doc.model_copy(update={"metadata": meta})

    # ------------------------------------------------------------------ chunk level
    def annotate_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Mark the older document's affected chunks as superseded_by the newer document."""
        out: list[Chunk] = []
        for c in chunks:
            by = [s.newer for s in self.supersessions if s.older == c.doc_id and _section_matches(c, s)]
            if by:
                c = c.model_copy(update={"metadata": {**c.metadata, "superseded_by": sorted(set(by))}})
            out.append(c)
        return out


def effective_date(doc: Document) -> str | None:
    declared = doc.metadata.get("effective_date")
    if declared:
        return str(declared)
    m = _EFFECTIVE_RE.search(doc.text[:2000])  # stated near the top in Northwind policies
    if m:
        return m.group(1)
    return doc.updated_at


def _section_matches(chunk: Chunk, rule: Supersession) -> bool:
    if not rule.older_sections:
        return True
    path = " / ".join(chunk.section_path).lower()
    return any(s.lower() in path for s in rule.older_sections)


def _as_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return [str(value)]


__all__ = ["AuthorityRules", "LevelRule", "Supersession", "effective_date"]
