# path: book/projects/p3-rag-assistant/rag_assistant/ingestion/processing.py
"""Parse -> normalize -> annotate -> chunk -> enrich, using ragkit for every step.

This module only composes. Parsing and normalization are ragkit's (Chapter 11), contextual
enrichment is ragkit's ContextualEnricher (Chapter 12), authority annotation is ours
(domain/authority.py). The output is deterministic for a given input and configuration,
which is what lets the worker compare fingerprints and diff chunk ids.
"""
from __future__ import annotations

from pathlib import PurePosixPath

from ragkit.chunking import MarkdownSectionChunker
from ragkit.documents import Chunk, Document
from ragkit.normalize import normalize_document
from ragkit.parsers import ParseError, parser_for
from ragkit.retrieval import ContextualEnricher
from reliability import PermanentJobError

from ..config import AssistantSettings
from ..domain.authority import AuthorityRules
from ..domain.keys import doc_fingerprint


class InvalidDocument(PermanentJobError):
    """Retrying will not help: unparseable, no ACL, or an identity conflict. Goes to the DLQ."""


class DocumentProcessor:
    def __init__(self, settings: AssistantSettings, rules: AuthorityRules,
                 enricher: ContextualEnricher | None = None) -> None:
        self.settings = settings
        self.rules = rules
        self.enricher = enricher
        self.chunker = MarkdownSectionChunker(max_tokens=settings.chunk_max_tokens)

    def parse(self, raw: bytes, uri: str) -> Document:
        try:
            docs = parser_for(PurePosixPath(uri).name).parse(raw, source_uri=uri)
        except ParseError as exc:
            raise InvalidDocument(f"{uri}: {exc}") from exc
        if len(docs) != 1:
            raise InvalidDocument(f"{uri}: expected one document, parser produced {len(docs)}")
        doc = normalize_document(docs[0])
        if not doc.has_acl:  # fail closed: a document nobody can be authorized for is not indexed
            raise InvalidDocument(f"{uri}: document {doc.id} has no tenant or acl_groups")
        if not doc.text.strip():
            raise InvalidDocument(f"{uri}: no extractable text")
        return self.rules.annotate_document(doc)

    def fingerprint(self, doc: Document) -> str:
        return doc_fingerprint(doc, chunker=self.chunker.fingerprint(), rules=self.rules.fingerprint,
                               pipeline_version=self.settings.pipeline_version,
                               enrichment=self.enricher is not None)

    def chunk(self, doc: Document) -> list[Chunk]:
        chunks = self.rules.annotate_chunks(self.chunker.chunk(doc))
        if self.enricher is not None:
            chunks = self.enricher.enrich(chunks, [doc])
        return chunks


__all__ = ["DocumentProcessor", "InvalidDocument"]
