# path: book/capstone/northwind-assist/northwind_assist/rag/knowledge.py
"""The knowledge base: ingestion (Ch 11, 15), lexical and dense indexes (Ch 9, 12), versions (Ch 32).

Two backends behind one interface (`pipeline`, `get_chunks`, `index_version`, `doc_visible`,
`delete`, `reindex`, `fingerprint`):

* `local` (default; tests, the eval gate, a single-process demo). ragkit loads the documents
  (refusing any without tenant and ACL metadata), Project 3's AuthorityRules annotate authority,
  effective dates and supersession, and a BM25 index plus a dense index hold the chunks. Both
  indexes filter on tenant and groups *inside* the search. The reranker is Project 3's
  AuthorityReranker around ragkit's lexical reranker, so the current PTO policy outranks the
  stale FAQ section it supersedes (the gold set's conflicting-versions slice).
* `p3` (Docker Compose, production shape). Project 3's container owns the registry, the job
  queue, the worker, blue/green index versions, tombstones and pgvector; the capstone asks it for
  a retrieval pipeline per principal (`rag_assistant.retrieval.wiring.build_pipeline`). The
  capstone keeps the request path: guards, streaming, caches, routing, tracing.

`enforce_acl=False` (local backend only) reproduces a real ingestion bug, ACL metadata dropped on
the way into the index. It exists for the release gate's negative test; production settings
refuse it.
"""
from __future__ import annotations

import hashlib
import threading
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

from aie_core.embeddings import EmbeddingClient, FakeEmbeddings
from aie_core.settings import Settings as CoreSettings
from aie_core.settings import make_embedding_client
from caching import EmbeddingCache, TTLCache  # type: ignore[import-not-found]
from rag_assistant.domain.authority import AuthorityRules
from rag_assistant.retrieval.wiring import AuthorityReranker
from ragkit import Chunk, Document, MarkdownSectionChunker, chunk_documents, diff_chunks, load_documents
from ragkit.retrieval import BM25Index, DenseRetriever, LexicalOverlapReranker, Principal, RetrievalPipeline

from .. import _paths
from ..config import Settings
from ..llm.demo import content_words

CHUNKER_VERSION = "section-v1"
AUTHORITY_RULES = _paths.PROJECTS / "p3-rag-assistant" / "rag_assistant" / "data" / "authority_rules.json"


class Knowledge(Protocol):
    settings: Settings
    index_version: str

    def pipeline(self, *, rerank: bool, principal: Principal) -> RetrievalPipeline: ...
    def get_chunks(self, chunk_ids: list[str], principal: Principal) -> list[Chunk]: ...
    def doc_visible(self, doc_id: str, tenant: str, groups: Iterable[str]) -> bool: ...
    def doc_tenant(self, doc_id: str) -> str | None: ...
    def delete(self, doc_id: str) -> int: ...
    def reindex(self) -> str: ...
    def fingerprint(self) -> dict[str, str]: ...
    def chunk_count(self) -> int: ...


def corpus_vocabulary(docs: Iterable[Document], size: int = 768) -> list[str]:
    """Offline embedding space: bag of words over the corpus's most frequent content words."""
    df: Counter[str] = Counter()
    for d in docs:
        df.update(content_words(f"{d.title} {d.text}"))
    return sorted(w for w, _ in df.most_common(size))


def default_embeddings(docs: list[Document]) -> EmbeddingClient:
    core = CoreSettings()
    if core.embedding_provider == "fake":
        return FakeEmbeddings(vocabulary=corpus_vocabulary(docs), model="fake-embedding-vocab")
    return make_embedding_client(core)


# ============================================================================ local backend
class KnowledgeBase:
    backend = "local"

    def __init__(self, settings: Settings, *, docs_dirs: Iterable[str | Path] | None = None,
                 embeddings: EmbeddingClient | None = None, tracer: Any = None) -> None:
        self.settings = settings
        dirs: list[str | Path] = list(docs_dirs) if docs_dirs is not None else [_paths.DOCS_DIR]
        if settings.extra_docs_dir:
            dirs.append(settings.extra_docs_dir)
        self.docs_dirs = [Path(d) for d in dirs]
        self.tracer = tracer
        self.authority = AuthorityRules.load(AUTHORITY_RULES)
        self._lock = threading.RLock()
        self.documents: dict[str, Document] = {}
        self.chunks: dict[str, Chunk] = {}
        self._doc_chunks: dict[str, list[str]] = {}
        report = self._load(self._paths())
        self.rejected = [r.model_dump(mode="json") for r in report.rejected]
        for d in report.documents:
            self.documents[d.id] = d
        inner = embeddings or default_embeddings(list(self.documents.values()))
        # Chapter 30: vectors are cached by embedding-space fingerprint, never by model name alone.
        self.embeddings = EmbeddingCache(inner, TTLCache(max_size=50_000), model_version=getattr(inner, "model", ""))
        self.chunker = MarkdownSectionChunker(max_tokens=settings.chunk_max_tokens)
        self.bm25 = BM25Index()
        self.dense = DenseRetriever(self.embeddings, index_name=settings.index_name, index_version=CHUNKER_VERSION)
        for doc in list(self.documents.values()):
            self._index_document(doc)
        self._refresh_pipelines()

    # ------------------------------------------------------------------ ingestion
    def _paths(self) -> list[Path]:
        return sorted(p for d in self.docs_dirs for p in Path(d).glob("*.md"))

    def _load(self, paths: list[Path]) -> Any:
        inside = all(_paths.BOOK_ROOT in p.resolve().parents for p in paths)
        report = load_documents(paths, root=_paths.BOOK_ROOT if inside else None)
        report.documents = [self.authority.annotate_document(d) for d in report.documents]
        return report

    def _prepare(self, chunk: Chunk) -> Chunk:
        meta = {**chunk.metadata, "chunker_version": CHUNKER_VERSION}
        if not self.settings.rag_enforce_acl:
            # The deliberate bug: the index loses who may read the chunk.
            return chunk.model_copy(update={"tenant": "shared", "acl_groups": ["all"], "metadata": meta})
        return chunk.model_copy(update={"metadata": meta})

    def _index_document(self, doc: Document) -> dict[str, int]:
        new_chunks = [self._prepare(c) for c in self.authority.annotate_chunks(chunk_documents([doc], self.chunker))]
        diff = diff_chunks(self._doc_chunks.get(doc.id, []), new_chunks)
        # Chunk ids hash the document id and content, so a permission or metadata change keeps every id.
        # Compare the whole chunk: an ACL edit must reach the index even when no text changed.
        changed = any(self.chunks.get(c.id) != c for c in new_chunks)
        if diff.added or diff.removed or changed:
            self.bm25.replace_document(doc.id, new_chunks)
            self.dense.index(new_chunks)          # one atomic replace per document version
        for cid in diff.removed:
            self.chunks.pop(cid, None)
        for c in new_chunks:
            self.chunks[c.id] = c
        self._doc_chunks[doc.id] = [c.id for c in new_chunks]
        return {"added": len(diff.added), "unchanged": len(diff.unchanged), "removed": len(diff.removed)}

    def upsert(self, path: str | Path) -> dict[str, int]:
        """Incremental ingestion of one file: only changed chunks are re-embedded."""
        report = self._load([Path(path)])
        if not report.documents:
            raise ValueError(f"document rejected by the loader: {path}")
        with self._lock:
            doc = report.documents[0]
            self.documents[doc.id] = doc
            stats = self._index_document(doc)
            self._refresh_pipelines()
            return stats

    def delete(self, doc_id: str) -> int:
        """Remove a document from both indexes; the new index version retires cached results."""
        with self._lock:
            removed = self._doc_chunks.pop(doc_id, [])
            self.bm25.delete_document(doc_id)
            self.dense.delete_document(doc_id)
            for cid in removed:
                self.chunks.pop(cid, None)
            self.documents.pop(doc_id, None)
            self._refresh_pipelines()
            return len(removed)

    def reindex(self) -> str:
        """Re-read every source directory (runbook: reindex). Returns the new index version."""
        with self._lock:
            current = {d.id for d in self._load(self._paths()).documents}
            for doc_id in sorted(set(self.documents) - current):
                self.delete(doc_id)
            for path in self._paths():
                self.upsert(path)
            return self.index_version

    # ------------------------------------------------------------------ retrieval
    def _refresh_pipelines(self) -> None:
        # Who may read a chunk is part of the version: an ACL change must retire cached results too.
        digest = hashlib.sha256("\n".join(
            f"{cid}|{c.tenant}|{','.join(sorted(c.acl_groups))}" for cid, c in sorted(self.chunks.items())
        ).encode()).hexdigest()[:10]
        self.index_version = f"{self.settings.index_name}@{digest}"
        s = self.settings
        retrievers = {"bm25": self.bm25, "dense": self.dense}
        self.full = RetrievalPipeline(retrievers, reranker=AuthorityReranker(LexicalOverlapReranker()),
                                      candidate_k=s.candidate_k, rerank_k=s.rerank_k, final_k=s.final_k,
                                      tracer=self.tracer)
        self.reduced = RetrievalPipeline(retrievers, reranker=AuthorityReranker(None), candidate_k=s.candidate_k,
                                         rerank_k=s.rerank_k, final_k=None, tracer=self.tracer)

    def pipeline(self, *, rerank: bool, principal: Principal | None = None) -> RetrievalPipeline:
        return self.full if rerank else self.reduced

    def get_chunks(self, chunk_ids: list[str], principal: Principal) -> list[Chunk]:
        """Ids to live chunks, ACL-checked (ragkit BM25Index.get_chunks): deleted ids simply vanish."""
        return self.bm25.get_chunks(chunk_ids, principal)

    def chunk_count(self) -> int:
        return len(self.chunks)

    def doc_visible(self, doc_id: str, tenant: str, groups: Iterable[str]) -> bool:
        """The ACL truth from the *source* documents, independent of what the index holds."""
        doc = self.documents.get(doc_id)
        if doc is None:
            return False
        return doc.tenant in (None, "shared", tenant) and bool(set(doc.acl_groups) & set(groups))

    def doc_tenant(self, doc_id: str) -> str | None:
        """Owning tenant from the source document; None if the document is unknown."""
        doc = self.documents.get(doc_id)
        return None if doc is None else (doc.tenant or "shared")

    def fingerprint(self) -> dict[str, str]:
        return {"backend": self.backend, "index_version": self.index_version, "chunker": CHUNKER_VERSION,
                "embedding_space": self.embeddings.space, "embedding_model": self.embeddings.model,
                "authority_rules": self.authority.fingerprint}


# ============================================================================ Project 3 backend
class P3KnowledgeBase:
    """Project 3's ingestion tier (registry, queue, worker, index versions, tombstones, pgvector)
    as the capstone's knowledge source. In Compose the worker container runs Project 3's worker
    against the same Redis queue, registry database and vector store."""

    backend = "p3"

    def __init__(self, settings: Settings, *, embeddings: EmbeddingClient | None = None, tracer: Any = None,
                 sync: bool = True, p3_settings: Any = None) -> None:
        from rag_assistant.config import AssistantSettings  # noqa: PLC0415
        from rag_assistant.wiring import build_container  # noqa: PLC0415

        self.settings = settings
        self.p3s = p3_settings or AssistantSettings(chunk_max_tokens=settings.chunk_max_tokens,
                                                    candidate_k=settings.candidate_k, rerank_k=settings.rerank_k,
                                                    final_k=settings.final_k, index_name=settings.index_name)
        self.p3 = build_container(self.p3s, embeddings=embeddings, tracer=tracer)
        self.tracer = tracer
        if sync:
            self.p3.ingestion.sync("folder")
            self.p3.drain()
        self.p3.startup()

    @property
    def index_version(self) -> str:
        gens = self.p3.registry.generations()
        digest = hashlib.sha256(repr(sorted(gens.items())).encode()).hexdigest()[:10]
        return f"{self.p3.index_set.active_version}@{digest}"

    def pipeline(self, *, rerank: bool, principal: Principal) -> RetrievalPipeline:
        from rag_assistant.retrieval.wiring import build_pipeline  # noqa: PLC0415

        self.p3.index_set.refresh()     # pick up snapshots the worker published since the last request
        hidden = frozenset(r.doc_id for r in self.p3.registry.all("deleting"))
        pipe = build_pipeline(self.p3.index_set, principal, self.p3s, breakers=self.p3.breakers, hidden=hidden,
                              degrade_level=0 if rerank else 1, tracer=self.tracer)
        return pipe

    def get_chunks(self, chunk_ids: list[str], principal: Principal) -> list[Chunk]:
        """Project 3's IndexSet.get_chunks: ACL-checked and limited to documents active in the registry."""
        return self.p3.index_set.get_chunks(chunk_ids, principal)

    def chunk_count(self) -> int:
        return sum(len(r.chunk_ids) for r in self.p3.registry.all("active"))

    def doc_visible(self, doc_id: str, tenant: str, groups: Iterable[str]) -> bool:
        rec = self.p3.registry.get(doc_id)
        if rec is None or rec.status != "active":
            return False
        return rec.tenant in ("shared", tenant) and bool(set(rec.acl_groups) & set(groups))

    def doc_tenant(self, doc_id: str) -> str | None:
        rec = self.p3.registry.get(doc_id)
        return None if rec is None else rec.tenant

    def delete(self, doc_id: str) -> int:
        rec = self.p3.registry.get(doc_id)
        self.p3.ingestion.delete(doc_id)
        self.p3.drain()
        return len(rec.chunk_ids) if rec else 0

    def reindex(self) -> str:
        self.p3.ingestion.sync("folder")
        self.p3.drain()
        return self.index_version

    def fingerprint(self) -> dict[str, str]:
        return {"backend": self.backend, "index_version": self.index_version,
                "embedding_model": str(getattr(self.p3.embeddings, "model", "")),
                "embedding_space": str(getattr(self.p3.embeddings, "space_fingerprint", "")),
                "pipeline_version": self.p3s.pipeline_version}


def build_knowledge(settings: Settings, *, backend: str = "local", docs_dirs: Iterable[str | Path] | None = None,
                    embeddings: EmbeddingClient | None = None, tracer: Any = None) -> Any:
    if backend == "p3":
        if not settings.rag_enforce_acl:
            raise ValueError("the ACL-stripping negative test exists only for the local backend")
        return P3KnowledgeBase(settings, embeddings=embeddings, tracer=tracer, sync=settings.knowledge_sync_on_start)
    return KnowledgeBase(settings, docs_dirs=docs_dirs, embeddings=embeddings, tracer=tracer)


__all__ = ["Knowledge", "KnowledgeBase", "P3KnowledgeBase", "build_knowledge", "CHUNKER_VERSION",
           "corpus_vocabulary", "default_embeddings"]
