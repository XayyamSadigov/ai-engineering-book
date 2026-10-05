# path: book/projects/ragkit/ragkit/retrieval/parent.py
"""Parent-document retrieval: search small children, return their larger parents.

Chapter 11's ParentChildChunker emits both roles. Index only the children (precise matches);
at query time map each child hit to its parent section (complete context) with
`expand_to_parents`, keeping the best child's rank and score. Because several children of one
parent collapse into one hit, the child retriever is asked for `fanout * k` hits so that k
distinct parents usually survive.
"""
from __future__ import annotations

from typing import Iterable, Mapping

from ..chunking import ParentChildChunker, expand_to_parents, split_roles
from ..documents import Chunk, Document
from .common import Stopwatch
from .types import RetrievalQuery, RetrievalResult, Retriever, ScoredChunk, visible


def parent_child_index(docs: Iterable[Document], chunker: ParentChildChunker | None = None) -> tuple[list[Chunk], dict[str, Chunk]]:
    """Chunk documents into (children to index, parents by id)."""
    chunker = chunker or ParentChildChunker()
    chunks: list[Chunk] = []
    for d in docs:
        chunks.extend(chunker.chunk(d))
    parents, children = split_roles(chunks)
    return children, {p.id: p for p in parents}


class ParentDocumentRetriever:
    """Wrap any child-level Retriever; return parent chunks with child evidence in signals."""

    name = "parent"

    def __init__(self, child_retriever: Retriever, parents: Mapping[str, Chunk], *, fanout: int = 3) -> None:
        if fanout < 1:
            raise ValueError("fanout must be >= 1")
        self.child_retriever = child_retriever
        self.parents = dict(parents)
        self.fanout = fanout

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        with Stopwatch() as sw:
            child_query = query.model_copy(update={"k": query.k * self.fanout})
            child_result = self.child_retriever.retrieve(child_query)
            children = child_result.hits
            by_parent: dict[str, list[ScoredChunk]] = {}  # same mapping rule as expand_to_parents
            for hit in children:
                target = self.parents.get(hit.chunk.parent_id or "", hit.chunk)
                by_parent.setdefault(target.id, []).append(hit)
            hits: list[ScoredChunk] = []
            dropped = 0
            for parent in expand_to_parents([h.chunk for h in children], self.parents):
                if not visible(parent, query.principal):  # parents inherit the doc ACL; check anyway
                    dropped += 1
                    continue
                kids = by_parent[parent.id]  # an orphan child (parent unknown) maps to itself
                best = kids[0]
                hits.append(ScoredChunk(
                    chunk=parent,
                    score=best.score,
                    stage=self.name,
                    rank=len(hits) + 1,
                    signals={**best.signals, "child_rank": float(best.rank), "child_hits": float(len(kids))},
                ))
                if len(hits) == query.k:
                    break
        trace = {
            "stage": self.name,
            "k": query.k,
            "child_k": child_query.k,
            "child_ids": [h.chunk.id for h in children],
            "candidate_ids": [h.chunk.id for h in hits],
            "acl_dropped": dropped,
            "child_trace": child_result.trace,
            "latency_ms": sw.ms,
        }
        return RetrievalResult(query=query, hits=hits, trace=trace)


__all__ = ["ParentDocumentRetriever", "parent_child_index"]
