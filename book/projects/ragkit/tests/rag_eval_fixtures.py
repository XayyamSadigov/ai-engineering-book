# path: book/projects/ragkit/tests/rag_eval_fixtures.py
"""Tiny hand-built corpus, fake retriever and fake generator for the Chapter 14 tests."""
from __future__ import annotations

from typing import Any

from evalkit import EvalCase

from ragkit.documents import Chunk
from ragkit.eval.rag_dataset import RagExpectation, RagInput, RagOutput
from ragkit.retrieval.types import Principal, RetrievalQuery, RetrievalResult, ScoredChunk


def chunk(doc_id: str, n: int = 0, text: str = "", *, tenant: str = "shared", groups: list[str] | None = None) -> Chunk:
    return Chunk(
        id=f"{doc_id}:c{n}", doc_id=doc_id, version="1", text=text or f"{doc_id} passage {n}.",
        content_hash=f"h-{doc_id}-{n}", tenant=tenant, acl_groups=groups or ["all"], position=n,
        char_start=0, char_end=10, token_count=10, chunker="test",
    )


EMPLOYEE = Principal(user_id="u1", tenant="shared", groups=["all"])


def case(
    cid: str = "C1",
    *,
    required: list[str] | None = None,
    acceptable: list[str] | None = None,
    forbidden: list[str] | None = None,
    abstain: bool = False,
    tags: list[str] | None = None,
    rubric: list[str] | None = None,
    principal: Principal = EMPLOYEE,
    question: str = "How many PTO days carry over?",
) -> EvalCase:
    exp = RagExpectation(required_doc_ids=required or [], acceptable_doc_ids=acceptable or [],
                         forbidden_doc_ids=forbidden or [], expect_abstain=abstain)
    return EvalCase(id=cid, input=RagInput(question=question, principal=principal).model_dump(mode="json"),
                    expected=exp.model_dump(mode="json"), rubric=rubric or [], tags=tags or [])


def result(hits: list[Chunk], *, stages: list[dict[str, Any]] | None = None, principal: Principal = EMPLOYEE) -> RetrievalResult:
    q = RetrievalQuery(text="q", principal=principal, k=len(hits))
    return RetrievalResult(
        query=q,
        hits=[ScoredChunk(chunk=c, score=1.0 / (i + 1), stage="final", rank=i + 1) for i, c in enumerate(hits)],
        trace={"stages": stages} if stages is not None else {},
    )


def output(hits: list[Chunk], *, packed: int | list[Chunk] | None = None, cited: list[str] | None = None,
           answer: str = "Ten days carry over.", abstained: bool = False,
           stages: list[dict[str, Any]] | None = None, principal: Principal = EMPLOYEE) -> RagOutput:
    packed_chunks = hits[:packed] if isinstance(packed, int) else (packed if packed is not None else hits)
    return RagOutput(answer="" if abstained else answer, abstained=abstained,
                     cited_chunk_ids=cited if cited is not None else [packed_chunks[0].id] if packed_chunks and not abstained else [],
                     packed_chunks=packed_chunks, retrieval=result(hits, stages=stages, principal=principal))


class FakeRetriever:
    """Returns a fixed ranking per question, filtered by ACL, with a two-stage trace."""

    def __init__(self, chunks: list[Chunk], rankings: dict[str, list[str]], *, enforce_acl: bool = True) -> None:
        self.by_id = {c.id: c for c in chunks}
        self.rankings = rankings
        self.enforce_acl = enforce_acl

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        from ragkit.retrieval.types import visible

        ranked = [self.by_id[i] for i in self.rankings.get(query.text, [])]
        if self.enforce_acl:
            ranked = [c for c in ranked if visible(c, query.principal)]
        hits = ranked[: query.k]
        return RetrievalResult(
            query=query,
            hits=[ScoredChunk(chunk=c, score=1.0, stage="lexical", rank=i + 1) for i, c in enumerate(hits)],
            trace={"stages": [{"name": "lexical", "kind": "candidate", "chunk_ids": [c.id for c in ranked]}]},
        )
