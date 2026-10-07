# path: book/projects/ragkit/ragkit/eval/stage_isolation.py
"""Stage isolation: for every failing case, name the stage that lost the evidence.

The input is what a traced RAG request already records: the gold expectation, the principal,
the per-stage candidate ids in `RetrievalResult.trace`, the final hits, the packed chunks, the
answer with its citations, and the answer-quality scores from the run. The output is one label
per case, chosen by walking the debugging decision tree in pipeline order:

    permission leak?            -> permission              (overrides everything)
    abstention expected?        -> ok | abstention-missed
    required doc not indexed?   -> not-in-corpus
    required doc not visible?   -> permission              (over-filtering or a wrong gold label)
    not in any candidate list?  -> not-retrieved
    candidate, not after fusion -> dropped-by-fusion
    fused, not after rerank     -> dropped-by-rerank
    in final hits, not packed   -> truncated-in-packing
    packed, answer wrong/absent -> generation-ignored-evidence
    answer fine, citation wrong -> citation-error
    otherwise                   -> ok

With several required documents the earliest loss wins: if one document never left the
index and another was cut in packing, fixing packing will not fix the answer.

Trace format. The reader accepts `trace["stages"]` as an ordered list of
`{"name": ..., "kind": "candidate"|"retrieve"|"fusion"|"rerank", "chunk_ids" or "candidate_ids": [...]}`
(the layout Chapter 12's RetrievalPipeline writes; stages without ids, such as query
transforms, are skipped), or flat keys
`trace["bm25"] = [...]` / `trace["bm25_ids"] = [...]` whose kind is inferred from the name. The
flat traces of Chapter 12's single retrievers (`{"stage", "candidate_ids"}`) and of its
HybridRetriever (`{"retrievers": {name: sub_trace}, "fused_ids"}`) are read as well.
A retriever that records nothing still gets a coarse diagnosis from its final hits.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from typing import Any, Literal

from evalkit import Dataset, EvalCase, Run
from pydantic import BaseModel, Field

from ..retrieval.types import Principal, RetrievalResult
from .rag_dataset import RagExpectation, RagOutput, doc_id_of, expectation, rag_input
from .rag_metrics import leak_report


class FailureStage(str, Enum):
    OK = "ok"
    PERMISSION = "permission"
    ABSTENTION_MISSED = "abstention-missed"
    NOT_IN_CORPUS = "not-in-corpus"
    NOT_RETRIEVED = "not-retrieved"
    DROPPED_BY_FUSION = "dropped-by-fusion"
    DROPPED_BY_RERANK = "dropped-by-rerank"
    TRUNCATED_IN_PACKING = "truncated-in-packing"
    GENERATION_IGNORED_EVIDENCE = "generation-ignored-evidence"
    CITATION_ERROR = "citation-error"
    UNCHECKED = "unchecked"  # the system call failed, so nothing about this case was verified


# pipeline order, used to pick the earliest loss across several required documents
PIPELINE_ORDER = [
    FailureStage.NOT_IN_CORPUS,
    FailureStage.PERMISSION,
    FailureStage.NOT_RETRIEVED,
    FailureStage.DROPPED_BY_FUSION,
    FailureStage.DROPPED_BY_RERANK,
    FailureStage.TRUNCATED_IN_PACKING,
]

# where to look first for each label: the chapter that owns the fix
OWNER = {
    FailureStage.PERMISSION: "ACL metadata and retrieval filters (Ch 15, Ch 27)",
    FailureStage.ABSTENTION_MISSED: "grounding contract and abstention threshold (Ch 13)",
    FailureStage.NOT_IN_CORPUS: "ingestion coverage, parsing, freshness (Ch 11, Ch 15)",
    FailureStage.NOT_RETRIEVED: "chunking, query transforms, lexical/dense choice, k (Ch 11, Ch 12)",
    FailureStage.DROPPED_BY_FUSION: "fusion method and per-list depth (Ch 12)",
    FailureStage.DROPPED_BY_RERANK: "reranker model, rerank depth, final k (Ch 12)",
    FailureStage.TRUNCATED_IN_PACKING: "evidence budget, dedupe, ordering (Ch 13)",
    FailureStage.GENERATION_IGNORED_EVIDENCE: "generation prompt, model, conflict handling (Ch 13)",
    FailureStage.CITATION_ERROR: "citation mapping and validation (Ch 13)",
    FailureStage.UNCHECKED: "the system call itself: timeouts, outages, crashes (Ch 29)",
}

StageKind = Literal["candidate", "fusion", "rerank"]
_KIND_BY_NAME: dict[str, StageKind] = {
    "bm25": "candidate", "lexical": "candidate", "dense": "candidate", "vector": "candidate",
    "candidates": "candidate", "first_stage": "candidate", "multi_query": "candidate", "hyde": "candidate",
    "fusion": "fusion", "hybrid": "fusion", "rrf": "fusion", "fused": "fusion",
    "rerank": "rerank", "reranker": "rerank", "reranked": "rerank",
}


_KIND_ALIASES: dict[str, StageKind] = {
    "candidate": "candidate", "retrieve": "candidate", "first_stage": "candidate",
    "fusion": "fusion", "rerank": "rerank",
    # Chapter 12's optional MMR stage is part of the precision stage: a document it drops is
    # reported as dropped-by-rerank, whose owner is "the reranker and its depth".
    "diversify": "rerank",
}


class StageList(BaseModel):
    name: str
    kind: StageKind
    chunk_ids: list[str]


def stage_lists(result: RetrievalResult | None) -> list[StageList]:
    """Read per-stage candidate ids from a trace, tolerating both supported layouts."""
    if result is None:
        return []
    trace = result.trace or {}
    out: list[StageList] = []
    if isinstance(trace.get("stages"), list):
        for s in trace["stages"]:
            if not isinstance(s, Mapping):
                continue
            ids = s.get("chunk_ids", s.get("candidate_ids"))
            if ids is None:  # e.g. a query-transform stage: no candidates to trace
                continue
            name = str(s.get("name", "stage"))
            kind = _KIND_ALIASES.get(str(s.get("kind", "")), None) or _KIND_BY_NAME.get(
                name.split("#")[0].split(":")[0], "candidate")
            out.append(StageList(name=name, kind=kind, chunk_ids=list(ids)))
        return out
    # Chapter 12 HybridRetriever: {"retrievers": {name: sub_trace}, "fused_ids": [...]}
    if isinstance(trace.get("retrievers"), Mapping):
        for name, sub in trace["retrievers"].items():
            if isinstance(sub, Mapping) and isinstance(sub.get("candidate_ids"), list):
                out.append(StageList(name=str(name), kind="candidate", chunk_ids=list(sub["candidate_ids"])))
        if isinstance(trace.get("fused_ids"), list):
            out.append(StageList(name="fusion", kind="fusion", chunk_ids=list(trace["fused_ids"])))
        return out
    # Chapter 12 single retrievers (BM25Index, DenseRetriever, ...): {"stage": "bm25", "candidate_ids": [...]}
    if isinstance(trace.get("candidate_ids"), list):
        name = str(trace.get("stage", "candidates"))
        out.append(StageList(name=name, kind=_KIND_BY_NAME.get(name, "candidate"), chunk_ids=list(trace["candidate_ids"])))
        return out
    for key, value in trace.items():
        base = key[:-4] if key.endswith("_ids") else key
        if base in _KIND_BY_NAME and isinstance(value, list) and all(isinstance(v, str) for v in value):
            out.append(StageList(name=base, kind=_KIND_BY_NAME[base], chunk_ids=list(value)))
    return out


class DocPath(BaseModel):
    """Where one required document was seen, stage by stage (best rank, 1-based, or None)."""

    doc_id: str
    in_corpus: bool = True
    visible: bool = True
    ranks: dict[str, int | None] = Field(default_factory=dict)
    in_final: int | None = None
    packed: bool = False
    cited: bool = False
    lost_at: FailureStage | None = None


class StageDiagnosis(BaseModel):
    case_id: str
    stage: FailureStage
    detail: str = ""
    paths: list[DocPath] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


def _best_rank(chunk_ids: Sequence[str], doc: str, doc_of: Callable[[str], str]) -> int | None:
    for i, cid in enumerate(chunk_ids, start=1):
        if doc_of(cid) == doc:
            return i
    return None


def diagnose(
    case_id: str,
    exp: RagExpectation,
    principal: Principal,
    output: RagOutput,
    *,
    answer_ok: bool | None = None,
    corpus: Mapping[str, Any] | None = None,
    doc_visible: Callable[[str, Principal], bool] | None = None,
    tags: Sequence[str] = (),
) -> StageDiagnosis:
    """Classify one case. `answer_ok` is the verdict of the answer judges (None if not judged).

    `corpus` maps doc_id to anything (its presence is what matters); `doc_visible` answers
    whether the principal may see a document. Without them, not-in-corpus and the
    over-filtering branch of permission cannot be distinguished from not-retrieved.
    """
    def result(stage: FailureStage, detail: str = "", paths: list[DocPath] | None = None) -> StageDiagnosis:
        return StageDiagnosis(case_id=case_id, stage=stage, detail=detail, paths=paths or [], tags=list(tags))

    leaks = leak_report(output, exp, principal)
    if leaks.leaked:
        return result(FailureStage.PERMISSION, f"leaked: {', '.join(leaks.leaked_doc_ids)}")
    if exp.expect_abstain:
        if output.abstained:
            return result(FailureStage.OK, "correctly abstained")
        return result(FailureStage.ABSTENTION_MISSED, "answered although no permitted evidence exists")
    if not exp.required_doc_ids:
        return result(FailureStage.OK, "no required evidence to trace")

    chunk_map = output.chunk_doc_map()
    stages = stage_lists(output.retrieval)
    for s in stages:  # trace ids may reference chunks we never saw as objects
        for cid in s.chunk_ids:
            chunk_map.setdefault(cid, doc_id_of(cid))

    def doc_of(cid: str) -> str:
        return chunk_map.get(cid, doc_id_of(cid))

    final_ids = output.retrieval.chunk_ids if output.retrieval is not None else []
    packed_docs = set(output.packed_doc_ids)
    cited_docs = set(output.cited_doc_ids)
    candidates = [s for s in stages if s.kind == "candidate"]
    fusions = [s for s in stages if s.kind == "fusion"]
    reranks = [s for s in stages if s.kind == "rerank"]

    paths: list[DocPath] = []
    for doc in exp.required_doc_ids:
        p = DocPath(doc_id=doc)
        p.ranks = {s.name: _best_rank(s.chunk_ids, doc, doc_of) for s in stages}
        p.in_final = _best_rank(final_ids, doc, doc_of)
        p.packed = doc in packed_docs
        p.cited = doc in cited_docs
        if corpus is not None and doc not in corpus:
            p.in_corpus, p.lost_at = False, FailureStage.NOT_IN_CORPUS
        elif doc_visible is not None and not doc_visible(doc, principal):
            p.visible, p.lost_at = False, FailureStage.PERMISSION
        else:
            seen_candidate = any(p.ranks[s.name] is not None for s in candidates) if candidates else None
            seen_fusion = any(p.ranks[s.name] is not None for s in fusions) if fusions else None
            seen_rerank = any(p.ranks[s.name] is not None for s in reranks) if reranks else None
            if seen_candidate is False or (not stages and p.in_final is None):
                p.lost_at = FailureStage.NOT_RETRIEVED
            elif seen_fusion is False:
                p.lost_at = FailureStage.DROPPED_BY_FUSION
            elif seen_rerank is False or (reranks and p.in_final is None):
                p.lost_at = FailureStage.DROPPED_BY_RERANK
            elif p.in_final is None:
                # it survived every recorded stage but not the final cut: blame the last stage recorded
                p.lost_at = (FailureStage.DROPPED_BY_RERANK if reranks else
                             FailureStage.DROPPED_BY_FUSION if fusions else FailureStage.NOT_RETRIEVED)
            elif not p.packed:
                p.lost_at = FailureStage.TRUNCATED_IN_PACKING
        paths.append(p)

    lost = [p for p in paths if p.lost_at is not None]
    if lost:
        first = min(lost, key=lambda p: PIPELINE_ORDER.index(p.lost_at))  # type: ignore[arg-type]
        return result(first.lost_at, f"{first.doc_id} lost at {first.lost_at.value}", paths)  # type: ignore[union-attr]
    if output.abstained or answer_ok is False:
        # A required doc packed only as truncated blocks may have lost the very sentence needed:
        # blame the packing budget, not the generator (Chapter 13 marks such blocks truncated).
        truncated = set(output.metadata.get("truncated_chunk_ids", []))
        if truncated:
            starved = [p.doc_id for p in paths
                       if all(c.id in truncated for c in output.packed_chunks if c.doc_id == p.doc_id)]
            if starved:
                return result(FailureStage.TRUNCATED_IN_PACKING,
                              f"required evidence packed only as truncated blocks: {starved}", paths)
        if output.abstained:
            return result(FailureStage.GENERATION_IGNORED_EVIDENCE, "abstained although the evidence was packed", paths)
        return result(FailureStage.GENERATION_IGNORED_EVIDENCE, "evidence packed, answer judged wrong or ungrounded", paths)
    packed_ids = set(output.packed_chunk_ids)
    invalid = [c for c in output.cited_chunk_ids if c not in packed_ids]
    uncited = [p.doc_id for p in paths if not p.cited]
    if invalid or uncited:
        bits = ([f"invalid ids {invalid}"] if invalid else []) + ([f"required not cited {uncited}"] if uncited else [])
        return result(FailureStage.CITATION_ERROR, "; ".join(bits), paths)
    return result(FailureStage.OK, "", paths)


# ============================================================================ over a run
AnswerOk = Callable[[dict[str, float | None]], bool | None]


def default_answer_ok(scores: Mapping[str, float | None]) -> bool | None:
    """Answer is acceptable when the judged dimensions that were measured are perfect."""
    judged = [scores.get(m) for m in ("groundedness", "rubric_coverage") if scores.get(m) is not None]
    if not judged:
        return None
    return all(v == 1.0 for v in judged)  # type: ignore[comparison-overlap]


def diagnose_run(
    run: Run,
    dataset: Dataset,
    *,
    corpus: Mapping[str, Any] | None = None,
    doc_visible: Callable[[str, Principal], bool] | None = None,
    answer_ok: AnswerOk = default_answer_ok,  # type: ignore[assignment]
) -> list[StageDiagnosis]:
    """Diagnose every case of a run (first repeat). Target errors are reported, not guessed at."""
    out: list[StageDiagnosis] = []
    for r in run.results:
        if r.repeat:
            continue
        case: EvalCase = dataset.get(r.case_id)
        if r.error is not None or r.output is None:
            out.append(StageDiagnosis(case_id=r.case_id, stage=FailureStage.UNCHECKED, detail=f"target error: {r.error}",
                                      tags=list(case.tags)))
            continue
        out.append(
            diagnose(
                r.case_id, expectation(case), rag_input(case).principal, RagOutput.coerce(r.output),
                answer_ok=answer_ok(r.scores), corpus=corpus, doc_visible=doc_visible, tags=case.tags,
            )
        )
    return out


def stage_counts(diagnoses: Sequence[StageDiagnosis]) -> dict[str, int]:
    c = Counter(d.stage.value for d in diagnoses)
    return {s.value: c.get(s.value, 0) for s in FailureStage if c.get(s.value, 0)}


def stage_shift(baseline: Sequence[StageDiagnosis], candidate: Sequence[StageDiagnosis]) -> list[tuple[str, str, str]]:
    """Cases whose label changed between two configurations: (case_id, before, after)."""
    before = {d.case_id: d.stage.value for d in baseline}
    return sorted(
        (d.case_id, before[d.case_id], d.stage.value)
        for d in candidate if d.case_id in before and before[d.case_id] != d.stage.value
    )


__all__ = [
    "FailureStage", "PIPELINE_ORDER", "OWNER", "StageList", "stage_lists", "DocPath", "StageDiagnosis",
    "diagnose", "default_answer_ok", "diagnose_run", "stage_counts", "stage_shift",
]
