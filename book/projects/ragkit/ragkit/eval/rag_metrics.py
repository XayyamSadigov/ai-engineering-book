# path: book/projects/ragkit/ragkit/eval/rag_metrics.py
"""Deterministic RAG metrics: retrieval ranking, permission leaks, context, citations, abstention.

Everything here is pure arithmetic over ids, so it is cheap, reproducible, and never delegated
to a judge. Two granularities are used on purpose:

* Ranking metrics (hit@k, recall@k, MRR, nDCG@k) are computed over the ranked list of
  *documents*, first occurrence wins, because the gold set labels documents. Five chunks of the
  same relevant policy are one relevant document, not five.
* precision@k and context relevance are computed over *chunks*, because they measure what the
  generator has to read: three chunks of a distractor waste three slots.

Permission leaks are counted separately from quality and are never averaged into it. One leak
fails the run no matter how good recall is.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

from evalkit import EvalCase, FunctionEvaluator, Score
from pydantic import BaseModel, Field

from ..documents import Chunk
from ..retrieval.types import Principal, visible
from .rag_dataset import RagExpectation, RagOutput, expectation, rag_input


# ============================================================================ ranking metrics
def dedupe(ids: Iterable[str]) -> list[str]:
    seen: list[str] = []
    for i in ids:
        if i not in seen:
            seen.append(i)
    return seen


def hit_at_k(ranked: Sequence[str], required: Iterable[str], k: int) -> float:
    """1.0 if at least one required item is in the top k."""
    req = set(required)
    return 1.0 if any(d in req for d in ranked[:k]) else 0.0


def recall_at_k(ranked: Sequence[str], required: Iterable[str], k: int) -> float:
    """Fraction of required items found in the top k. 1.0 means the evidence is all there."""
    req = set(required)
    if not req:
        raise ValueError("recall is undefined without required items")
    return len(req & set(ranked[:k])) / len(req)


def precision_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Relevant items in the top k divided by k, not by the number returned.

    The denominator is the budget (k slots), so a retriever that returns two items for k=5
    scores at most 0.4. Use a different denominator only if you document it next to the number.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    rel = set(relevant)
    return sum(1 for d in ranked[:k] if d in rel) / k


def reciprocal_rank(ranked: Sequence[str], relevant: Iterable[str]) -> float:
    rel = set(relevant)
    for i, d in enumerate(ranked, start=1):
        if d in rel:
            return 1.0 / i
    return 0.0


def dcg(gains: Sequence[float]) -> float:
    return sum((2.0**g - 1.0) / math.log2(i + 1) for i, g in enumerate(gains, start=1))


def ndcg_at_k(ranked: Sequence[str], grades: Mapping[str, int], k: int) -> float:
    """Normalized discounted cumulative gain with graded relevance (required 2, acceptable 1).

    gain = 2^grade - 1, discounted by log2(rank + 1). The ideal ordering sorts all graded items
    by grade. Duplicates in `ranked` earn nothing after their first occurrence.
    """
    seen: set[str] = set()
    gains: list[float] = []
    for d in ranked[:k]:
        gains.append(0.0 if d in seen else float(grades.get(d, 0)))
        seen.add(d)
    ideal = sorted((float(g) for g in grades.values() if g > 0), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


# ============================================================================ permissions
class LeakReport(BaseModel):
    forbidden_retrieved: list[str] = Field(default_factory=list)  # gold-forbidden docs in the hits
    forbidden_packed: list[str] = Field(default_factory=list)
    forbidden_cited: list[str] = Field(default_factory=list)
    acl_violations: list[str] = Field(default_factory=list)  # chunk ids the principal may not see

    @property
    def leaked(self) -> bool:
        return bool(self.forbidden_retrieved or self.forbidden_packed or self.forbidden_cited or self.acl_violations)

    @property
    def leaked_doc_ids(self) -> list[str]:
        from .rag_dataset import doc_id_of

        return dedupe([*self.forbidden_retrieved, *self.forbidden_packed, *self.forbidden_cited,
                       *(doc_id_of(c) for c in self.acl_violations)])


def acl_violations(chunks: Iterable[Chunk], principal: Principal) -> list[str]:
    """Chunks the principal is not allowed to see. Independent of the gold set: a leak is a leak."""
    return dedupe(c.id for c in chunks if not visible(c, principal))


def leak_report(output: RagOutput, exp: RagExpectation, principal: Principal) -> LeakReport:
    forbidden = set(exp.forbidden_doc_ids)
    hits = [h.chunk for h in output.retrieval.hits] if output.retrieval is not None else []
    return LeakReport(
        forbidden_retrieved=[d for d in output.retrieved_doc_ids if d in forbidden],
        forbidden_packed=dedupe(d for d in output.packed_doc_ids if d in forbidden),
        forbidden_cited=[d for d in output.cited_doc_ids if d in forbidden],
        # Chapter 12's pipeline removes invisible chunks at a final check and records their ids in
        # trace["acl_violations"]. The user never saw them, but they got past the pre-filter into
        # the candidate lists, so the evaluation counts them: that list must always be empty.
        acl_violations=dedupe([*acl_violations([*hits, *output.packed_chunks], principal),
                               *_trace_violations(output)]),
    )


def _trace_violations(output: RagOutput) -> list[str]:
    if output.retrieval is None:
        return []
    ids = output.retrieval.trace.get("acl_violations") or []
    return [str(i) for i in ids] if isinstance(ids, list) else []


# ============================================================================ context and citations
def context_relevance(packed_doc_ids: Sequence[str], relevant: Iterable[str]) -> float:
    """Share of packed chunks whose document is relevant. Low values mean noise in the prompt."""
    if not packed_doc_ids:
        return 0.0
    rel = set(relevant)
    return sum(1 for d in packed_doc_ids if d in rel) / len(packed_doc_ids)


class CitationScores(BaseModel):
    precision: float  # cited docs that are relevant / cited docs
    recall: float  # required docs that were cited / required docs
    valid: bool  # every cited chunk was actually packed into the prompt
    invalid_chunk_ids: list[str] = Field(default_factory=list)


def citation_scores(output: RagOutput, exp: RagExpectation) -> CitationScores:
    cited_docs = output.cited_doc_ids
    packed = set(output.packed_chunk_ids)
    invalid = [c for c in output.cited_chunk_ids if c not in packed]
    precision = (sum(1 for d in cited_docs if d in exp.relevant_doc_ids) / len(cited_docs)) if cited_docs else 0.0
    req = set(exp.required_doc_ids)
    recall = len(req & set(cited_docs)) / len(req) if req else 1.0
    return CitationScores(precision=precision, recall=recall, valid=not invalid, invalid_chunk_ids=invalid)


AbstentionCell = Literal["answered", "correct-abstain", "false-abstain", "false-answer"]


def abstention_cell(abstained: bool, expect_abstain: bool) -> AbstentionCell:
    """The four outcomes. false-answer is the dangerous one, false-abstain the annoying one."""
    if expect_abstain:
        return "correct-abstain" if abstained else "false-answer"
    return "false-abstain" if abstained else "answered"


# ============================================================================ evaluators for evalkit
DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10)


def retrieval_scores(case: EvalCase, output: RagOutput, *, ks: Sequence[int] = DEFAULT_KS, ndcg_k: int = 10) -> list[Score]:
    exp, principal = expectation(case), rag_input(case).principal
    leaks = leak_report(output, exp, principal)
    scores = [
        Score(name="no_permission_leak", value=0.0 if leaks.leaked else 1.0, passed=not leaks.leaked,
              detail=leaks.model_dump() if leaks.leaked else None)
    ]
    if not exp.answerable:
        # Inverted cases (forbidden-doc, abstain): ranking quality is meaningless, only leaks count.
        return scores
    docs = output.retrieved_doc_ids
    chunk_docs = output.retrieved_chunk_doc_ids
    for k in ks:
        h = hit_at_k(docs, exp.required_doc_ids, k)
        r = recall_at_k(docs, exp.required_doc_ids, k)
        scores += [
            Score(name=f"hit@{k}", value=h, passed=h == 1.0),
            Score(name=f"recall@{k}", value=r, passed=r == 1.0),
            Score(name=f"precision@{k}", value=precision_at_k(chunk_docs, exp.relevant_doc_ids, k)),
        ]
    scores.append(Score(name="mrr", value=reciprocal_rank(docs, exp.required_doc_ids)))
    scores.append(Score(name=f"ndcg@{ndcg_k}", value=ndcg_at_k(docs, exp.grades(), ndcg_k)))
    packed = output.packed_doc_ids
    scores.append(Score(name="context_relevance", value=context_relevance(packed, exp.relevant_doc_ids)))
    sufficient = set(exp.required_doc_ids) <= set(packed)
    scores.append(Score(name="evidence_packed", value=1.0 if sufficient else 0.0, passed=sufficient))
    return scores


def answer_scores(case: EvalCase, output: RagOutput) -> list[Score]:
    exp = expectation(case)
    cell = abstention_cell(output.abstained, exp.expect_abstain)
    ok = cell in ("answered", "correct-abstain")
    scores = [Score(name="abstention_correct", value=1.0 if ok else 0.0, passed=ok, detail=cell)]
    if exp.answerable and not output.abstained:
        cs = citation_scores(output, exp)
        scores += [
            Score(name="citation_precision", value=cs.precision, passed=cs.precision == 1.0),
            Score(name="citation_recall", value=cs.recall, passed=cs.recall == 1.0),
            Score(name="citations_valid", value=1.0 if cs.valid else 0.0, passed=cs.valid,
                  detail=cs.invalid_chunk_ids or None),
        ]
    return scores


def retrieval_metric_names(ks: Sequence[int] = DEFAULT_KS, ndcg_k: int = 10) -> list[str]:
    names = ["no_permission_leak"]
    for k in ks:
        names += [f"hit@{k}", f"recall@{k}", f"precision@{k}"]
    return names + ["mrr", f"ndcg@{ndcg_k}", "context_relevance", "evidence_packed"]


ANSWER_METRIC_NAMES = ["abstention_correct", "citation_precision", "citation_recall", "citations_valid"]


def retrieval_evaluator(*, ks: Sequence[int] = DEFAULT_KS, ndcg_k: int = 10) -> FunctionEvaluator:
    return FunctionEvaluator(
        "retrieval",
        lambda case, out: retrieval_scores(case, RagOutput.coerce(out), ks=ks, ndcg_k=ndcg_k),
        version=f"1;ks={','.join(map(str, ks))};ndcg={ndcg_k}",
        metric_names=retrieval_metric_names(ks, ndcg_k),
    )


def answer_evaluator() -> FunctionEvaluator:
    return FunctionEvaluator(
        "answer_deterministic",
        lambda case, out: answer_scores(case, RagOutput.coerce(out)),
        version="1",
        metric_names=ANSWER_METRIC_NAMES,
    )


__all__ = [
    "dedupe", "hit_at_k", "recall_at_k", "precision_at_k", "reciprocal_rank", "dcg", "ndcg_at_k",
    "LeakReport", "acl_violations", "leak_report", "context_relevance", "CitationScores", "citation_scores",
    "AbstentionCell", "abstention_cell", "DEFAULT_KS", "retrieval_scores", "answer_scores",
    "retrieval_metric_names", "ANSWER_METRIC_NAMES", "retrieval_evaluator", "answer_evaluator",
]
