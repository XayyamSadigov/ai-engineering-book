# path: book/projects/examples/ch25/taskevals/rag.py
"""A small RAG answer evaluator over evidence: groundedness, answer relevance, citations, abstention.

Chapter 14 owns RAG evaluation in depth (retrieval metrics, stage isolation, the gold set).
This module is the compact version a task suite needs when a feature contains a RAG step:

    input    = {"question": "...", "evidence": [{"id": "pto-policy", "text": "..."}]}
    expected = {"answerable": true, "required_sources": ["pto-policy"]}      # optional
    output   = {"answer": "... [pto-policy]", "citations": ["pto-policy"]}

Groundedness here is claim-level: split the answer into sentences, call a sentence supported
when one evidence passage covers enough of its content words and contains every number and
identifier it states. It is a lexical proxy; `judge_evaluators` adds evalkit's calibrated
GROUNDEDNESS and RELEVANCE judges for the semantic version.
"""
from __future__ import annotations

from typing import Any

from aie_core import LLMClient

from evalkit import GROUNDEDNESS, RELEVANCE, EvalCase, LLMJudge, Score

from .text_support import content_words, overlap, split_sentences, unsupported_atoms

ABSTAIN_MARKERS = ("insufficient evidence", "i don't know", "i do not know", "cannot find", "not in the provided")


def is_abstention(answer: str) -> bool:
    low = answer.lower()
    return any(m in low for m in ABSTAIN_MARKERS)


def claim_support(answer: str, evidence: list[dict[str, str]], *, min_overlap: float = 0.6) -> list[dict[str, Any]]:
    """Per sentence: the best-supporting passage id, its overlap, and unsupported atoms."""
    out = []
    for claim in split_sentences(answer):
        best_id, best_ov, best_atoms = None, 0.0, unsupported_atoms(claim, "")
        for passage in evidence:
            ov = overlap(claim, passage["text"])
            atoms = unsupported_atoms(claim, passage["text"])
            if (not atoms, ov) > (not best_atoms, best_ov):
                best_id, best_ov, best_atoms = passage["id"], ov, atoms
        out.append({"claim": claim, "passage": best_id, "overlap": round(best_ov, 3), "unsupported_atoms": best_atoms,
                    "supported": best_ov >= min_overlap and not best_atoms})
    return out


def answer_relevance(question: str, answer: str) -> float:
    """Share of the question's content words the answer engages with (lexical proxy)."""
    q = content_words(question)
    return len(q & content_words(answer)) / len(q) if q else 1.0


RAG_METRICS = ["rag_groundedness", "rag_answer_relevance", "rag_citations_valid", "rag_abstention_correct"]


class RagAnswerEvaluator:
    name = "rag_answer"
    version = "1"
    metric_names = RAG_METRICS

    def __init__(self, min_relevance: float = 0.3) -> None:
        self.min_relevance = min_relevance

    def __call__(self, case: EvalCase, output: Any) -> list[Score]:
        answer = output["answer"]
        evidence = case.input.get("evidence", [])
        expected = case.expected or {}
        answerable = expected.get("answerable", True)
        abstained = is_abstention(answer)
        abstention_ok = abstained != answerable  # abstain exactly when the evidence cannot answer
        evidence_ids = {p["id"] for p in evidence}
        cited = set(output.get("citations", []))
        citations_ok = cited <= evidence_ids and (abstained or bool(cited))
        required = set(expected.get("required_sources", []))
        if required and not abstained:
            citations_ok = citations_ok and required <= cited
        scores = [
            Score(name="rag_abstention_correct", value=float(abstention_ok), passed=abstention_ok),
            Score(name="rag_citations_valid", value=float(citations_ok), passed=citations_ok,
                  detail={"cited": sorted(cited), "retrieved": sorted(evidence_ids)} if not citations_ok else None),
        ]
        if abstained:
            # Nothing was claimed: groundedness is vacuous, relevance is decided by abstention correctness.
            scores += [Score(name="rag_groundedness", value=1.0, passed=True),
                       Score(name="rag_answer_relevance", value=float(abstention_ok), passed=abstention_ok)]
            return scores
        claims = claim_support(answer, evidence)
        grounded = sum(c["supported"] for c in claims) / len(claims) if claims else 1.0
        rel = answer_relevance(case.input["question"], answer)
        scores += [
            Score(name="rag_groundedness", value=grounded, passed=grounded >= 1.0,
                  detail=[c for c in claims if not c["supported"]] or None),
            Score(name="rag_answer_relevance", value=min(1.0, rel), passed=rel >= self.min_relevance),
        ]
        return scores


def judge_evaluators(client: LLMClient, *, model: str | None = None) -> list[Any]:
    """Semantic groundedness and relevance via evalkit judges (calibrate before gating on them)."""
    evidence_fn = lambda case, out: "\n\n".join(f"[{p['id']}] {p['text']}" for p in case.input.get("evidence", []))  # noqa: E731
    return [
        LLMJudge(client, GROUNDEDNESS, model=model).as_evaluator(
            input_fn=lambda c: c.input["question"], answer_fn=lambda o: o["answer"], evidence_fn=evidence_fn),
        LLMJudge(client, RELEVANCE, model=model).as_evaluator(
            input_fn=lambda c: c.input["question"], answer_fn=lambda o: o["answer"]),
    ]


__all__ = ["is_abstention", "claim_support", "answer_relevance", "RagAnswerEvaluator", "RAG_METRICS",
           "judge_evaluators"]
