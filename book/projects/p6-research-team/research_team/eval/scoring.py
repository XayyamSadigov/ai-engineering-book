# path: book/projects/p6-research-team/research_team/eval/scoring.py
"""Deterministic quality checks for one answer, plus a four-point rubric.

Caveat stated up front: citation validity uses the same `deterministic_support` definition as
the verification guard, so configurations with the guard pass it by construction offline. It
measures "did an unsupported claim ship", not independent groundedness. With a live model, add
an LLM groundedness judge (Chapter 24's evalkit) as an independent measurement.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..checks import deterministic_support
from ..contracts import AnswerReport
from ..corpus import Corpus
from ..render import parse_cited_lines

QUESTIONS_PATH = Path(__file__).with_name("questions.jsonl")


class Question(BaseModel):
    id: str
    kind: str                      # "cross" needs several policy areas, "control" needs one
    question: str
    required_docs: list[str]
    required_facts: list[str]
    expects_conflict: bool = False


class Score(BaseModel):
    qid: str
    kind: str
    architecture: str
    status: str
    doc_recall: float
    fact_recall: float
    citation_validity: float
    unsupported_shipped: int
    conflict_flagged: bool
    rubric: int = Field(ge=0, le=4)
    rubric_detail: dict[str, bool]
    tokens: int
    cost_usd: float
    model_calls: int
    agents: int
    wall_ms: float
    rejected_by_verifier: int
    duplicate_claims: int


def load_questions(path: Path = QUESTIONS_PATH) -> list[Question]:
    return [Question.model_validate(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def score(q: Question, report: AnswerReport, corpus: Corpus, principal: dict[str, Any]) -> Score:
    claims = parse_cited_lines(report.answer)
    valid = 0
    for c in claims:
        ok = False
        for e in c.evidence:
            p = corpus.get(e.passage_id, principal)
            if p is not None and deterministic_support(c.text, p.text)[0]:
                ok = True
                break
        valid += ok
    validity = valid / len(claims) if claims else 0.0
    cited = {e.doc_id for c in claims for e in c.evidence}
    doc_recall = len(set(q.required_docs) & cited) / len(q.required_docs)
    low = report.answer.lower()
    fact_recall = sum(f.lower() in low for f in q.required_facts) / len(q.required_facts)
    flagged = "sources disagree" in low
    detail = {
        "coverage": doc_recall == 1.0,
        "key_facts": fact_recall >= 0.75,
        "grounded": bool(claims) and valid == len(claims),
        "conflicts_handled": flagged if q.expects_conflict else not flagged,
    }
    return Score(
        qid=q.id, kind=q.kind, architecture=report.architecture, status=report.status,
        doc_recall=round(doc_recall, 3), fact_recall=round(fact_recall, 3), citation_validity=round(validity, 3),
        unsupported_shipped=len(claims) - valid, conflict_flagged=flagged, rubric=sum(detail.values()),
        rubric_detail=detail, tokens=report.usage.total_tokens, cost_usd=round(report.usage.cost_usd, 6),
        model_calls=report.usage.model_calls, agents=sum(1 for c in report.children if c.run_id),
        wall_ms=report.wall_ms, rejected_by_verifier=len(report.rejected_claims),
        duplicate_claims=report.duplicate_claims)


__all__ = ["Question", "Score", "load_questions", "score"]
