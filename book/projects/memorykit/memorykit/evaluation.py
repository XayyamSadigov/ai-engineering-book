# path: book/projects/memorykit/memorykit/evaluation.py
"""Memory evaluation: two small harnesses that cover the two halves of a memory system.

- Read side: does recall return the right memories, and never the forbidden ones
  (stale, superseded, deleted, other user, other tenant)?
- Write side: does the policy store what it should and refuse what it must? A false accept
  on a poisoning case is a security defect; a false reject is a usefulness defect. They are
  reported separately because they are fixed by different people.

Chapter 24 owns the general evaluation toolkit; these are memory-specific metrics.
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from pydantic import BaseModel, Field

from .models import MemoryRecord, Owner
from .policy import Decision, WritePolicy
from .semantic import SemanticMemory


class RecallCase(BaseModel):
    name: str
    owner: Owner
    query: str
    expected_ids: set[str]
    forbidden_ids: set[str] = Field(default_factory=set)


class RecallCaseResult(BaseModel):
    name: str
    returned: list[str]
    hit: bool
    reciprocal_rank: float
    forbidden_returned: list[str]


class RecallReport(BaseModel):
    n: int
    hit_rate: float
    mrr: float
    forbidden_rate: float  # fraction of cases that returned at least one forbidden memory; target 0
    cases: list[RecallCaseResult]


def evaluate_recall(memory: SemanticMemory, cases: Sequence[RecallCase], *, k: int = 5) -> RecallReport:
    results: list[RecallCaseResult] = []
    for c in cases:
        returned = [m.record.id for m in memory.recall(c.owner, c.query, k=k).memories]
        rr = 0.0
        for rank, rid in enumerate(returned, start=1):
            if rid in c.expected_ids:
                rr = 1.0 / rank
                break
        results.append(RecallCaseResult(
            name=c.name, returned=returned, hit=rr > 0, reciprocal_rank=rr,
            forbidden_returned=[r for r in returned if r in c.forbidden_ids],
        ))
    n = len(results) or 1
    return RecallReport(
        n=len(results),
        hit_rate=sum(r.hit for r in results) / n,
        mrr=sum(r.reciprocal_rank for r in results) / n,
        forbidden_rate=sum(bool(r.forbidden_returned) for r in results) / n,
        cases=results,
    )


class WriteCase(BaseModel):
    name: str
    record: MemoryRecord
    should_store: bool  # True: accept or pending is correct; False: reject is correct


class WritePolicyReport(BaseModel):
    n: int
    false_accepts: list[str]  # stored something it must refuse (poisoning, secrets, PII)
    false_rejects: list[str]  # refused something legitimate
    accuracy: float


def evaluate_write_policy(policy: WritePolicy, cases: Sequence[WriteCase], *, now: datetime) -> WritePolicyReport:
    fa: list[str] = []
    fr: list[str] = []
    for c in cases:
        stored = policy.evaluate(c.record, now=now).decision != Decision.REJECT
        if stored and not c.should_store:
            fa.append(c.name)
        elif not stored and c.should_store:
            fr.append(c.name)
    n = len(cases) or 1
    return WritePolicyReport(n=len(cases), false_accepts=fa, false_rejects=fr, accuracy=1 - (len(fa) + len(fr)) / n)


__all__ = [
    "RecallCase",
    "RecallCaseResult",
    "RecallReport",
    "evaluate_recall",
    "WriteCase",
    "WritePolicyReport",
    "evaluate_write_policy",
]
