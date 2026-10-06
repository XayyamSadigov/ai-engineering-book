# path: book/projects/ragkit/ragkit/generation/abstain.py
"""Abstention and escalation: decide what the user sees when the answer is weak, partial or risky.

Two decision points:

* before generation (`pre_generation`): if retrieval found nothing usable, do not call the
  model at all. Abstaining here is cheaper and safer than asking a model to abstain;
* after validation (`decide`): combine the repaired status, validator issues, conflicts,
  flagged sources and topic rules into one action: answer, answer_with_caveat, abstain, or
  escalate to a human queue.

User-facing messages never reveal that a document exists but is not visible to the user.
"I could not find this in the documents available to you" is the only safe wording; naming
a restricted title would leak it.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..retrieval.types import ScoredChunk
from .packer import PackedEvidence
from .validator import ValidationReport

Action = Literal["answer", "answer_with_caveat", "abstain", "escalate"]

ABSTAIN_MESSAGE = "I could not find an answer to this in the documents available to you."


class AbstentionPolicy(BaseModel):
    # Score floors per retrieval stage. Scores are not comparable across stages (an RRF score of
    # 0.03 can be strong, a reranker score of 0.03 is weak), so each stage needs its own floor,
    # calibrated on the gold set (Chapter 14). An empty dict disables the check.
    min_top_score: dict[str, float] = Field(default_factory=dict)
    min_blocks: int = 1
    max_dropped_ratio: float = 0.5  # abstain when the validator dropped more than this share of claims
    escalate_on_conflict: bool = False
    escalate_tags: list[str] = Field(default_factory=list)  # e.g. ["legal", "payroll-dispute"]
    escalation_queue: str = "assist-human-review"
    fallback_contact: str = "the owning team through the Beacon portal"


class Escalation(BaseModel):
    queue: str
    reason: str
    summary: str


class AbstentionDecision(BaseModel):
    action: Action
    reasons: list[str] = Field(default_factory=list)  # machine-readable, for metrics
    user_message: str | None = None  # replaces the answer text when abstaining or escalating
    notices: list[str] = Field(default_factory=list)  # caveats shown above an answer
    escalation: Escalation | None = None
    security_events: list[str] = Field(default_factory=list)  # never shown to the user


def pre_generation(hits: list[ScoredChunk], packed: PackedEvidence,
                   policy: AbstentionPolicy) -> AbstentionDecision | None:
    """Return an abstain decision when generation is pointless; None means go ahead."""
    if len(packed.blocks) < policy.min_blocks:
        return AbstentionDecision(action="abstain", reasons=["no_evidence"],
                                  user_message=f"{ABSTAIN_MESSAGE} You can ask {policy.fallback_contact}.")
    if policy.min_top_score and hits:
        best: dict[str, float] = {}
        for h in hits:
            best[h.stage] = max(best.get(h.stage, float("-inf")), h.score)
        for stage, floor in policy.min_top_score.items():
            if stage in best and best[stage] < floor:
                return AbstentionDecision(
                    action="abstain", reasons=[f"low_retrieval_score:{stage}"],
                    user_message=f"{ABSTAIN_MESSAGE} You can ask {policy.fallback_contact}.")
    return None


def decide(packed: PackedEvidence, report: ValidationReport, policy: AbstentionPolicy,
           hits: list[ScoredChunk] | None = None) -> AbstentionDecision:
    if hits is not None:
        early = pre_generation(hits, packed, policy)
        if early is not None:
            return early

    answer = report.repaired
    reasons: list[str] = []
    notices: list[str] = []
    security = [f"flagged evidence {b.eid} from {b.doc_id} v{b.version}" for b in packed.blocks if b.flagged]

    if answer.status == "insufficient_evidence":
        reasons.append("model_abstained" if report.original.status == "insufficient_evidence" else "validation_emptied")
        return AbstentionDecision(action="abstain", reasons=reasons, security_events=security,
                                  user_message=f"{ABSTAIN_MESSAGE} You can ask {policy.fallback_contact}.")

    total = len(report.original.claims)
    if total and len(report.dropped_claims) / total > policy.max_dropped_ratio:
        return AbstentionDecision(action="abstain", reasons=["most_claims_unsupported"], security_events=security,
                                  user_message=f"{ABSTAIN_MESSAGE} You can ask {policy.fallback_contact}.")

    cited = {e for c in answer.claims for e in c.citations}
    cited_tags = {t for b in packed.blocks if b.eid in cited for t in b.tags}
    risky = sorted(cited_tags & set(policy.escalate_tags))
    if risky or (policy.escalate_on_conflict and answer.status == "conflict"):
        reason = f"topic:{','.join(risky)}" if risky else "conflict"
        return AbstentionDecision(
            action="escalate", reasons=[reason], security_events=security,
            user_message="This question needs a person to confirm the answer. It has been sent for review.",
            escalation=Escalation(queue=policy.escalation_queue, reason=reason,
                                  summary=f"status={answer.status} cited={sorted(cited)}"))

    if answer.status == "conflict":
        reasons.append("conflict")
        notices.append("Sources disagree. The answer follows the most recent source; the older one is mentioned.")
    if answer.status == "partial":
        reasons.append("partial")
        if answer.missing_info:
            notices.append("Not covered by the available documents: " + "; ".join(answer.missing_info))
        else:
            notices.append("This answer is incomplete: parts could not be confirmed from the available documents.")
    if "stale_source_preferred" in report.codes():
        reasons.append("stale_source")
        notices.append("A newer document may change part of this answer.")
    if report.dropped_claims:
        reasons.append("claims_dropped")
    action: Action = "answer_with_caveat" if notices else "answer"
    return AbstentionDecision(action=action, reasons=reasons or ["ok"], notices=notices, security_events=security)


__all__ = ["ABSTAIN_MESSAGE", "AbstentionDecision", "AbstentionPolicy", "Action", "Escalation", "decide",
           "pre_generation"]
