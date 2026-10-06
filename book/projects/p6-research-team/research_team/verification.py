# path: book/projects/p6-research-team/research_team/verification.py
"""Independent verification of claims: a verifier agent plus a deterministic guard.

A claim is accepted only when both agree it is supported. The verifier agent judges meaning
(dropped conditions, changed scope); the deterministic guard catches changed numbers and
invented content even if the verifier is wrong or was manipulated. If the verifier cannot run
(spawn refused, budget, failure), verification degrades to the guard alone and says so.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .checks import deterministic_support
from .contracts import Claim, RejectedClaim, ResultEnvelope, Role, TaskEnvelope, TaskStatus, VerificationReport
from .corpus import Corpus
from .roles import AgentFactory, skipped


@dataclass
class VerificationOutcome:
    accepted: list[Claim] = field(default_factory=list)
    rejected: list[RejectedClaim] = field(default_factory=list)
    envelope: ResultEnvelope | None = None
    degraded: bool = False


def guard(claim: Claim, corpus: Corpus, principal: dict[str, Any], min_overlap: float) -> tuple[bool, str]:
    """Every citation must resolve to a passage this principal can read, under the document it
    names, and support the claim. One good citation cannot carry a fake or irrelevant one."""
    if not claim.evidence:
        return False, "no evidence cited"
    whys = []
    for ev in claim.evidence:
        p = corpus.get(ev.passage_id, principal)
        if p is None:
            return False, f"{ev.passage_id}: unknown passage"
        if ev.doc_id != p.doc_id:
            return False, f"{ev.passage_id}: belongs to {p.doc_id}, not {ev.doc_id}"
        ok, why = deterministic_support(claim.text, p.text, min_overlap=min_overlap)
        if not ok:
            return False, f"{ev.passage_id}: {why}"
        whys.append(why)
    return True, "; ".join(whys)


def verify_claims(
    claims: list[Claim],
    *,
    factory: AgentFactory,
    corpus: Corpus,
    make_envelope: Callable[[dict[str, Any]], TaskEnvelope],
    admit: Callable[[TaskEnvelope], tuple[bool, str, TaskEnvelope]],
    settle: Callable[[ResultEnvelope], None],
    principal: dict[str, Any],
    min_overlap: float = 0.6,
) -> VerificationOutcome:
    out = VerificationOutcome()
    if not claims:
        return out
    inputs = {"claims": [{"claim_id": c.claim_id, "text": c.text,
                          "evidence": [{"doc_id": e.doc_id, "passage_id": e.passage_id} for e in c.evidence]}
                         for c in claims]}          # quotes withheld: the verifier must read the source itself
    env = make_envelope(inputs)
    ok, reason, env = admit(env)
    verdicts: dict[str, tuple[bool, str]] = {}
    if not ok:
        out.envelope = skipped(env, reason)
        out.degraded = True
    else:
        out.envelope, _ = factory.run(env)
        settle(out.envelope)
        if out.envelope.status is TaskStatus.SUCCEEDED and out.envelope.output is not None:
            report = VerificationReport.model_validate(out.envelope.output)
            verdicts = {v.claim_id: (v.supported, v.reason) for v in report.verdicts}
        else:
            out.degraded = True
    for c in claims:
        g_ok, g_why = guard(c, corpus, principal, min_overlap)
        if out.degraded:
            if g_ok:
                out.accepted.append(c)
            else:
                out.rejected.append(RejectedClaim(claim=c, reason=g_why, rejected_by="deterministic"))
            continue
        v_ok, v_why = verdicts.get(c.claim_id, (False, "no verdict"))
        if v_ok and g_ok:
            out.accepted.append(c)
        else:
            by = "both" if not (v_ok or g_ok) else ("verifier" if not v_ok else "deterministic")
            out.rejected.append(RejectedClaim(claim=c, reason=v_why if not v_ok else g_why, rejected_by=by))
    return out


__all__ = ["VerificationOutcome", "guard", "verify_claims"]
