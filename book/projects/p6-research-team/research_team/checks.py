# path: book/projects/p6-research-team/research_team/checks.py
"""Deterministic checks: claim support, conflicts between claims, and agentkit DoD verifiers
that work on JSON outputs. None of these call a model, so they cannot be talked out of a verdict."""
from __future__ import annotations

import json
import re
from datetime import date
from typing import Callable, Iterable

from pydantic import BaseModel, ValidationError

from agentkit import AgentState, Check
from aie_core.llm.structured import extract_json

from .contracts import Claim, Conflict
from .text import content_terms, numbers, quantities

CITATION = re.compile(r"\[([a-z0-9][a-z0-9-]*#[a-z0-9-]+)\]")


def deterministic_support(claim: str, passage: str, *, min_overlap: float = 0.6) -> tuple[bool, str]:
    """A claim is supported only if every number in it occurs in the passage and most of its
    content words do. Cheap, strict on numbers, lenient on wording; a guard, not an entailment model."""
    missing = sorted(numbers(claim) - numbers(passage))
    if missing:
        return False, f"numbers not in source: {missing}"
    terms = content_terms(claim)
    if not terms:
        return False, "claim has no content words"
    overlap = len(terms & content_terms(passage)) / len(terms)
    if overlap < min_overlap:
        return False, f"only {overlap:.0%} of the claim's terms appear in the source"
    return True, f"numbers present, {overlap:.0%} term overlap"


def find_conflicts(claims: Iterable[Claim], updated: Callable[[str], date | None],
                   *, min_jaccard: float = 0.25) -> list[Conflict]:
    """Pairs of claims from different documents about the same subject with disjoint values for
    the same unit. The more recently updated document is preferred; the user still sees both."""
    items = [c for c in claims if c.evidence]
    out: list[Conflict] = []
    seen: set[tuple[frozenset[str], str]] = set()
    for i, a in enumerate(items):
        for b in items[i + 1:]:
            da, db = a.evidence[0].doc_id, b.evidence[0].doc_id
            if da == db:
                continue
            ta, tb = content_terms(a.text), content_terms(b.text)
            if not ta or not tb or len(ta & tb) / len(ta | tb) < min_jaccard:
                continue
            qa, qb = quantities(a.text), quantities(b.text)
            for unit in sorted(set(qa) & set(qb)):
                key = (frozenset((da, db)), unit)
                if qa[unit].isdisjoint(qb[unit]) and key not in seen:
                    seen.add(key)
                    ua, ub = updated(da) or date.min, updated(db) or date.min
                    preferred = da if ua >= ub else db
                    out.append(Conflict(
                        claim_ids=(a.claim_id, b.claim_id), doc_ids=(da, db), unit=unit,
                        values=(sorted(qa[unit]), sorted(qb[unit])), preferred_doc=preferred,
                        reason=f"{da} says {sorted(qa[unit])} {unit}, {db} says {sorted(qb[unit])} {unit}; "
                               f"{preferred} is newer"))
                    break
    return out


def parse_json_answer(answer: str) -> dict | None:
    try:
        data = json.loads(extract_json(answer))
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


# ----------------------------------------------------------------------------- DoD verifiers
def valid_json(model: type[BaseModel]) -> Check:
    """Like agentkit's json_schema verifier, but tolerant of Markdown fences around the JSON."""

    def fn(answer: str, state: AgentState) -> tuple[bool, str]:
        try:
            model.model_validate_json(extract_json(answer))
            return True, ""
        except ValidationError as exc:
            return False, f"answer must be JSON matching {model.__name__}: {str(exc).splitlines()[0]}"

    return Check(f"valid_json:{model.__name__}", fn)


def evidence_observed() -> Check:
    """Every passage_id cited in a ResearchFindings answer must appear in a tool result the
    agent actually received. Stops a worker from citing ids it never saw."""

    def fn(answer: str, state: AgentState) -> tuple[bool, str]:
        data = parse_json_answer(answer)
        if data is None:
            return False, "answer is not JSON"
        seen = state.observation_text()
        cited = {e.get("passage_id", "") for c in data.get("claims", []) for e in c.get("evidence", [])}
        unseen = sorted(p for p in cited if f"[{p}]" not in seen)
        if unseen:
            return False, f"evidence not found in any tool result: {unseen}"
        return True, ""

    return Check("evidence_observed", fn)


def verdicts_cover(claim_ids: list[str]) -> Check:
    def fn(answer: str, state: AgentState) -> tuple[bool, str]:
        data = parse_json_answer(answer) or {}
        got = {v.get("claim_id") for v in data.get("verdicts", [])}
        missing = sorted(set(claim_ids) - got)
        return (not missing, f"no verdict for claims {missing}" if missing else "")

    return Check("verdicts_cover_all_claims", fn)


def cites_only(allowed: set[str], *, min_citations: int = 1) -> Check:
    """The synthesis may cite only passages behind verified claims."""

    def fn(answer: str, state: AgentState) -> tuple[bool, str]:
        cited = set(CITATION.findall(answer))
        if len(cited) < min_citations:
            return False, f"cite at least {min_citations} verified passage(s) as [passage-id]"
        extra = sorted(cited - allowed)
        return (not extra, f"citations outside the verified evidence: {extra}" if extra else "")

    return Check("cites_only_verified", fn)


__all__ = ["CITATION", "cites_only", "deterministic_support", "evidence_observed", "find_conflicts",
           "parse_json_answer", "valid_json", "verdicts_cover"]
