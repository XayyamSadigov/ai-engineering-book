# path: book/projects/p6-research-team/research_team/render.py
"""Deterministic answer rendering and parsing. Used as the synthesis fallback, by the
single-agent-plus-verifier configuration, and by the offline scripted model."""
from __future__ import annotations

import re
from typing import Iterable

from .checks import CITATION
from .contracts import Claim, Conflict, EvidenceRef


def render_answer(question: str, claims: Iterable[Claim], *, conflicts: Iterable[Conflict] = (),
                  gaps: Iterable[str] = (), skipped: Iterable[str] = (),
                  sections: dict[str, str] | None = None) -> str:
    """Group claims by the task that produced them; one cited bullet per claim."""
    claims = list(claims)
    sections = sections or {}
    lines = [f"Answer to: {question}", ""]
    groups: dict[str, list[Claim]] = {}
    for c in claims:
        groups.setdefault(c.source_task, []).append(c)
    for task, items in groups.items():
        title = sections.get(task)
        if title:
            lines.append(f"{title}:")
        for c in items:
            cites = " ".join(f"[{e.passage_id}]" for e in c.evidence)
            lines.append(f"- {c.text} {cites}")
        lines.append("")
    conflicts = list(conflicts)
    if conflicts:
        lines.append("Sources disagree:")
        for x in conflicts:
            lines.append(f"- {x.reason}. Follow {x.preferred_doc}, the newer document.")
        lines.append("")
    missing = list(gaps) + [f"not researched ({s})" for s in skipped]
    if missing:
        lines.append("Not covered:")
        lines.extend(f"- {g}" for g in missing)
    return "\n".join(lines).strip()


def parse_cited_lines(answer: str, *, source_task: str = "answer") -> list[Claim]:
    """Every line with at least one [doc#section] citation becomes a claim."""
    out: list[Claim] = []
    for line in answer.splitlines():
        pids = CITATION.findall(line)
        if not pids:
            continue
        text = CITATION.sub("", line).strip().lstrip("-* ").strip()
        if len(text) < 3:
            continue
        evidence = [EvidenceRef(doc_id=p.split("#", 1)[0], passage_id=p) for p in dict.fromkeys(pids)]
        out.append(Claim(claim_id=f"{source_task}.c{len(out)}", text=text, evidence=evidence,
                         source_task=source_task))
    return out


def claim_key(claim: Claim) -> str:
    """Identity of a claim for deduplication: normalized text plus the first cited passage."""
    norm = " ".join(re.findall(r"[a-z0-9]+", claim.text.lower()))
    return f"{claim.evidence[0].passage_id}|{norm}"


def short_label(objective: str, limit: int = 90) -> str:
    s = re.sub(r"\s+", " ", objective).strip()
    return s if len(s) <= limit else s[: limit - 3] + "..."


__all__ = ["claim_key", "parse_cited_lines", "render_answer", "short_label"]
