# path: book/projects/p5-incident-agent/incident_agent/domain/dod.py
"""The deterministic Definition of Done for an incident report.

These checks are the hard gate: a report that fails any of them is never shown for approval.
They are cheap, exact, and cannot be talked out of a verdict, which is why they run before
the rubric judge and why their output is the most useful feedback for a revision.
"""
from __future__ import annotations

from collections.abc import Iterable

from .models import Problem
from .report import CLAIM_SECTIONS, REQUIRED_SECTIONS, claims, cited, normalize_heading, parse_sections


def check_report(report: str, evidence_ids: Iterable[str], runbook_catalog: Iterable[str]) -> list[Problem]:
    evidence, catalog = set(evidence_ids), set(runbook_catalog)
    sections = parse_sections(report)
    problems: list[Problem] = []

    for name in REQUIRED_SECTIONS:
        if not sections.get(normalize_heading(name), "").strip():
            problems.append(Problem(code="missing_section", message=f"section '{name}' is missing or empty"))

    for name in CLAIM_SECTIONS:
        for claim in claims(sections.get(normalize_heading(name), "")):
            if not cited(claim):
                problems.append(Problem(code="uncited_claim", message=f"{name}: claim has no [source-id]: {claim[:120]!r}"))

    for source in sorted(set(cited(report))):
        if source not in evidence:
            problems.append(Problem(code="unknown_citation",
                                    message=f"[{source}] is not a source this investigation observed"))

    body = sections.get(normalize_heading("Recommended runbook"), "")
    if body:
        named = cited(body)
        runbooks = [n for n in named if n in catalog]
        for n in named:
            if n not in catalog and not n.startswith(("metric:", "deploy:", "inc-")):
                problems.append(Problem(code="runbook_not_in_catalog",
                                        message=f"recommended runbook [{n}] does not exist; choose one from the "
                                                f"retrieved runbooks"))
        if not runbooks:
            problems.append(Problem(code="runbook_missing", message="recommend exactly one runbook by its [id]"))
        elif len(set(runbooks)) > 1:
            problems.append(Problem(code="runbook_ambiguous",
                                    message=f"recommend exactly one runbook, not {sorted(set(runbooks))}"))
        for n in runbooks:
            if n not in evidence:
                problems.append(Problem(code="runbook_not_retrieved",
                                        message=f"runbook [{n}] exists but was never retrieved in this investigation"))
    return problems


def feedback(problems: list[Problem]) -> list[str]:
    return [p.message for p in problems]


__all__ = ["check_report", "feedback"]
