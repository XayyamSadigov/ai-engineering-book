# path: book/projects/p5-incident-agent/incident_agent/domain/report.py
"""The report contract: required sections, what counts as a claim, how citations are written."""
from __future__ import annotations

import re

REQUIRED_SECTIONS = ("Summary", "Impact", "Timeline", "Likely cause", "Recommended runbook", "Next steps")
CLAIM_SECTIONS = ("Summary", "Impact", "Timeline", "Likely cause")   # every claim here needs a citation
CITATION = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9_.:/-]*)\]")
_HEADING = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_BULLET = re.compile(r"^\s*(?:[-*]|\d+\.)\s+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def normalize_heading(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().rstrip(":")).lower()


def parse_sections(markdown: str) -> dict[str, str]:
    """Map normalized level-2 headings to their bodies."""
    out: dict[str, str] = {}
    matches = list(_HEADING.finditer(markdown))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        out[normalize_heading(m.group(1))] = markdown[m.end():end].strip()
    return out


def claims(body: str) -> list[str]:
    """Each bullet is one claim; prose paragraphs are split into sentences."""
    out: list[str] = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            out.extend(s.strip() for s in _SENTENCE_END.split(" ".join(paragraph)) if s.strip())
            paragraph.clear()

    for line in body.splitlines():
        if not line.strip() or line.lstrip().startswith("|"):
            flush()
            continue
        if _BULLET.match(line):
            flush()
            out.append(_BULLET.sub("", line).strip())
        else:
            paragraph.append(line.strip())
    flush()
    return [c for c in out if c]


def cited(text: str) -> list[str]:
    return CITATION.findall(text)


__all__ = ["REQUIRED_SECTIONS", "CLAIM_SECTIONS", "CITATION", "normalize_heading", "parse_sections", "claims", "cited"]
