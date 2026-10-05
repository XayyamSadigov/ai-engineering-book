# path: book/projects/p6-research-team/research_team/text.py
"""Small, deterministic text utilities shared by search, verification, and scoring."""
from __future__ import annotations

import re

STOPWORDS = frozenset(
    "a an and are as at be before by can could do does for from had has have how i if in into is it its may "
    "me must my no not of on or our should so than that the their them then there these they this to under "
    "up was we what when where which while who why will with within without you your any all also each per "
    "about after other only one more most such same need needs".split()
)

_WORD = re.compile(r"[a-z0-9]+(?:[-/][a-z0-9]+)*")
_NUMBER = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?(?![\w])")
_UNITS = r"(days?|usd|hours?|weeks?|months?|years?|km|nights?|minutes?|gb|%)"
_QUANTITY = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:[a-z-]+\s+){0,2}?" + _UNITS + r"\b", re.IGNORECASE)


def stem(word: str) -> str:
    """Crude suffix stripping: good enough to match travel/travelling, policy/policies, day/days."""
    w = word
    if len(w) > 5 and w.endswith("ing"):
        w = w[:-3]
        if len(w) > 3 and w[-1] == w[-2]:
            w = w[:-1]
    elif len(w) > 4 and w.endswith("ies"):
        w = w[:-3] + "y"
    elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    return w


def tokens(text: str) -> list[str]:
    out: list[str] = []
    for raw in _WORD.findall(text.lower()):
        for part in re.split(r"[-/]", raw):
            if part and part not in STOPWORDS:
                out.append(stem(part))
    return out


def content_terms(text: str) -> set[str]:
    """Non-numeric content words, stemmed."""
    return {t for t in tokens(text) if not t.isdigit() and len(t) > 1}


def numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUMBER.findall(text)}


def quantities(text: str) -> dict[str, set[str]]:
    """Numbers attached to a unit within two words: '10 unused PTO days' -> {'day': {'10'}}."""
    out: dict[str, set[str]] = {}
    for value, unit in _QUANTITY.findall(text):
        out.setdefault(stem(unit.lower()), set()).add(value.replace(",", ""))
    return out


def clean_markdown(line: str) -> str:
    """Turn a table row, bullet, or emphasized line into a plain sentence."""
    s = line.strip()
    if s.startswith("|") and s.endswith("|"):
        cells = [c.strip() for c in s.strip("|").split("|")]
        cells = [c for c in cells if c]
        s = cells[0] + ": " + "; ".join(cells[1:]) if len(cells) > 1 else (cells[0] if cells else "")
    s = re.sub(r"^\s*(?:[-*]|\d+\.)\s+", "", s)
    s = s.replace("**", "").replace("`", "")
    return re.sub(r"\s+", " ", s).strip()


def _cells(row: str) -> list[str]:
    return [c.strip().replace("**", "").replace("`", "") for c in row.strip().strip("|").split("|")]


def _blocks(text: str) -> list[tuple[str, str]]:
    """Re-join hard-wrapped Markdown into ('item', text) blocks: paragraphs and bullets with their
    continuation lines, and table rows rendered as 'Header: value; Header: value'."""
    blocks: list[tuple[str, str]] = []
    lines = text.splitlines()
    header: list[str] | None = None
    for i, line in enumerate(lines):
        raw = line.strip()
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if raw.startswith("|"):
            if re.fullmatch(r"\|?[\s:|-]+\|?", raw):
                continue
            if re.fullmatch(r"\|?[\s:|-]+\|?", nxt):      # header row of a new table
                header = _cells(raw)
                continue
            cells = _cells(raw)
            if header and len(header) == len(cells):
                blocks.append(("row", "; ".join(f"{h}: {c}" for h, c in zip(header, cells) if c)))
            else:
                blocks.append(("row", "; ".join(c for c in cells if c)))
            continue
        header = None
        if not raw or raw.startswith("#"):
            blocks.append(("break", ""))
        elif re.match(r"^(?:[-*]|\d+\.)\s+", raw) or not blocks or blocks[-1][0] != "item":
            blocks.append(("item", raw))
        else:
            blocks[-1] = ("item", blocks[-1][1] + " " + raw)
    return [b for b in blocks if b[0] != "break"]


def units_of_text(text: str) -> list[str]:
    """Split a passage into claim-sized units: sentences, bullets, and table rows."""
    units: list[str] = []
    for kind, raw in _blocks(text):
        cleaned = clean_markdown(raw) if kind == "item" else re.sub(r"\s+", " ", raw).strip()
        if kind == "row":
            if cleaned:
                units.append(cleaned)
            continue
        for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", cleaned):
            if len(sent) >= 12 and not sent.endswith("?"):
                units.append(sent.strip())
    return units


__all__ = ["STOPWORDS", "clean_markdown", "content_terms", "numbers", "quantities", "stem", "tokens", "units_of_text"]
