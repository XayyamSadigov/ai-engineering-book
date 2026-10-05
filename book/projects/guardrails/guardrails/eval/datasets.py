# path: book/projects/guardrails/guardrails/eval/datasets.py
"""Labeled cases for measuring guardrails: benign traffic (for false positives) and attacks (for
bypass rate). Sources: the package's own JSONL files, the Northwind shared corpus when present,
and the Chapter 26 adversarial corpus."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from ..pipeline import Stage

PROJECT_DIR = Path(__file__).resolve().parents[2]
PROJECTS_DIR = PROJECT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"


@dataclass(frozen=True)
class Case:
    id: str
    stage: Stage
    text: str
    is_attack: bool
    note: str = ""


def load_ch26() -> ModuleType:
    """Import `attack_corpus.py` from Chapter 26 by path (it is an example module, not a package)."""
    directory = Path(os.environ.get("GUARDRAILS_CH26_DIR", PROJECTS_DIR / "examples" / "ch26"))
    path = directory / "attack_corpus.py"
    if "attack_corpus" in sys.modules:
        return sys.modules["attack_corpus"]
    spec = importlib.util.spec_from_file_location("attack_corpus", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Chapter 26 attack corpus not found at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["attack_corpus"] = module
    spec.loader.exec_module(module)
    return module


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def input_cases() -> list[Case]:
    """User messages: benign questions (some deliberately tricky) and direct injection attempts."""
    rows = _jsonl(DATA_DIR / "input_cases.jsonl")
    return [Case(r["id"], Stage.INPUT, r["text"], bool(r["attack"]), r.get("note", "")) for r in rows]


def _strip_front_matter(text: str) -> str:
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            return text[end + 5:]
    return text


def context_cases(include_shared: bool = True) -> list[Case]:
    """Untrusted documents: the Northwind corpus and tickets (benign) and the Chapter 26 carriers."""
    cases: list[Case] = []
    shared = PROJECTS_DIR / "shared-data"
    if include_shared and (shared / "docs").is_dir():
        for p in sorted((shared / "docs").glob("*.md")):
            raw = p.read_text(encoding="utf-8")
            # The shared corpus ships one deliberate injection fixture, tagged `security-test`.
            is_fixture = "security-test" in raw.split("\n---\n", 1)[0]
            cases.append(Case(f"doc:{p.stem}", Stage.CONTEXT, _strip_front_matter(raw), is_fixture,
                              "shared-data injection fixture" if is_fixture else ""))
        for row in _jsonl(shared / "tickets.jsonl"):
            cases.append(Case(f"ticket:{row['id']}", Stage.CONTEXT, f"{row['subject']}\n\n{row['body']}", False))
    for row in _jsonl(DATA_DIR / "benign_context.jsonl"):
        cases.append(Case(row["id"], Stage.CONTEXT, row["text"], False, row.get("note", "")))
    ac = load_ch26()
    for d in ac.adversarial_documents():
        cases.append(Case(d.doc_id, Stage.CONTEXT, d.body, True, d.variant.value))
    return cases


def output_cases() -> list[Case]:
    """Model answers: benign (with allowlisted links and citations) and exfiltration attempts."""
    rows = _jsonl(DATA_DIR / "output_cases.jsonl")
    return [Case(r["id"], Stage.OUTPUT, r["text"], bool(r["attack"]), r.get("note", "")) for r in rows]


__all__ = ["Case", "load_ch26", "input_cases", "context_cases", "output_cases", "DATA_DIR"]
