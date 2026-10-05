# path: book/capstone/northwind-assist/northwind_assist/_paths.py
"""Locate the book's example modules and shared data.

Chapters 4, 5, 7, 25, 26, 30, 31, 32 ship their code as example directories, not installable
packages, and the integration notes say to put the directory on sys.path. This module does that
once, in one place, so the rest of the capstone imports `instrument`, `caching`, `router` and so
on by name. Directories are appended, never prepended, so an example module can never shadow an
installed package. Override the book location with NA_BOOK_ROOT (the Docker image sets it).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
BOOK_ROOT = Path(os.environ.get("NA_BOOK_ROOT", _HERE.parents[3]))
PROJECTS = BOOK_ROOT / "projects"
EXAMPLES = PROJECTS / "examples"
SHARED_DATA = PROJECTS / "shared-data"
DOCS_DIR = SHARED_DATA / "docs"
GOLD_PATH = SHARED_DATA / "eval" / "retrieval_gold.jsonl"
CAPSTONE_ROOT = _HERE.parents[1]

# Chapter -> what the capstone imports from it.
EXAMPLE_DIRS: dict[str, str] = {
    "ch04": "prompts (PromptRegistry)",
    "ch05": "context (ContextBuilder)",
    "ch07": "catalog, router (model routing)",
    "ch25": "taskevals, ci/release_gate (trajectory evaluators, release gate)",
    "ch26": "attack_corpus (red-team documents)",
    "ch30": "caching, budgets (scoped caches, SpendGuard)",
    "ch31": "semconv, instrument, otel_setup (AITracer, OTel)",
    "ch32": "northwind_triage.flags, northwind_triage.version_manifest",
}


def install() -> None:
    for name in EXAMPLE_DIRS:
        path = str(EXAMPLES / name)
        if path not in sys.path:
            sys.path.append(path)
    for path in (str(EXAMPLES / "ch25" / "ci"), str(SHARED_DATA)):
        if path not in sys.path:
            sys.path.append(path)


install()

__all__ = ["BOOK_ROOT", "PROJECTS", "EXAMPLES", "SHARED_DATA", "DOCS_DIR", "GOLD_PATH", "CAPSTONE_ROOT", "install"]
