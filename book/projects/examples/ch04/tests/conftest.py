# path: book/projects/examples/ch04/tests/conftest.py
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prompts import PromptRegistry, load_cases  # noqa: E402


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def registry() -> PromptRegistry:
    return PromptRegistry.from_directory(ROOT / "prompt_files")


@pytest.fixture(scope="session")
def ticket_cases():
    return load_cases(ROOT / "cases" / "ticket_classify.jsonl")


@pytest.fixture(scope="session")
def answer_cases():
    return load_cases(ROOT / "cases" / "assist_answer.jsonl")
