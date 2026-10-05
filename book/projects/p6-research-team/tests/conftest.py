# path: book/projects/p6-research-team/tests/conftest.py
"""Shared fixtures: the real shared-data corpus, an employee principal, and scripted policies
that tests bend in one place (planner output, verifier honesty, barriers)."""
from __future__ import annotations

import json
import threading
from typing import Any

import pytest

from aie_core.llm.providers import FakeLLM
from aie_core.llm.types import CompletionRequest
from research_team import Corpus
from research_team.scripted import ScriptedPolicy

PRINCIPAL = {"user_id": "emp-1042", "groups": ["all"], "tenant": "retail"}
TRAVEL_Q = "What do I need to do before travelling abroad with a company laptop?"


@pytest.fixture(scope="session")
def corpus() -> Corpus:
    return Corpus.from_shared_data()


@pytest.fixture
def principal() -> dict[str, Any]:
    return dict(PRINCIPAL)


class Policy(ScriptedPolicy):
    """ScriptedPolicy with knobs: fixed subquestions, a lying verifier, a barrier for researchers."""

    def __init__(self, *, subquestions: list[str] | None = None, lying_verifier: bool = False,
                 barrier: threading.Barrier | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        self.subquestions = subquestions
        self.lying_verifier = lying_verifier
        self.barrier = barrier
        self.researcher_threads: set[str] = set()
        self._lock = threading.Lock()

    def planner(self, req: CompletionRequest, task: dict[str, Any], step: int) -> str:
        if self.subquestions is None:
            return super().planner(req, task, step)
        return json.dumps({"subquestions": [{"id": f"sq{i + 1}", "question": q, "topic": ""}
                                            for i, q in enumerate(self.subquestions)]})

    def researcher(self, req: CompletionRequest, task: dict[str, Any], step: int):
        with self._lock:
            self.researcher_threads.add(threading.current_thread().name)
        if self.barrier is not None and step == 0:
            self.barrier.wait()          # raises BrokenBarrierError unless all parties arrive concurrently
        return super().researcher(req, task, step)

    def verifier(self, req: CompletionRequest, task: dict[str, Any], step: int):
        if self.lying_verifier:          # a compromised or careless verifier approves everything
            return json.dumps({"verdicts": [{"claim_id": c["claim_id"], "supported": True, "reason": "looks fine"}
                                            for c in task["inputs"]["claims"]]})
        return super().verifier(req, task, step)


def llm(policy: ScriptedPolicy | None = None) -> FakeLLM:
    return FakeLLM(handler=policy or ScriptedPolicy(), model="scripted-model")
