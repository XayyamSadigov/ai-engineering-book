# path: book/projects/agentkit/agentkit/dod.py
"""Definition of Done: composable verifiers that decide whether a final answer is accepted.

"The model says it is done" is a claim. A verifier turns it into a check that is harder to
game than a self-report: an artifact exists, a required tool ran, every citation points at
something the agent actually observed, the output parses against a schema.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Protocol

from pydantic import BaseModel, ValidationError

from .state import AgentState


@dataclass(frozen=True)
class Verdict:
    name: str
    passed: bool
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "passed": self.passed, "reason": self.reason}


class Verifier(Protocol):
    name: str

    def __call__(self, answer: str, state: AgentState) -> Verdict: ...


@dataclass
class Check:
    """Wrap any predicate `(answer, state) -> bool | (bool, reason)` as a named verifier."""

    name: str
    fn: Callable[[str, AgentState], bool | tuple[bool, str]]
    failure_hint: str = ""

    def __call__(self, answer: str, state: AgentState) -> Verdict:
        out = self.fn(answer, state)
        passed, reason = out if isinstance(out, tuple) else (bool(out), "")
        return Verdict(self.name, passed, reason or ("" if passed else self.failure_hint))


# ----------------------------------------------------------------------------- built-in verifiers
def non_empty(min_chars: int = 1) -> Check:
    return Check(f"non_empty>={min_chars}", lambda a, s: len(a.strip()) >= min_chars,
                 f"the answer must contain at least {min_chars} characters")


def contains_all(*terms: str, case_sensitive: bool = False) -> Check:
    def fn(answer: str, state: AgentState) -> tuple[bool, str]:
        hay = answer if case_sensitive else answer.lower()
        missing = [t for t in terms if (t if case_sensitive else t.lower()) not in hay]
        return (not missing, f"missing required content: {missing}" if missing else "")
    return Check(f"contains_all{list(terms)}", fn)


def matches(pattern: str, description: str = "") -> Check:
    rx = re.compile(pattern, re.MULTILINE)
    return Check(f"matches:{description or pattern}", lambda a, s: bool(rx.search(a)),
                 f"the answer must match {description or pattern}")


def tool_was_called(*tool_names: str) -> Check:
    def fn(answer: str, state: AgentState) -> tuple[bool, str]:
        missing = [t for t in tool_names if t not in state.tools_called()]
        return (not missing, f"you must call {missing} before answering" if missing else "")
    return Check(f"tool_was_called{list(tool_names)}", fn)


def has_artifact(*names: str) -> Check:
    def fn(answer: str, state: AgentState) -> tuple[bool, str]:
        missing = [n for n in names if n not in state.artifacts]
        return (not missing, f"required artifacts not produced: {missing}" if missing else "")
    return Check(f"has_artifact{list(names)}", fn)


# Ids may contain "#" so passage ids like [hr-pto-policy#c3] or [doc#section] are accepted.
CITATION = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9_.:/#-]*)\]")


def citations_grounded(min_citations: int = 1, pattern: re.Pattern[str] = CITATION) -> Check:
    """Every `[source-id]` in the answer must appear in at least one successful observation."""

    def fn(answer: str, state: AgentState) -> tuple[bool, str]:
        cited = pattern.findall(answer)
        if len(set(cited)) < min_citations:
            return False, f"cite at least {min_citations} source(s) as [source-id]"
        seen = state.observation_text()
        ungrounded = sorted({c for c in cited if c not in seen})
        if ungrounded:
            return False, f"citations not found in any tool result: {ungrounded}"
        return True, ""

    return Check(f"citations_grounded>={min_citations}", fn)


def json_schema(model: type[BaseModel]) -> Check:
    def fn(answer: str, state: AgentState) -> tuple[bool, str]:
        try:
            model.model_validate(json.loads(answer))
            return True, ""
        except (json.JSONDecodeError, ValidationError) as exc:
            return False, f"answer must be JSON matching {model.__name__}: {str(exc).splitlines()[0]}"
    return Check(f"json_schema:{model.__name__}", fn)


# ----------------------------------------------------------------------------- combinators
@dataclass
class AllOf:
    verifiers: list[Verifier]
    name: str = "all_of"

    def __call__(self, answer: str, state: AgentState) -> Verdict:
        failed = [v for v in (ver(answer, state) for ver in self.verifiers) if not v.passed]
        return Verdict(self.name, not failed, "; ".join(f"{v.name}: {v.reason}" for v in failed))


@dataclass
class AnyOf:
    verifiers: list[Verifier]
    name: str = "any_of"

    def __call__(self, answer: str, state: AgentState) -> Verdict:
        verdicts = [ver(answer, state) for ver in self.verifiers]
        if any(v.passed for v in verdicts):
            return Verdict(self.name, True)
        return Verdict(self.name, False, " OR ".join(f"{v.name}: {v.reason}" for v in verdicts))


def all_of(*verifiers: Verifier) -> AllOf:
    return AllOf(list(verifiers))


def any_of(*verifiers: Verifier) -> AnyOf:
    return AnyOf(list(verifiers))


@dataclass
class DoDResult:
    passed: bool
    verdicts: list[Verdict] = field(default_factory=list)

    def feedback(self) -> str:
        failed = [v for v in self.verdicts if not v.passed]
        lines = [f"- {v.name}: {v.reason}" for v in failed]
        return "Your final answer was not accepted. Unmet criteria:\n" + "\n".join(lines) + \
            "\nContinue working: gather what is missing, then answer again."


class DefinitionOfDone:
    """A named list of verifiers, all of which must pass. Every verdict is recorded."""

    def __init__(self, *verifiers: Verifier, description: str = "") -> None:
        self.verifiers = list(verifiers)
        self.description = description

    def verify(self, answer: str, state: AgentState) -> DoDResult:
        verdicts = [v(answer, state) for v in self.verifiers]
        return DoDResult(passed=all(v.passed for v in verdicts), verdicts=verdicts)

    def as_prompt(self) -> str:
        """The same criteria, stated to the model. The prompt guides; the verifier decides."""
        names = "\n".join(f"- {getattr(v, 'name', type(v).__name__)}" for v in self.verifiers)
        head = self.description or "Your answer will be checked automatically against:"
        return f"{head}\n{names}"


__all__ = [
    "Verdict", "Verifier", "Check", "non_empty", "contains_all", "matches", "tool_was_called", "has_artifact",
    "citations_grounded", "json_schema", "CITATION", "AllOf", "AnyOf", "all_of", "any_of", "DoDResult",
    "DefinitionOfDone",
]
