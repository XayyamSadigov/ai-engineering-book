# path: book/projects/examples/ch04/prompts/judge.py
"""An LLM judge for one dimension (groundedness), itself a registered, versioned prompt.

A judge is a prompt whose output is a score. It gets the same treatment as any other
prompt: a version, a schema, golden cases, and calibration against human labels before
anyone trusts it (Chapter 24). It is never ground truth by definition.
"""
from __future__ import annotations

import json
from typing import Any

from aie_core.llm.client import LLMClient
from aie_core.llm.errors import MalformedResponseError
from aie_core.observability import Tracer

from .registry import PromptVersion
from .regression import Case, JudgeVerdict, schema_errors
from .tracing import traced_complete


class LLMJudge:
    def __init__(
        self,
        prompt: PromptVersion,
        client: LLMClient,
        *,
        pass_score: int = 3,
        evidence_variable: str = "documents",
        tracer: Tracer | None = None,
    ) -> None:
        self.prompt = prompt
        self.client = client
        self.pass_score = pass_score
        self.evidence_variable = evidence_variable
        self.tracer = tracer

    def __call__(self, case: Case, output: str) -> JudgeVerdict:
        evidence: Any = case.reference.get("evidence", case.variables.get(self.evidence_variable))
        if evidence is None:
            raise ValueError(f"case {case.id}: no evidence for the judge")
        try:
            answer = json.loads(output).get("answer", output)
        except (json.JSONDecodeError, AttributeError):
            answer = output
        rendered = self.prompt.render({"evidence": evidence, "answer": answer})
        completion = traced_complete(self.client, rendered, self.tracer)
        try:
            verdict = json.loads(completion.text)
        except json.JSONDecodeError as exc:
            raise MalformedResponseError(f"judge returned non-JSON: {completion.text[:80]!r}") from exc
        errors = schema_errors(verdict, self.prompt.output_schema or {})
        if errors:
            raise MalformedResponseError(f"judge output violates schema: {errors[:2]}")
        unsupported = verdict.get("unsupported_claims", [])
        return JudgeVerdict(
            score=float(verdict["score"]),
            passed=int(verdict["score"]) >= self.pass_score,
            rationale="; ".join(unsupported)[:300],
        )


__all__ = ["LLMJudge"]
