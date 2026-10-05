# path: book/projects/examples/ch25/taskevals/prompts.py
"""Prompt-level evaluators: a deterministic contract plus one rubric judge.

Chapter 4's regression harness (examples/ch04/prompts/regression.py) is the narrow tool a
prompt author runs before opening a pull request: one prompt, its cases, assertions, a diff.
This module lifts the same checks into evalkit evaluators so prompt cases share a Run, the
statistics, the report, and the release gate with every other task in the system.

Case format:
    expected = {"required": ["30 days"], "forbidden": ["password"], "max_chars": 900,
                "schema": {...JSON Schema...}}            # every key optional
"""
from __future__ import annotations

import json
from typing import Any

from aie_core import LLMClient

from evalkit import EvalCase, LLMJudge, Rubric, RubricLevel, Score
from evalkit.metrics import contains, forbids, json_schema_valid

REPLY_ADDRESSES_REQUEST = Rubric(
    name="addresses_request",
    task="Judge whether the support reply addresses the employee's actual request with a concrete next step.",
    levels=[
        RubricLevel(score=0, description="does not address the request, or gives wrong instructions"),
        RubricLevel(score=1, description="addresses the topic but gives no concrete next step or misses part of the request"),
        RubricLevel(score=2, description="addresses every part of the request and states a concrete next step"),
    ],
    pass_threshold=2,
    flagged_label="unaddressed_parts",
)

PROMPT_METRICS = ["prompt_schema_valid", "prompt_required", "prompt_forbidden", "prompt_length_ok"]


class PromptContractEvaluator:
    """Deterministic checks a prompt's output must always satisfy. Output: str or JSON-able dict."""

    name = "prompt_contract"
    version = "1"
    metric_names = PROMPT_METRICS

    def __call__(self, case: EvalCase, output: Any) -> list[Score]:
        exp = case.expected or {}
        text = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
        scores = []
        if "schema" in exp:
            ok, errors = json_schema_valid(output, exp["schema"])
            scores.append(Score(name="prompt_schema_valid", value=float(ok), passed=ok, detail=errors or None))
        else:
            scores.append(Score(name="prompt_schema_valid", value=1.0, passed=True))
        ok_req, missing = contains(text, exp.get("required", []), mode="all") if exp.get("required") else (True, [])
        ok_forb, found = forbids(text, exp.get("forbidden", [])) if exp.get("forbidden") else (True, [])
        length_ok = len(text) <= exp.get("max_chars", 10**9)
        scores += [
            Score(name="prompt_required", value=float(ok_req), passed=ok_req, detail=missing or None),
            Score(name="prompt_forbidden", value=float(ok_forb), passed=ok_forb, detail=found or None),
            Score(name="prompt_length_ok", value=float(length_ok), passed=length_ok),
        ]
        return scores


def reply_judge(client: LLMClient, *, model: str | None = None) -> Any:
    return LLMJudge(client, REPLY_ADDRESSES_REQUEST, model=model).as_evaluator(
        input_fn=lambda c: c.input.get("request", c.input),
        answer_fn=lambda o: o if isinstance(o, str) else o.get("reply", o),
    )


__all__ = ["REPLY_ADDRESSES_REQUEST", "PromptContractEvaluator", "PROMPT_METRICS", "reply_judge"]
