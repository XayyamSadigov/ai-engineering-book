# path: book/projects/examples/ch25/taskevals/tools.py
"""Tool-usage evaluators for single-decision function calling.

Case format:
    input    = {"request": "...", "tools": ["search_tickets", "create_ticket", ...]}  # offered tools
    expected = {"tool": "create_ticket" | null, "arguments": {"priority": "P1"}}     # null: answer without a tool
output   = list of tool-call dicts [{"id", "name", "arguments"}] (what evalkit records for a
           Completion that returned tool calls) or a plain string when the model answered directly.

Selection accuracy is a classification problem over {each tool, no tool}, so the same advice
applies as for any classifier: look at the confusion between tools, not only the accuracy.
Argument validity is a schema check; argument correctness compares the values the task fixes.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from evalkit import EvalCase, Run, Score
from evalkit.metrics import ConfusionMatrix, json_schema_valid, normalize_text

from .trajectory import NORTHWIND_TOOLS, ToolInfo

NO_TOOL = "<none>"


def first_call(output: Any) -> dict[str, Any] | None:
    if isinstance(output, list) and output and isinstance(output[0], dict) and "name" in output[0]:
        return output[0]
    return None


def _values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, str) and isinstance(b, str):
        return normalize_text(a) == normalize_text(b)
    return a == b


TOOL_METRICS = ["tool_selected_correctly", "tool_known", "tool_args_valid", "tool_args_correct"]


class ToolUseEvaluator:
    name = "tool_use"
    version = "1"
    metric_names = TOOL_METRICS

    def __init__(self, catalog: Mapping[str, ToolInfo] = NORTHWIND_TOOLS) -> None:
        self.catalog = catalog

    def __call__(self, case: EvalCase, output: Any) -> list[Score]:
        call = first_call(output)
        chosen = call["name"] if call else NO_TOOL
        want = case.expected.get("tool") or NO_TOOL
        offered = set(case.input.get("tools", self.catalog))
        known = chosen == NO_TOOL or (chosen in self.catalog and chosen in offered)
        selected = chosen == want
        scores = [
            Score(name="tool_selected_correctly", value=float(selected), passed=selected,
                  detail={"gold": want, "pred": chosen}),
            Score(name="tool_known", value=float(known), passed=known,
                  detail=None if known else f"hallucinated or not offered: {chosen}"),
        ]
        if call is None or chosen not in self.catalog:
            # No arguments to judge: valid iff no call was expected; correct iff selection was correct.
            scores += [Score(name="tool_args_valid", value=float(known), passed=known),
                       Score(name="tool_args_correct", value=float(selected), passed=selected)]
            return scores
        ok, errors = json_schema_valid(call.get("arguments", {}), self.catalog[chosen].parameters)
        wrong = {k: call.get("arguments", {}).get(k) for k, v in (case.expected.get("arguments") or {}).items()
                 if not _values_equal(call.get("arguments", {}).get(k), v)}
        args_correct = selected and not wrong
        scores += [
            Score(name="tool_args_valid", value=float(ok), passed=ok, detail=errors or None),
            Score(name="tool_args_correct", value=float(args_correct), passed=args_correct,
                  detail=wrong or None),
        ]
        return scores


def tool_confusion(run: Run) -> ConfusionMatrix:
    rows = [r.details["tool_selected_correctly"] for r in run.results
            if isinstance(r.details.get("tool_selected_correctly"), dict)]
    return ConfusionMatrix([d["gold"] for d in rows], [d["pred"] for d in rows])


__all__ = ["NO_TOOL", "first_call", "ToolUseEvaluator", "TOOL_METRICS", "tool_confusion"]
