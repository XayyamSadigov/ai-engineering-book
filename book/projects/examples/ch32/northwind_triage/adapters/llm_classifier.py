# path: book/projects/examples/ch32/northwind_triage/adapters/llm_classifier.py
"""Outbound adapter: ClassifierPort implemented on top of aie_core's provider-neutral client.

This is the anti-corruption layer. Everything provider-shaped (CompletionRequest, Completion,
the LLMError taxonomy) is translated here into application types (ClassifierOutput,
ClassifierUnavailable). Swapping providers, or replacing the LLM with a fine-tuned small
model behind an HTTP endpoint, changes this file and the composition root, nothing else.
"""
from __future__ import annotations

from aie_core import CompletionRequest, Message
from aie_core.llm.client import LLMClient
from aie_core.llm.errors import LLMError

from ..application.ports import ClassifierOutput, ClassifierUnavailable
from ..domain import TriageDecision

TRIAGE_JSON_SCHEMA = TriageDecision.model_json_schema()


class LLMClassifier:
    def __init__(self, client: LLMClient, *, max_tokens: int = 300, timeout_s: float = 8.0,
                 use_response_schema: bool = True) -> None:
        self.client = client
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.use_response_schema = use_response_schema

    def classify(self, *, system: str, user: str, model: str) -> ClassifierOutput:
        req = CompletionRequest(
            messages=[Message.system(system), Message.user(user)],
            model=model,
            temperature=0.0,
            max_tokens=self.max_tokens,
            response_schema=TRIAGE_JSON_SCHEMA if self.use_response_schema else None,
            timeout_s=self.timeout_s,
            metadata={"use_case": "triage"},
        )
        try:
            completion = self.client.complete(req)
        except LLMError as exc:
            raise ClassifierUnavailable(f"{type(exc).__name__}: {exc}",
                                        retryable=getattr(exc, "retryable", False)) from exc
        if completion.finish_reason not in {"stop", "end_turn", "tool_calls"}:
            # Truncated output ("length") is a failure to classify, not a short answer.
            raise ClassifierUnavailable(f"finish_reason={completion.finish_reason}")
        return ClassifierOutput(
            text=completion.text,
            served_model=completion.model,
            provider=completion.provider,
            input_tokens=completion.usage.input_tokens,
            output_tokens=completion.usage.output_tokens,
            latency_ms=completion.latency_ms,
        )
