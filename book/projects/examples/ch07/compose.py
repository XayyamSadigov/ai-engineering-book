# path: book/projects/examples/ch07/compose.py
"""One Part II request: registry -> ContextBuilder -> Router -> ModelGateway -> complete_structured."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from aie_core import complete_structured
from aie_core.observability import NoopTracer, Tracer
from router import Router, RouteDecision, Rule, northwind_routes, northwind_rules

HERE = Path(__file__).resolve().parent
sys.path += [str(HERE.parent / "ch04"), str(HERE.parent / "ch05")]   # Chapters 4 and 5, unchanged
from context import BuildResult, ContextItem, Trust  # noqa: E402
from prompts import PromptRef, PromptRegistry  # noqa: E402

PROMPTS = PromptRegistry.from_directory(HERE / "prompt_files")
HINT_ROUTES = {"small": "small_first", "medium": "general", "large": "reasoning"}


class GroundedAnswer(BaseModel):
    answer: str
    citations: list[str]


@dataclass
class Composed:
    answer: GroundedAnswer
    prompt: PromptRef          # which contract produced it
    context: BuildResult       # what the model saw, and what was dropped and why
    route: RouteDecision       # which model, chosen how
    served_by: str


def hinted_router(catalog, clients, **kwargs) -> Router:
    """Policy rules first; a prompt's tier hint decides only when no policy rule matched."""
    hints = [Rule(f"prompt_hint_{tier}", route, lambda req, need, t=tier: req.metadata.get("prompt.tier") == t)
             for tier, route in HINT_ROUTES.items()]
    return Router(catalog, clients, northwind_routes(), default_route="general",
                  rules=[*northwind_rules(), *hints], **kwargs)


class _RouterClient:
    """complete_structured expects an LLMClient; keep every RoutedCompletion for the result."""
    provider = "router"

    def __init__(self, router: Router) -> None:
        self.router, self.routed = router, []

    def complete(self, req):
        self.routed.append(self.router.complete(req))
        return self.routed[-1].completion


def answer(question: str, evidence: list[ContextItem], scope, *, builder, router: Router,
           tracer: Tracer | None = None, **request_metadata) -> Composed:
    prompt = PROMPTS.get("assist.grounded", "prod").render({})               # 1. versioned contract
    items = [ContextItem(kind="instructions", content=m.text, source_id=f"prompt:{prompt.ref}",
                         trust=Trust.TRUSTED, pinned=True) for m in prompt.messages]
    items += [*evidence, ContextItem(kind="query", content=question, source_id="user:request")]
    with (tracer or NoopTracer()).span("assist.answer", **prompt.ref.span_attributes()):
        built = builder.build(items, scope)                                   # 2. budget, order, labels
        req = prompt.to_request(messages=built.messages)
        req = req.model_copy(update={"metadata": {**req.metadata, **request_metadata,
                                                  "prompt.tier": prompt.spec.model_hints.tier}})
        client = _RouterClient(router)                                         # 3-4. route, then gateway
        parsed, _ = complete_structured(client, req, GroundedAnswer)           # 5. validate and repair
    last = client.routed[-1]
    return Composed(parsed, prompt.ref, built, last.decision, last.served_by)
