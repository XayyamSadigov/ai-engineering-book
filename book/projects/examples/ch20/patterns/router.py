# path: book/projects/examples/ch20/patterns/router.py
"""Router: classify the request once, then hand it to exactly one specialist agent.

Deterministic rules run first (cheap, auditable); the model classifies only what the rules
cannot, into a closed set of routes; low confidence or an unknown label goes to a fallback
(here: a human queue) instead of to the closest-sounding specialist.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field, create_model

from aie_core.llm.client import LLMClient
from aie_core.llm.errors import MalformedResponseError
from agentkit import Budget, DefinitionOfDone, EventStore

from .common import PatternResult, ask_structured, make_agent


@dataclass
class Specialist:
    name: str
    description: str
    tools: Sequence[Any]
    instructions: str
    budget: Budget = field(default_factory=lambda: Budget(max_steps=5, max_tool_calls=4))
    dod: DefinitionOfDone | None = None


@dataclass(frozen=True)
class Rule:
    """A keyword rule: if `pattern` matches the request, take `route` without asking the model."""

    route: str
    pattern: str

    def matches(self, text: str) -> bool:
        return re.search(self.pattern, text, re.IGNORECASE) is not None


class AgentRouter:
    def __init__(self, llm: LLMClient, specialists: Sequence[Specialist], rules: Sequence[Rule] = (), *,
                 min_confidence: float = 0.6, fallback: str = "human", store: EventStore | None = None) -> None:
        self.llm = llm
        self.specialists = {s.name: s for s in specialists}
        self.rules = list(rules)
        self.min_confidence = min_confidence
        self.fallback = fallback
        self.store = store
        names = tuple(self.specialists) + (fallback,)
        # The label space is closed: the schema itself rejects a route that does not exist.
        self._schema: type[BaseModel] = create_model(
            "RouteDecision",
            route=(Literal[names], ...),  # type: ignore[valid-type]
            confidence=(float, Field(ge=0.0, le=1.0)),
            reason=(str, ""),
        )

    def classify(self, request: str) -> tuple[str, str, float]:
        """Return (route, decided_by, confidence)."""
        for rule in self.rules:
            if rule.matches(request):
                return rule.route, "rule", 1.0
        menu = "\n".join(f"- {s.name}: {s.description}" for s in self.specialists.values())
        system = (f"Classify the employee request into exactly one route.\n{menu}\n- {self.fallback}: "
                  "anything else, or when unsure. Return route, confidence in [0,1], and a short reason.")
        try:
            decision: Any = ask_structured(self.llm, "router", system, request, self._schema)
        except MalformedResponseError:
            return self.fallback, "fallback:malformed", 0.0
        if decision.route != self.fallback and decision.confidence < self.min_confidence:
            return self.fallback, "fallback:low_confidence", decision.confidence
        return decision.route, "model", decision.confidence

    def run(self, request: str, *, principal: dict[str, Any] | None = None) -> PatternResult:
        route, how, confidence = self.classify(request)
        data = {"route": route, "decided_by": how, "confidence": confidence}
        if route == self.fallback:
            return PatternResult("router", False, None, detail=f"handed to {self.fallback} ({how})", data=data)
        spec = self.specialists[route]
        runtime = make_agent(self.llm, spec.tools, role=f"specialist:{spec.name}", instructions=spec.instructions,
                             budget=spec.budget, dod=spec.dod, store=self.store, principal=principal)
        run = runtime.run(request, metadata={"route": route, "decided_by": how})
        return PatternResult("router", run.ok, run.final_answer, [run],
                             detail=run.stop_reason.value if run.stop_reason else "", data=data)


__all__ = ["Specialist", "Rule", "AgentRouter"]
