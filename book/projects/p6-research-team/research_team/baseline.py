# path: book/projects/p6-research-team/research_team/baseline.py
"""The single-agent baseline the team must beat. Same model, same tools, same corpus, same
output contract. `verify=True` bolts the verifier on as a fixed second step: a workflow, not a
team, and the cheapest way to get independent verification without multi-agent coordination."""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from aie_core.llm.client import LLMClient
from aie_core.llm.gateway import PricingTable
from aie_core.observability import NoopTracer, Tracer
from agentkit import EventStore, InMemoryEventStore, JsonlEventStore

from .checks import find_conflicts
from .contracts import AgentUsage, AnswerReport, BudgetSlice, Role, TaskEnvelope
from .corpus import Corpus
from .render import parse_cited_lines, render_answer
from .roles import TOOLS_BY_ROLE, AgentFactory
from .tools import make_research_tools
from .verification import verify_claims


class SingleAgent:
    def __init__(self, llm: LLMClient, corpus: Corpus, *, verify: bool = False,
                 budget: BudgetSlice | None = None, verifier_budget: BudgetSlice | None = None,
                 store: EventStore | None = None, tracer: Tracer | None = None,
                 pricing: PricingTable | None = None, model: str | None = None,
                 log_dir: str | Path | None = None, min_overlap: float = 0.6) -> None:
        self.llm, self.corpus, self.verify = llm, corpus, verify
        self.budget = budget or BudgetSlice(max_steps=14, max_tokens=80_000, max_tool_calls=None)
        self.verifier_budget = verifier_budget or BudgetSlice(max_steps=4, max_tokens=20_000, max_tool_calls=24)
        self.store, self.tracer, self.pricing, self.model = store, tracer or NoopTracer(), pricing, model
        self.log_dir = Path(log_dir) if log_dir else None
        self.min_overlap = min_overlap
        self.tools = make_research_tools(corpus)
        self.architecture = "single+verify" if verify else "single"

    def _env(self, trace_id: str, name: str, role: Role, objective: str, budget: BudgetSlice,
             inputs: dict[str, Any], schema: str, principal: dict[str, Any]) -> TaskEnvelope:
        return TaskEnvelope(task_id=f"{trace_id}.{name}", parent_id=trace_id, trace_id=trace_id, sender="caller",
                            recipient=role, depth=0, objective=objective, inputs=inputs, output_schema=schema,
                            allowed_tools=list(TOOLS_BY_ROLE[role]), budget=budget, principal=principal)

    def ask(self, question: str, principal: dict[str, Any], *, trace_id: str | None = None) -> AnswerReport:
        started = time.perf_counter()
        trace_id = trace_id or uuid.uuid4().hex[:12]
        run_dir = self.log_dir / trace_id if self.log_dir else None
        store = self.store or (JsonlEventStore(run_dir / "agents") if run_dir else InMemoryEventStore())
        factory = AgentFactory(self.llm, self.tools, store, self.tracer, self.pricing, self.model)
        with self.tracer.span("single.run", **{"trace.id": trace_id, "architecture": self.architecture}):
            env = self._env(trace_id, "single", Role.SINGLE, question, self.budget,
                            {"question": question, "catalog": self.corpus.catalog(principal)}, "", principal)
            res, _ = factory.run(env)
            children = [res]
            answer = str((res.output or {}).get("answer") or "")
            claims = parse_cited_lines(answer, source_task=res.task_id)
            report = AnswerReport(architecture=self.architecture, question=question, trace_id=trace_id,
                                  status="complete" if res.ok else "failed", answer=answer,
                                  accepted_claims=claims, children=children)
            if self.verify and res.ok:
                outcome = verify_claims(
                    claims, factory=factory, corpus=self.corpus,
                    make_envelope=lambda inputs: self._env(trace_id, "verify", Role.VERIFIER,
                                                           "Verify each claim against its cited passage.",
                                                           self.verifier_budget, inputs, "VerificationReport",
                                                           principal),
                    admit=lambda e: (True, "ok", e), settle=lambda r: None, principal=principal,
                    min_overlap=self.min_overlap)
                if outcome.envelope is not None:
                    children.append(outcome.envelope)
                conflicts = find_conflicts(outcome.accepted, self.corpus.doc_updated)
                report.accepted_claims, report.rejected_claims, report.conflicts = (
                    outcome.accepted, outcome.rejected, conflicts)
                report.answer = render_answer(question, outcome.accepted, conflicts=conflicts)
                if outcome.degraded:
                    report.status, report.notes = "partial", ["verification degraded to deterministic checks only"]
        report.usage = sum((c.usage for c in children), AgentUsage())
        report.wall_ms = round((time.perf_counter() - started) * 1000, 3)
        return report


__all__ = ["SingleAgent"]
