# path: book/projects/p6-research-team/research_team/team.py
"""The research team: supervisor decomposes, researchers run in parallel, a verifier checks
every claim, the supervisor synthesizes. Coordination is plain code; judgment is in agents.

    plan (agent) -> admit + dispatch researchers (agents, parallel) -> verify (agent + guard)
        -> optional follow-up round for empty subquestions -> detect conflicts -> synthesize (agent)
"""
from __future__ import annotations

import contextvars
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aie_core.llm.client import LLMClient
from aie_core.llm.gateway import PricingTable
from aie_core.observability import NoopTracer, Tracer
from agentkit import EventStore, InMemoryEventStore, JsonlEventStore, RunResult

from .checks import CITATION, find_conflicts
from .contracts import (
    AgentUsage, AnswerReport, BudgetSlice, Claim, Plan, RejectedClaim, ResearchFindings, ResultEnvelope, Role,
    SubQuestion, TaskEnvelope, TaskStatus,
)
from .corpus import Corpus
from .ledger import BudgetLedger, TeamBudget, TeamLog
from .render import claim_key, render_answer, short_label
from .text import numbers
from .roles import TOOLS_BY_ROLE, AgentFactory, skipped
from .tools import make_research_tools, observed_passage_ids
from .verification import verify_claims


@dataclass
class TeamConfig:
    budget: TeamBudget = field(default_factory=TeamBudget)
    planner: BudgetSlice = field(default_factory=lambda: BudgetSlice(max_steps=3, max_tokens=8_000, max_tool_calls=0))
    verifier: BudgetSlice = field(default_factory=lambda: BudgetSlice(max_steps=4, max_tokens=20_000,
                                                                      max_tool_calls=24))
    synthesizer: BudgetSlice = field(default_factory=lambda: BudgetSlice(max_steps=3, max_tokens=10_000,
                                                                         max_tool_calls=0))
    verify_reserve_tokens: int = 20_000
    max_rounds: int = 2                 # round 2 retries subquestions that produced no accepted claim
    min_overlap: float = 0.6            # deterministic guard threshold


class ResearchTeam:
    architecture = "team"

    def __init__(self, llm: LLMClient, corpus: Corpus, *, config: TeamConfig | None = None,
                 store: EventStore | None = None, tracer: Tracer | None = None,
                 pricing: PricingTable | None = None, model: str | None = None,
                 log_dir: str | Path | None = None) -> None:
        self.llm, self.corpus = llm, corpus
        self.config = config or TeamConfig()
        self.store, self.tracer, self.pricing, self.model = store, tracer or NoopTracer(), pricing, model
        self.log_dir = Path(log_dir) if log_dir else None
        self.tools = make_research_tools(corpus)
        self.last_log: TeamLog | None = None
        self.last_runs: dict[str, RunResult] = {}

    # ------------------------------------------------------------------ public
    def ask(self, question: str, principal: dict[str, Any], *, trace_id: str | None = None) -> AnswerReport:
        started = time.perf_counter()
        trace_id = trace_id or uuid.uuid4().hex[:12]
        run_dir = self.log_dir / trace_id if self.log_dir else None
        log = TeamLog(trace_id, run_dir / "team.jsonl" if run_dir else None)
        store = self.store or (JsonlEventStore(run_dir / "agents") if run_dir else InMemoryEventStore())
        factory = AgentFactory(self.llm, self.tools, store, self.tracer, self.pricing, self.model)
        ledger = BudgetLedger(self.config.budget)
        self.last_log, self.last_runs = log, {}
        run = _Run(self, question, principal, trace_id, factory, ledger, log)
        with self.tracer.span("team.run", **{"trace.id": trace_id, "question": question[:200]}) as root:
            run.root_span = root.span_id
            log.emit("team_started", span_id=root.span_id,
                     data={"question": question, "budget": self.config.budget.model_dump()})
            report = run.execute()
            root.set_attribute("status", report.status)
            root.set_attribute("tokens", report.usage.total_tokens)
            root.set_attribute("children", ledger.children)
        report.wall_ms = round((time.perf_counter() - started) * 1000, 3)
        log.emit("team_finished", data={"status": report.status, "usage": report.usage.model_dump(),
                                        "ledger": ledger.snapshot(), "wall_ms": report.wall_ms})
        return report


class _Run:
    """State of one team run. Lives for one ask() call; never shared across threads except the
    ledger and log, which lock internally."""

    def __init__(self, team: ResearchTeam, question: str, principal: dict[str, Any], trace_id: str,
                 factory: AgentFactory, ledger: BudgetLedger, log: TeamLog) -> None:
        self.team, self.cfg = team, team.config
        self.question, self.principal, self.trace_id = question, principal, trace_id
        self.factory, self.ledger, self.log = factory, ledger, log
        self.root_span: str | None = None
        self.duplicate_claims = 0
        self.children: list[ResultEnvelope] = []
        self.notes: list[str] = []

    # ------------------------------------------------------------------ helpers
    def envelope(self, task_id: str, role: Role, objective: str, budget: BudgetSlice, *, depth: int,
                 inputs: dict[str, Any] | None = None, schema: str = "", parent_span: str | None = None,
                 constraints: list[str] | None = None) -> TaskEnvelope:
        return TaskEnvelope(task_id=f"{self.trace_id}.{task_id}", parent_id=self.trace_id, trace_id=self.trace_id,
                            parent_span_id=parent_span or self.root_span, sender="supervisor", recipient=role,
                            depth=depth, objective=objective, inputs=inputs or {}, output_schema=schema,
                            allowed_tools=list(TOOLS_BY_ROLE[role]), budget=budget, principal=self.principal,
                            constraints=constraints or [])

    def admit(self, env: TaskEnvelope, *, child: bool) -> tuple[bool, str, TaskEnvelope]:
        a = self.ledger.admit(env, counts_as_child=child)
        if not a.admitted:
            self.log.emit("spawn_refused", task_id=env.task_id, parent_id=env.parent_id,
                          data={"reason": a.reason, "role": env.recipient.value, "objective": env.objective,
                                "ledger": self.ledger.snapshot()})
            return False, a.reason, env
        granted = env.model_copy(update={"budget": a.granted})
        self.log.emit("task_dispatched", task_id=env.task_id, parent_id=env.parent_id, span_id=env.parent_span_id,
                      data={"role": env.recipient.value, "objective": env.objective,
                            "budget": a.granted.model_dump() if a.granted else None, "depth": env.depth})
        return True, "ok", granted

    def finish(self, env_result: ResultEnvelope, run: RunResult | None = None) -> None:
        self.ledger.settle(env_result.task_id, env_result.usage.total_tokens, env_result.usage.cost_usd)
        self.children.append(env_result)
        if run is not None:
            self.team.last_runs[env_result.task_id] = run
        self.log.emit("task_finished", task_id=env_result.task_id, parent_id=env_result.parent_id,
                      run_id=env_result.run_id,
                      data={"role": env_result.sender.value, "status": env_result.status.value,
                            "stop_reason": env_result.stop_reason, "usage": env_result.usage.model_dump(),
                            "errors": [e.model_dump() for e in env_result.errors]})

    def run_agent(self, env: TaskEnvelope) -> ResultEnvelope:
        result, run = self.factory.run(env)
        self.finish(result, run)
        return result

    # ------------------------------------------------------------------ phases
    def execute(self) -> AnswerReport:
        plan = self.plan()
        self.ledger.hold("synthesis", self.cfg.synthesizer.max_tokens)
        accepted: list[Claim] = []
        rejected: list[RejectedClaim] = []
        gaps: list[str] = []
        labels: dict[str, str] = {}
        pending: list[SubQuestion] = list(plan.subquestions)
        degraded = False
        seen_claims: dict[str, str] = {}
        verified_ok: set[str] = set()
        answered: dict[str, bool] = {}          # base subquestion id -> has at least one verified claim
        base_of = {sq.id: sq.id for sq in plan.subquestions}   # follow-up id -> the planner's id
        for rnd in range(1, self.cfg.max_rounds + 1):
            if not pending:
                break
            self.ledger.hold("verify", self.cfg.verify_reserve_tokens)
            results = self.dispatch(pending, rnd)
            claims: list[Claim] = []
            by_sq: dict[str, list[str]] = {}
            duplicates = 0
            for sq, res in results:
                labels[res.task_id] = sq.topic or short_label(sq.question)
                if res.ok and res.output is not None:
                    findings = ResearchFindings.model_validate(res.output)
                    gaps.extend(f"{short_label(sq.question, 60)}: {g}" for g in findings.gaps)
                    for i, c in enumerate(self.observed_only(res, findings.claims, rejected)):
                        claim = c.model_copy(update={"claim_id": f"{res.task_id}.c{i}", "source_task": res.task_id})
                        key = claim_key(claim)
                        if key in seen_claims:          # duplicated work: another worker already found it
                            duplicates += 1
                            by_sq.setdefault(sq.id, []).append(seen_claims[key])
                            continue
                        seen_claims[key] = claim.claim_id
                        claims.append(claim)
                        by_sq.setdefault(sq.id, []).append(claim.claim_id)
            if duplicates:
                self.log.emit("duplicate_work", data={"round": rnd, "duplicate_claims": duplicates,
                                                      "unique_claims": len(claims)})
                self.duplicate_claims += duplicates
            self.ledger.release("verify")
            outcome = self.verify(claims, rnd)
            degraded = degraded or outcome.degraded
            accepted.extend(outcome.accepted)
            rejected.extend(outcome.rejected)
            verified_ok |= {c.claim_id for c in outcome.accepted}
            for sq, res in results:
                if res.status is not TaskStatus.SKIPPED:
                    base = base_of[sq.id]
                    answered[base] = answered.get(base, False) or bool(set(by_sq.get(sq.id, [])) & verified_ok)
            # follow-up only for subquestions that ran but yielded nothing verified; skipped ones stay skipped
            pending = []
            for sq, res in results:
                if (res.status is TaskStatus.SKIPPED or set(by_sq.get(sq.id, [])) & verified_ok
                        or base_of[sq.id] != sq.id):        # only the planner's subquestions get one follow-up
                    continue
                fid = f"{sq.id[:20]}-f{rnd}"
                while fid in base_of:                        # never collide with a planner id
                    fid = f"{fid[:22]}x"
                base_of[fid] = sq.id
                pending.append(SubQuestion(id=fid, topic=sq.topic, question=sq.question + " Search with different "
                                           "keywords; look for exact rules, deadlines, and amounts."))
        skipped_objectives = [short_label(c.objective, 80) for c in self.children
                              if c.sender is Role.RESEARCHER and c.status is TaskStatus.SKIPPED]
        topics = {sq.id: sq.topic or short_label(sq.question, 60) for sq in plan.subquestions}
        unanswered = [topics.get(base, base) for base, ok in answered.items() if not ok]
        gaps.extend(f"no verified answer for: {t}" for t in unanswered)
        conflicts = find_conflicts(accepted, self.team.corpus.doc_updated)
        for x in conflicts:
            self.log.emit("conflict", data=x.model_dump())
        if degraded:
            self.notes.append("verification degraded to deterministic checks only")
        answer = self.synthesize(accepted, conflicts, gaps, skipped_objectives, labels)
        usage = sum((c.usage for c in self.children), AgentUsage())
        researchers = [c for c in self.children if c.sender is Role.RESEARCHER]
        if not accepted:
            status = "failed"
        elif degraded or any(not c.ok for c in researchers) or skipped_objectives or unanswered or self.notes:
            status = "partial"
        else:
            status = "complete"
        return AnswerReport(architecture=self.team.architecture, question=self.question, trace_id=self.trace_id,
                            status=status, answer=answer, accepted_claims=accepted, rejected_claims=rejected,
                            conflicts=conflicts, gaps=gaps, children=self.children, usage=usage, notes=self.notes,
                            duplicate_claims=self.duplicate_claims)

    def plan(self) -> Plan:
        env = self.envelope("planner", Role.PLANNER, self.question, self.cfg.planner, depth=0, schema="Plan",
                            inputs={"question": self.question, "catalog": self.team.corpus.catalog(self.principal),
                                    "max_subquestions": self.cfg.budget.max_children})
        ok, reason, env = self.admit(env, child=False)
        if ok:
            res = self.run_agent(env)
            if res.ok and res.output is not None:
                plan = Plan.model_validate(res.output)
                self.log.emit("plan_accepted", task_id=env.task_id,
                              data={"subquestions": [s.model_dump() for s in plan.subquestions]})
                return plan
        self.notes.append("planner failed; researching the question as a single task")
        return Plan(subquestions=[SubQuestion(id="sq1", question=self.question)])

    def dispatch(self, subquestions: list[SubQuestion], rnd: int) -> list[tuple[SubQuestion, ResultEnvelope]]:
        with self.team.tracer.span("team.dispatch", **{"trace.id": self.trace_id, "round": rnd,
                                                        "requested": len(subquestions)}) as span:
            admitted: list[tuple[SubQuestion, TaskEnvelope]] = []
            out: list[tuple[SubQuestion, ResultEnvelope]] = []
            for sq in subquestions:                      # admission is sequential and deterministic
                env = self.envelope(f"r{rnd}-{sq.id}", Role.RESEARCHER, sq.question, self.cfg.budget.child,
                                    depth=1, schema="ResearchFindings", parent_span=span.span_id,
                                    inputs={"subquestion_id": sq.id, "topic": sq.topic,
                                            "original_question": self.question},
                                    constraints=["read-only tools", "cite passages you read",
                                                 "report gaps instead of guessing"])
                ok, reason, env = self.admit(env, child=True)
                if ok:
                    admitted.append((sq, env))
                else:
                    res = skipped(env, reason)
                    self.children.append(res)
                    out.append((sq, res))
            span.set_attribute("admitted", len(admitted))
            with ThreadPoolExecutor(max_workers=self.cfg.budget.max_parallel,
                                    thread_name_prefix="researcher") as pool:
                # Copy the context per task so the dispatch span is the current span in each worker
                # thread: aie_core links a span to its parent through a context variable, and a
                # thread pool does not inherit context variables on its own.
                futures = [(sq, pool.submit(contextvars.copy_context().run, self.factory.run, env))
                           for sq, env in admitted]
                for sq, fut in futures:
                    res, run = fut.result()
                    self.finish(res, run)
                    out.append((sq, res))
        order = {sq.id: i for i, sq in enumerate(subquestions)}
        return sorted(out, key=lambda p: order[p[0].id])

    def observed_only(self, res: ResultEnvelope, claims: list[Claim], rejected: list[RejectedClaim]) -> list[Claim]:
        """Zero trust in children: re-check against the child's own event log that every cited
        passage was actually returned by a tool, even though the child's DoD already checked it."""
        run = self.team.last_runs.get(res.task_id)
        seen = set().union(*(observed_passage_ids(o.content) for o in run.state.observations if o.ok)) if run else set()
        kept = []
        for c in claims:
            if all(e.passage_id in seen for e in c.evidence):
                kept.append(c)
            else:
                rejected.append(RejectedClaim(claim=c, reason="evidence not observed by the worker",
                                              rejected_by="deterministic"))
        return kept

    def verify(self, claims: list[Claim], rnd: int):
        def make(inputs: dict[str, Any]) -> TaskEnvelope:
            return self.envelope(f"verify-r{rnd}", Role.VERIFIER, "Verify each claim against its cited passage.",
                                 self.cfg.verifier, depth=1, inputs=inputs, schema="VerificationReport")

        outcome = verify_claims(
            claims, factory=self.factory, corpus=self.team.corpus, make_envelope=make,
            admit=lambda env: self.admit(env, child=False), settle=lambda r: None, principal=self.principal,
            min_overlap=self.cfg.min_overlap)
        if outcome.envelope is not None:
            if outcome.envelope.run_id is None:
                self.children.append(outcome.envelope)
            else:
                self.finish(outcome.envelope)
        self.log.emit("verified", data={"round": rnd, "accepted": [c.claim_id for c in outcome.accepted],
                                        "rejected": [{"claim_id": r.claim.claim_id, "by": r.rejected_by,
                                                      "reason": r.reason} for r in outcome.rejected],
                                        "degraded": outcome.degraded})
        return outcome

    def synthesize(self, accepted: list[Claim], conflicts, gaps: list[str], skipped_objectives: list[str],
                   labels: dict[str, str]) -> str:
        self.ledger.release("synthesis")
        fallback = render_answer(self.question, accepted, conflicts=conflicts, gaps=gaps, skipped=skipped_objectives,
                                 sections=labels)
        if not accepted:
            return fallback
        env = self.envelope("synth", Role.SYNTHESIZER, self.question, self.cfg.synthesizer, depth=0, inputs={
            "question": self.question, "sections": labels,
            "claims": [c.model_dump(include={"claim_id", "text", "evidence", "source_task"}) for c in accepted],
            "conflicts": [x.model_dump() for x in conflicts], "gaps": gaps, "skipped": skipped_objectives})
        ok, reason, env = self.admit(env, child=False)
        if not ok:
            self.notes.append(f"synthesis not started ({reason}); deterministic rendering used")
            return fallback
        res = self.run_agent(env)
        if res.ok and res.output:
            answer = str(res.output["answer"])
            problem = self.unsupported_line(answer, accepted)
            if problem is None:
                return answer
            self.notes.append(f"synthesis rejected ({problem}); deterministic rendering used")
            return fallback
        self.notes.append(f"synthesis failed ({res.stop_reason}); deterministic rendering used")
        return fallback

    def unsupported_line(self, answer: str, accepted: list[Claim]) -> str | None:
        """The writer may reword verified claims but not add a number: every number on a cited line
        must come from the verified claims behind its citations, and every number on an uncited line
        from some verified claim (or the question). Wording is left to the writer."""
        texts: dict[str, list[str]] = {}
        for c in accepted:
            for e in c.evidence:
                texts.setdefault(e.passage_id, []).append(c.text)
        anywhere = numbers(" ".join(c.text for c in accepted) + " " + self.question)
        for raw in answer.splitlines():
            pids = CITATION.findall(raw)
            text = CITATION.sub("", raw)
            allowed = numbers(" ".join(t for p in pids for t in texts.get(p, []))) if pids else anywhere
            if extra := sorted(numbers(text) - allowed):
                return f"{short_label(text.strip(), 60)!r}: numbers not in verified claims: {extra}"
        return None


__all__ = ["ResearchTeam", "TeamConfig"]
