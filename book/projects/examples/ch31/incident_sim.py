# path: book/projects/examples/ch31/incident_sim.py
"""A synthetic Northwind Assist workload that produces the chapter's incident telemetry.

The retrieval, ranking, and model behavior are simulated (scripted, seeded), but every span
goes through the real instrumentation: AITracer, the stage helpers, aie_core's ModelGateway
with retries and cost accounting, and a JsonlTracer or InMemoryTracer sink. The scenario:

* Before 2026-10-01 09:00 (illustrative): index idx-2026-09-15 built with chunker c2
  (~380-token chunks), prompt rag-answer v7, app 2026.09.4.
* At 09:00 one release ships two changes: an index rebuild with chunker c3 (~900-token
  chunks, meant to "keep sections together"), and a canary of prompt v8 to half of traffic.
* The context packer's 3,000-token evidence budget now fits three chunks instead of six,
  so gold evidence ranked 4-6 is retrieved and reranked, then silently dropped.

Probe traffic (scheduled golden questions with known gold evidence) and a sampled judge
produce eval labels; users produce thumbs up/down that reference `response.id`.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aie_core.llm.errors import RateLimitError
from aie_core.llm.gateway import ModelGateway, PricingTable, RetryPolicy
from aie_core.llm.providers import FakeLLM
from aie_core.llm.types import Completion, CompletionRequest, Message, Usage
from aie_core.observability import InMemoryTracer, JsonlTracer, Tracer

from instrument import (AITracer, CapturePolicy, agent_run, agent_step, context_span, guardrail_span,
                        rerank_span, retrieval_span, tool_span, trace_request, traced_generation)
from semconv import Attr, ErrorClass, SpanName

DEPLOY_AT = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc).timestamp()
START_AT = DEPLOY_AT - 33 * 3600
END_AT = DEPLOY_AT + 6 * 3600
MODEL = "assist-small"
PRICING = PricingTable({MODEL: {"input_per_1m": 0.40, "output_per_1m": 1.60}})  # illustrative prices
CONTEXT_BUDGET = 3000

OLD = {Attr.APP_VERSION: "2026.09.4", Attr.INDEX_VERSION: "idx-2026-09-15", Attr.CHUNKER_VERSION: "c2"}
NEW = {Attr.APP_VERSION: "2026.10.1", Attr.INDEX_VERSION: "idx-2026-10-01", Attr.CHUNKER_VERSION: "c3"}
CHUNK_TOKENS = {"c2": 380, "c3": 900}

SHARED_DOCS = ["hr-pto-policy", "hr-faq", "hr-parental-leave-policy", "hr-expense-policy", "hr-travel-policy",
               "hr-remote-work-policy", "it-vpn-access-runbook", "it-password-reset-runbook", "it-faq",
               "it-laptop-replacement-runbook", "sec-access-control-policy", "it-incident-response-runbook"]
TENANT_DOCS = {"retail": ["prod-retail-pos-overview", "prod-retail-returns-api", "inc-2025-11-pos-outage"],
               "logistics": ["prod-logistics-tracking-api", "prod-logistics-route-planner", "inc-2026-02-tracking-latency"]}


@dataclass(frozen=True)
class Question:
    qid: str
    text: str
    gold: tuple[str, ...]
    tenant: str | None = None  # None: asked by both tenants


QUESTIONS: tuple[Question, ...] = (
    Question("RQ-001", "How many unused PTO days can I carry over, and by when must I use them?", ("hr-pto-policy",)),
    Question("RQ-004", "Am I eligible for parental leave if we are adopting?", ("hr-parental-leave-policy",)),
    Question("RQ-007", "What is the nightly hotel cap for business travel?", ("hr-travel-policy",)),
    Question("RQ-009", "My VPN says certificate expired, what do I do?", ("it-vpn-access-runbook",)),
    Question("RQ-011", "How do I reset my password if MFA is lost?", ("it-password-reset-runbook",)),
    Question("RQ-013", "Can I expense a home office chair?", ("hr-expense-policy",)),
    Question("RQ-015", "How do I process a return without a receipt at the POS?", ("prod-retail-returns-api",), "retail"),
    Question("RQ-018", "Why are tracking updates delayed for EU shipments?", ("prod-logistics-tracking-api",), "logistics"),
)

# gold rank after retrieval: index 0 = rank 1; mass beyond len() means "not in top 8"
GOLD_RANK_WEIGHTS = [0.38, 0.18, 0.12, 0.10, 0.08, 0.06, 0.02, 0.01]


class SimClock:
    def __init__(self, t: float) -> None:
        self.t = t

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds

    def advance_ms(self, ms: float) -> None:
        self.t += ms / 1000


@dataclass
class SimResult:
    tracer: AITracer
    spans: list[Any]
    evals: list[dict[str, Any]] = field(default_factory=list)
    feedback: list[dict[str, Any]] = field(default_factory=list)


class NorthwindSim:
    def __init__(self, seed: int = 7, sink: Tracer | None = None) -> None:
        self.rng = random.Random(seed)
        self.clock = SimClock(START_AT)
        self.sink = sink or InMemoryTracer()
        self.tracer = AITracer(
            self.sink,
            resource={"service.name": "northwind-assist", "deployment.environment": "prod"},
            capture=CapturePolicy(mode="hashed", sample_rate=0.05, sampled_mode="redacted", salt="sim-salt",
                                  tenant_ceiling={"logistics": "hashed"}),
            clock=self.clock.now,
            rng=random.Random(seed + 1),
        )
        self.gateway = ModelGateway(
            FakeLLM(handler=self._model, model=MODEL),
            retry=RetryPolicy(max_attempts=3, base_delay_s=0.4),
            pricing=PRICING, tracer=self.tracer,
            sleep=self.clock.advance, rng=random.Random(seed + 2),
        )
        self.evals: list[dict[str, Any]] = []
        self.feedback: list[dict[str, Any]] = []
        self._n = 0

    # ---------------------------------------------------------------- the scripted model
    def _model(self, req: CompletionRequest) -> Completion:
        sim = req.metadata.get("sim", {})
        if self.rng.random() < 0.01:
            self.clock.advance_ms(self.rng.uniform(80, 200))
            raise RateLimitError("429 from provider", retry_after_s=None)
        in_tokens = sim["input_tokens"]
        if sim.get("kind") == "agent":
            text, out_tokens = f"next: {sim['action']}", self.rng.randint(30, 80)
        else:
            out_tokens = int(self.rng.randint(110, 210) * (1.15 if sim["prompt_version"] == "8" else 1.0))
            text = sim["answer"]
        latency = 250 + 0.08 * in_tokens + 18 * out_tokens + self.rng.uniform(-150, 400)
        self.clock.advance_ms(latency)
        return Completion(message=Message.assistant(text), usage=Usage(input_tokens=in_tokens, output_tokens=out_tokens),
                          finish_reason="stop", model=MODEL, provider="fake", latency_ms=latency)

    # ---------------------------------------------------------------- one RAG request
    def _rag(self, q: Question, tenant: str, traffic: str, versions: dict[str, str], prompt_version: str) -> dict[str, Any]:
        rng, tr = self.rng, self.tracer
        self._n += 1
        response_id = f"resp-{self._n:05d}"
        query = q.text
        if traffic == "user" and rng.random() < 0.08:
            query += f" (reply to {rng.choice(['jane.doe', 'a.kim', 'p.rossi'])}@northwind.example)"
        visible = SHARED_DOCS + TENANT_DOCS[tenant]
        all_versions = {**versions, Attr.PROMPT_ID: "rag-answer", Attr.PROMPT_VERSION: prompt_version, Attr.LLM_MODEL: MODEL}
        with trace_request(tr, route="rag.answer", tenant=tenant, user_id=f"u{rng.randint(1, 900)}", traffic=traffic,
                           response_id=response_id, versions=all_versions) as root:
            with tr.span(SpanName.ROUTER, **{"router.route": "rag.answer", Attr.LLM_MODEL: MODEL}):
                self.clock.advance_ms(rng.uniform(1, 4))

            # retrieval: identical behavior on both indexes (the trap: it is not the culprit)
            with retrieval_span(tr, query, index_version=versions[Attr.INDEX_VERSION], top_k=8) as r:
                distractors = [d for d in visible if d not in q.gold]
                rng.shuffle(distractors)
                ranked = distractors[:8]
                slot = rng.choices(range(len(GOLD_RANK_WEIGHTS) + 1), weights=GOLD_RANK_WEIGHTS + [0.05])[0]
                if slot < 8:
                    ranked = ranked[:slot] + list(q.gold) + ranked[slot:7]
                scores = sorted((rng.uniform(0.35, 0.92) for _ in ranked), reverse=True)
                self.clock.advance_ms(rng.uniform(35, 110))
                r.results(ranked, scores, tenants=["shared" if d in SHARED_DOCS else tenant for d in ranked])

            with rerank_span(tr, model="rerank-small", candidates=len(ranked)) as rr:
                kept = ranked[:6]
                self.clock.advance_ms(rng.uniform(60, 140))
                rr.kept(kept)

            chunk = CHUNK_TOKENS[versions[Attr.CHUNKER_VERSION]]
            with context_span(tr, budget_tokens=CONTEXT_BUDGET) as c:
                included, dropped, used = [], [], 0
                for doc in kept:
                    size = int(chunk * rng.uniform(0.85, 1.15))
                    if used + size <= CONTEXT_BUDGET:
                        included.append(doc)
                        used += size
                    else:
                        dropped.append(doc)
                self.clock.advance_ms(rng.uniform(2, 6))
                c.packed(included, dropped, used)

            gold_in_context = set(q.gold) <= set(included)
            abstained = False
            if gold_in_context:
                correct = rng.random() < 0.95
                cited = list(q.gold) if correct or rng.random() < 0.5 else [included[0]]
                answer = f"Per {cited[0]}: ... [{cited[0]}]"
            elif included and rng.random() > 0.25:
                correct, cited = False, [included[0]]
                answer = f"Per {cited[0]}: ... [{cited[0]}]"
            else:
                correct, cited, abstained = False, [], True
                answer = "I could not find this in the documents you can access."

            req = CompletionRequest(
                model=MODEL,
                messages=[Message.system(f"You are Northwind Assist (rag-answer v{prompt_version}). Cite sources."),
                          Message.user(f"Question: {query}\nEvidence: {', '.join(included)}")],
                metadata={"sim": {"kind": "rag", "input_tokens": 380 + used, "answer": answer,
                                  "prompt_version": prompt_version}},
            )
            traced_generation(tr, self.gateway, req, prompt_id="rag-answer", prompt_version=prompt_version)

            with guardrail_span(tr, "citation_validator", stage="output", policy_version="cv-3") as g:
                self.clock.advance_ms(rng.uniform(1, 3))
                if set(cited) <= set(included):
                    g.decide("allow", "citations resolve to context")
                else:
                    g.decide("block", "citation not in context", error_class=ErrorClass.CITATION_MISMATCH)
            root.set_attribute(Attr.CITATION_IDS, cited)
            root.set_attribute(Attr.ABSTAINED, abstained)
            trace_id, start = root.trace_id, root.start

        return {"trace_id": trace_id, "response_id": response_id, "start": start, "correct": correct,
                "abstained": abstained, "q": q}

    # ---------------------------------------------------------------- one agent request
    def _agent(self, tenant: str, versions: dict[str, str], prompt_version: str) -> dict[str, Any]:
        rng, tr = self.rng, self.tracer
        self._n += 1
        response_id = f"resp-{self._n:05d}"
        looping = rng.random() < 0.03
        plan = ["search_tickets"] * 6 if looping else ["search_tickets", "get_service_status", "draft_reply"]
        all_versions = {**versions, Attr.PROMPT_ID: "incident-agent", Attr.PROMPT_VERSION: "3", Attr.LLM_MODEL: MODEL}
        with trace_request(tr, route="agent.incident", tenant=tenant, user_id=f"u{rng.randint(1, 900)}",
                           response_id=response_id, versions=all_versions) as root:
            with agent_run(tr, "incident-research", max_steps=6) as run:
                tokens = 0
                for i, action in enumerate(plan, start=1):
                    with agent_step(tr, i, action, tokens_cumulative=tokens):
                        req = CompletionRequest(model=MODEL, messages=[Message.user(f"step {i}")],
                                                metadata={"sim": {"kind": "agent", "action": action,
                                                                  "input_tokens": 900 + 350 * i}})
                        c = traced_generation(tr, self.gateway, req, prompt_id="incident-agent", prompt_version="3")
                        tokens += c.usage.input_tokens + c.usage.output_tokens
                        args = {"query": "EU tracking delay", "status": "open"} if action == "search_tickets" else \
                            {"service": "tracking-api"} if action == "get_service_status" else {"ticket": "T-1042"}
                        with tool_span(tr, action, args, call_id=f"call-{i}") as t:
                            self.clock.advance_ms(rng.uniform(40, 260))
                            t.result({"items": rng.randint(0, 5)} if action != "draft_reply" else "Draft saved.")
                if looping:
                    run.stop("max_steps", error_class=ErrorClass.LOOP)
                else:
                    run.stop("done")
            trace_id, start = root.trace_id, root.start
        return {"trace_id": trace_id, "response_id": response_id, "start": start, "correct": not looping,
                "abstained": False, "q": None}

    # ---------------------------------------------------------------- labels
    def _label(self, out: dict[str, Any], traffic: str) -> None:
        rng = self.rng
        if traffic == "probe":
            self.evals.append({"trace_id": out["trace_id"], "case_id": out["q"].qid, "name": "answer_correct",
                               "passed": out["correct"], "score": float(out["correct"]), "source": "probe",
                               "gold_ids": list(out["q"].gold), "ts": out["start"] + 30})
            return
        if out["q"] is not None and rng.random() < 0.10:  # sampled online judge, imperfect on purpose
            verdict = out["correct"] if rng.random() > 0.08 else not out["correct"]
            self.evals.append({"trace_id": out["trace_id"], "name": "judge_groundedness", "passed": verdict,
                               "score": float(verdict), "source": "judge", "ts": out["start"] + 120})
        if rng.random() < 0.30:
            bad = (not out["correct"]) or out["abstained"]
            negative = rng.random() < (0.45 if bad else 0.04)
            self.feedback.append({"response_id": out["response_id"], "value": -1 if negative else 1,
                                  "reason": rng.choice(["wrong", "outdated", "no answer"]) if negative else None,
                                  "ts": out["start"] + rng.uniform(5, 600)})

    # ---------------------------------------------------------------- the scenario
    def run(self, before: int = 600, after: int = 480) -> SimResult:
        for phase, count, t0, t1 in (("before", before, START_AT, DEPLOY_AT), ("after", after, DEPLOY_AT, END_AT)):
            step = (t1 - t0) / count
            for i in range(count):
                self.clock.t = t0 + i * step + self.rng.uniform(0, step * 0.5)
                tenant = "retail" if self.rng.random() < 0.6 else "logistics"
                versions = OLD if phase == "before" else NEW
                prompt_version = "7" if phase == "before" or self.rng.random() < 0.5 else "8"
                if tenant == "logistics" and self.rng.random() < 0.12:
                    self._label(self._agent(tenant, versions, prompt_version), "user")
                    continue
                traffic = "probe" if self.rng.random() < 0.25 else "user"
                pool = [q for q in QUESTIONS if q.tenant in (None, tenant)]
                out = self._rag(self.rng.choice(pool), tenant, traffic, versions, prompt_version)
                self._label(out, traffic)
        spans = list(getattr(self.sink, "spans", []))
        return SimResult(self.tracer, spans, self.evals, self.feedback)


def simulate(seed: int = 7, before: int = 600, after: int = 480) -> SimResult:
    return NorthwindSim(seed).run(before, after)


def write_incident(out_dir: str | Path, seed: int = 7) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"traces": out / "traces.jsonl", "evals": out / "eval_results.jsonl", "feedback": out / "feedback.jsonl"}
    for p in paths.values():
        p.unlink(missing_ok=True)
    result = NorthwindSim(seed, sink=JsonlTracer(paths["traces"])).run()
    for key in ("evals", "feedback"):
        with paths[key].open("w", encoding="utf-8") as f:
            for row in getattr(result, key):
                f.write(json.dumps(row) + "\n")
    return paths


if __name__ == "__main__":
    import sys

    written = write_incident(sys.argv[1] if len(sys.argv) > 1 else "data/incident")
    for name, path in written.items():
        print(f"{name}: {path}")
