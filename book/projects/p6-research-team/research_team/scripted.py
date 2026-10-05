# path: book/projects/p6-research-team/research_team/scripted.py
"""An offline stand-in for the model, used by tests, the demo, and the offline benchmark.

It is a policy, not a model: it reads the transcript, decides the next action by fixed rules,
and extracts claims verbatim from passages it has read. Both architectures get the *same*
policy functions (facet selection, extraction, optional fabrication), so offline comparisons
measure coordination structure, token growth, and verification, not model quality. Swap in
`aie_core.make_llm_client()` to compare real behaviour.

`fabricate_every=N` makes extraction change a number in roughly one of every N numeric claims,
keyed by a hash of the sentence so both architectures fabricate the same sentences. It is a
controlled, illustrative hallucination rate for exercising the verifier.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any

from aie_core.llm.providers import FakeLLM
from aie_core.llm.types import Completion, CompletionRequest, Role as MsgRole, ToolCall

from .checks import deterministic_support
from .contracts import Claim, Conflict, Role
from .render import render_answer
from .roles import PROMPTS
from .text import content_terms, numbers, tokens, units_of_text
from .tools import PASSAGE_HEADER, SEARCH_HEADER

# The "world knowledge" a real model brings to planning: which policy areas a phrase touches.
CONCEPTS: dict[str, tuple[str, ...]] = {
    "abroad": ("travel", "remote", "work", "vpn", "remote-access"),
    "travel": ("travel", "booking", "per-diem"),
    "trip": ("travel", "booking", "expense"),
    "conference": ("travel", "expense", "reimbursement"),
    "laptop": ("laptop", "hardware"),
    "lost": ("laptop", "hardware", "incident", "data"),
    "stolen": ("laptop", "hardware", "incident"),
    "confidential": ("data", "classification"),
    "file": ("data", "classification"),
    "customer": ("data", "classification"),
    "claim": ("expense", "reimbursement", "per-diem"),
    "reimburse": ("expense", "reimbursement"),
    "limit": ("expense",),
    "hotel": ("hotel", "travel"),
    "vpn": ("vpn", "remote-access", "networking"),
    "monitor": ("hardware", "equipment"),
    "home": ("remote", "equipment"),
    "remote": ("remote", "hybrid"),
    "ai": ("data", "classification"),
    "assistant": ("data", "classification"),
    "pto": ("pto", "leave", "carryover"),
    "carry": ("pto", "carryover"),
    # generic intent words: what a model infers the asker cares about
    "fast": ("hour", "immediately", "within"),
    "when": ("within", "deadline"),
    "need": ("require", "must", "approval"),
    "before": ("advance", "approval", "check"),
    "report": ("report", "incident"),
    "fix": ("fix", "error"),
}


GENERIC = frozenset(tokens("northwind focus say says policy runbook document company employee"))


def expand(text: str) -> set[str]:
    terms = set(tokens(text)) - GENERIC
    for t in list(terms):
        for c in CONCEPTS.get(t, ()):
            terms.update(tokens(c))
    return terms


def facets(question: str, catalog: list[dict[str, Any]], max_facets: int = 4) -> list[dict[str, Any]]:
    """Pick the policy areas a question touches from the catalog (title words and tags)."""
    q = expand(question)
    scored = []
    for d in catalog:
        meta = set(tokens(d["title"])) | {t for tag in d.get("tags", []) for t in tokens(tag)}
        meta -= {"policy", "runbook", "northwind", "faq"}
        score = len(q & meta)
        if score:
            scored.append((score, d["doc_id"], d))
    if not scored:
        return []
    scored.sort(key=lambda x: (-x[0], x[1]))
    top = scored[0][0]
    return [d for s, _, d in scored if s >= max(2, top * 0.5)][:max_facets] or [scored[0][2]]


def facet_query(question: str, facet: dict[str, Any]) -> str:
    """The subquestion text for one facet; the planner and the single agent use the same wording."""
    return f"{question} Focus: what the {facet['title']} says." if facet.get("title") else question


def _fabricate(text: str, every: int) -> str:
    if every <= 0 or not numbers(text):
        return text
    if int(hashlib.sha1(text.encode()).hexdigest(), 16) % every:
        return text
    return re.sub(r"\d+", lambda m: str(int(m.group()) + 7), text, count=1)


def extract_claims(passages: list[tuple[str, str]], query: str, *, per_passage: int = 3,
                   fabricate_every: int = 0) -> list[dict[str, Any]]:
    """Pick the sentences that best match the query from each passage; quote them verbatim."""
    q = expand(query)
    out: list[dict[str, Any]] = []
    for pid, text in passages:
        units = units_of_text(text)
        ranked = sorted(((len(q & content_terms(u)), -i, u) for i, u in enumerate(units)), reverse=True)
        for score, _, unit in ranked[:per_passage]:
            if score < 2:
                continue
            out.append({"text": _fabricate(unit, fabricate_every),
                        "evidence": [{"doc_id": pid.split("#")[0], "passage_id": pid, "quote": unit[:600]}]})
    return out


# ----------------------------------------------------------------------------- transcript helpers
def _task(req: CompletionRequest) -> dict[str, Any]:
    user = next(m.text for m in req.messages if m.role is MsgRole.USER)
    body = user.split("TASK\n", 1)[-1].split("\nOUTPUT JSON SCHEMA", 1)[0]
    return json.loads(body)


def _tool_texts(req: CompletionRequest) -> list[str]:
    return [m.text for m in req.messages if m.role is MsgRole.TOOL]


def _passages(texts: list[str]) -> list[tuple[str, str]]:
    out = []
    for t in texts:
        if t.startswith(PASSAGE_HEADER):
            head, _, body = t.partition("\n")
            m = re.search(r"\[([^\]]+)\]", head)
            if m:
                out.append((m.group(1), body))
    return out


def _search_hits(text: str, topic: str = "") -> list[str]:
    """Passage ids from a search result, those from the focus document first (a model told to
    focus on one policy reads that policy's hits before the others)."""
    hits = re.findall(r"^\[([^\]]+)\] (.*)$", text, re.MULTILINE)
    if topic:
        hits = [h for h in hits if h[1].startswith(topic)] + [h for h in hits if not h[1].startswith(topic)]
    return [pid for pid, _ in hits]


def _role(req: CompletionRequest) -> Role:
    system = next((m.text for m in req.messages if m.role is MsgRole.SYSTEM), "")
    for role, prompt in PROMPTS.items():
        if system.startswith(prompt):
            return role
    raise ValueError("scripted model: unknown role prompt")


class ScriptedPolicy:
    """Callable for FakeLLM(handler=...). Thread-safe: it keeps no mutable state."""

    def __init__(self, *, fabricate_every: int = 0, batch_tool_calls: bool = False, reads_per_search: int = 3) -> None:
        self.fabricate_every = fabricate_every
        self.batch = batch_tool_calls
        self.reads = reads_per_search

    def __call__(self, req: CompletionRequest) -> str | list[ToolCall]:
        role = _role(req)
        task = _task(req)
        step = sum(1 for m in req.messages if m.role is MsgRole.ASSISTANT)
        return getattr(self, role.value)(req, task, step)

    # ------------------------------------------------------------------ roles
    def planner(self, req: CompletionRequest, task: dict[str, Any], step: int) -> str:
        inputs = task["inputs"]
        chosen = facets(inputs["question"], inputs["catalog"], max_facets=int(inputs.get("max_subquestions", 4)))
        subs = [{"id": f"sq{i + 1}", "question": facet_query(inputs["question"], d), "topic": d["title"],
                 "rationale": f"catalog match on {d['doc_id']}"} for i, d in enumerate(chosen)]
        if not subs:
            subs = [{"id": "sq1", "question": inputs["question"], "rationale": "no catalog match"}]
        return json.dumps({"subquestions": subs})

    def researcher(self, req: CompletionRequest, task: dict[str, Any], step: int) -> str | list[ToolCall]:
        texts = _tool_texts(req)
        objective = task["objective"]
        if not texts:
            return [ToolCall(id=f"s{step}", name="search_docs", arguments={"query": objective})]
        passages = _passages(texts)
        if not passages:
            hits = _search_hits(texts[-1], task["inputs"].get("topic", ""))[: self.reads]
            if hits:
                return [ToolCall(id=f"r{step}.{i}", name="read_passage", arguments={"passage_id": h})
                        for i, h in enumerate(hits)]
        claims = extract_claims(passages, objective, fabricate_every=self.fabricate_every)
        gaps = [] if claims else ["no passage in the corpus answers this subquestion"]
        return json.dumps({"subquestion": objective, "claims": claims, "gaps": gaps})

    def verifier(self, req: CompletionRequest, task: dict[str, Any], step: int) -> str | list[ToolCall]:
        claims = task["inputs"]["claims"]
        texts = _tool_texts(req)
        read = dict(_passages(texts))
        wanted = list(dict.fromkeys(e["passage_id"] for c in claims for e in c["evidence"]))
        if not texts:
            return [ToolCall(id=f"v{step}.{i}", name="read_passage", arguments={"passage_id": p})
                    for i, p in enumerate(wanted)]
        verdicts = []
        for c in claims:
            results = [deterministic_support(c["text"], read[e["passage_id"]]) for e in c["evidence"]
                       if e["passage_id"] in read]
            ok = any(r[0] for r in results)
            verdicts.append({"claim_id": c["claim_id"], "supported": ok,
                             "reason": results[0][1] if results else "cited passage could not be read"})
        return json.dumps({"verdicts": verdicts})

    def synthesizer(self, req: CompletionRequest, task: dict[str, Any], step: int) -> str:
        inputs = task["inputs"]
        claims = [Claim.model_validate(c) for c in inputs["claims"]]
        conflicts = [Conflict.model_validate(x) for x in inputs.get("conflicts", [])]
        return render_answer(inputs["question"], claims, conflicts=conflicts, gaps=inputs.get("gaps", []),
                             skipped=inputs.get("skipped", []), sections=inputs.get("sections"))

    def single(self, req: CompletionRequest, task: dict[str, Any], step: int) -> str | list[ToolCall]:
        """One agent, one context. Sequential mode: per facet, one search step then one read step.
        Batched mode: all searches in one step, all reads in the next."""
        inputs = task["inputs"]
        question = inputs["question"]
        chosen = facets(question, inputs["catalog"]) or [{"title": "", "doc_id": "", "tags": []}]
        texts = _tool_texts(req)
        searches = [t for t in texts if t.startswith(SEARCH_HEADER)]
        read_ids = {pid for pid, _ in _passages(texts)}
        if self.batch:
            if not searches:
                return [ToolCall(id=f"s{step}.{i}", name="search_docs", arguments={"query": facet_query(question, f)})
                        for i, f in enumerate(chosen)]
            if not read_ids:
                ids = list(dict.fromkeys(h for f, s in zip(chosen, searches)
                                         for h in _search_hits(s, f["title"])[: self.reads]))
                if ids:
                    return [ToolCall(id=f"r{step}.{i}", name="read_passage", arguments={"passage_id": p})
                            for i, p in enumerate(ids)]
        else:
            last = texts[-1] if texts else ""
            if last.startswith(SEARCH_HEADER):
                topic = chosen[len(searches) - 1]["title"]
                ids = [h for h in _search_hits(last, topic)[: self.reads] if h not in read_ids]
                if ids:
                    return [ToolCall(id=f"r{step}.{i}", name="read_passage", arguments={"passage_id": p})
                            for i, p in enumerate(ids)]
            if len(searches) < len(chosen):
                f = chosen[len(searches)]
                return [ToolCall(id=f"s{step}", name="search_docs", arguments={"query": facet_query(question, f)})]
        read = dict(_passages(texts))
        claims = []
        for f, search in zip(chosen, searches):   # per facet: only the passages read for it, same query
            ids = [h for h in _search_hits(search, f["title"])[: self.reads] if h in read]
            claims.extend(extract_claims([(p, read[p]) for p in ids], facet_query(question, f),
                                         fabricate_every=self.fabricate_every))
        seen, lines = set(), []
        for c in claims:
            if c["text"] in seen:
                continue
            seen.add(c["text"])
            lines.append(f"- {c['text']} [{c['evidence'][0]['passage_id']}]")
        return "\n".join(lines) if lines else "The documents I can access do not answer this question."


class SimulatedLatencyLLM:
    """Wraps an LLMClient and sleeps in proportion to tokens, so offline runs show how
    parallelism and context growth affect wall-clock time. Coefficients are illustrative."""

    def __init__(self, inner: Any, *, base_ms: float = 40.0, per_input_token_ms: float = 0.01,
                 per_output_token_ms: float = 0.4, scale: float = 1.0) -> None:
        self.inner = inner
        self.provider = getattr(inner, "provider", "simulated")
        self.base_ms, self.in_ms, self.out_ms, self.scale = base_ms, per_input_token_ms, per_output_token_ms, scale

    def complete(self, req: CompletionRequest) -> Completion:
        c = self.inner.complete(req)
        ms = (self.base_ms + c.usage.input_tokens * self.in_ms + c.usage.output_tokens * self.out_ms) * self.scale
        if ms > 0:
            time.sleep(ms / 1000)
        return c.model_copy(update={"latency_ms": round(ms, 3)})

    def stream(self, req: CompletionRequest):  # pragma: no cover - not used by the loop
        return self.inner.stream(req)

    async def acomplete(self, req: CompletionRequest) -> Completion:  # pragma: no cover
        return self.complete(req)

    async def astream(self, req: CompletionRequest):  # pragma: no cover
        async for ev in self.inner.astream(req):
            yield ev


def offline_llm(*, fabricate_every: int = 0, batch_tool_calls: bool = False, latency_scale: float = 0.0) -> Any:
    fake = FakeLLM(handler=ScriptedPolicy(fabricate_every=fabricate_every, batch_tool_calls=batch_tool_calls),
                   model="scripted-model")
    return SimulatedLatencyLLM(fake, scale=latency_scale) if latency_scale > 0 else fake


__all__ = ["CONCEPTS", "ScriptedPolicy", "SimulatedLatencyLLM", "expand", "extract_claims", "facets", "offline_llm"]
