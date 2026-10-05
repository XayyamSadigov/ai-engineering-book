# path: book/projects/examples/ch10/failure_modes.py
"""Reproduce the seven naive-RAG failure modes of Chapter 10, one function each.

Every demo returns a FailureCase: which stage failed, what the pipeline produced, the
deterministic signal that detects it, and (where one exists inside this chapter's scope)
whether a minimal fix removes it. Models are scripted with FakeLLM so each run is identical;
the scripts imitate behaviors real models show, and the detectors never trust the model.

Run:  python failure_modes.py
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date

from aie_core import CompletionRequest, Message
from aie_core.llm.providers import FakeLLM

from minimal_rag import (
    CITATION_RE, FIXTURES_DIR, Doc, Hit, MinimalRAG, build_request, chunk_fixed, default_embedder,
    extractive_fake_handler, load_docs, pack_evidence,
)


CLAIM_RE = re.compile(r"\b\d[\d,.]*\s+[a-z]+")   # "14 calendar", "250 usd": a number and its unit


@dataclass
class FailureCase:
    name: str
    stage: str
    question: str
    observed: str
    detected: bool
    signal: str
    fixed: bool | None = None          # None: the fix belongs to a later chapter
    details: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- helpers
def make_doc(doc_id: str, body: str, *, version: str = "1.0", updated: str = "2026-01-01",
             acl: tuple[str, ...] = ("all",), title: str | None = None) -> Doc:
    return Doc(id=doc_id, title=title or doc_id, version=version, updated_at=date.fromisoformat(updated),
               owner="Chapter 10 fixture", tenant="shared", acl_groups=list(acl), tags=["fixture"],
               path=f"fixture://{doc_id}", body=body, sha256=hashlib.sha256(body.encode()).hexdigest())


def corpus_rag(llm: FakeLLM | None = None, extra: list[Doc] | None = None, k: int = 4) -> MinimalRAG:
    rag = MinimalRAG(llm or FakeLLM(handler=extractive_fake_handler), default_embedder(), k=k)
    rag.ingest(load_docs() + (extra or []))
    return rag


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[*`]", "", text).lower().split())


def numeric_claims(text: str) -> list[str]:
    return CLAIM_RE.findall(_norm(re.sub(r"\[[^\]]+\]", "", text)))


def unsupported_claims(answer: str, evidence_texts: list[str]) -> list[str]:
    """Number+unit phrases in the answer found in no evidence text: a cheap grounding check."""
    evidence = _norm(" ".join(evidence_texts))
    return [c for c in numeric_claims(answer) if c not in evidence]


NAIVE_SYSTEM = "You are a helpful assistant. Answer the question based on the context."


def naive_request(question: str, hits: list[Hit]) -> CompletionRequest:
    """The tutorial prompt: unlabeled context, no abstention path, no citation contract."""
    context = "\n\n".join(h.chunk.text for h in hits)
    return CompletionRequest(messages=[Message.system(NAIVE_SYSTEM),
                                       Message.user(f"Context:\n{context}\n\nQuestion: {question}")])


# --------------------------------------------------------------------------- 1. chunking
def wrong_chunk_boundary() -> FailureCase:
    fact = "within 10 working days of receiving the replacement"
    body = ("# Device returns\n\nEmployees who receive a replacement laptop keep the old device only for "
            "data transfer. The old laptop must be handed back to the service desk " + fact + ". "
            "Late returns are reported to the manager.\n\n## Accessories\n\nDocks, headsets and monitors "
            "are ordered in Beacon under IT Hardware and arrive within one week.")
    size = body.index("10 working")                    # the boundary lands mid-sentence
    doc = make_doc("it-device-returns", body)
    question = "Within how many days must the old laptop be handed back to the service desk?"

    def top_hit(overlap: int) -> Hit:
        rag = MinimalRAG(FakeLLM(handler=extractive_fake_handler), default_embedder(), k=1)
        rag.index.add(chunk_fixed(doc, size=size, overlap=overlap))
        return rag.index.search(question, k=1)[0]

    naive, fixed = top_hit(0), top_hit(80)
    return FailureCase(
        "wrong chunk boundary", "chunk", question,
        observed=f"top chunk {naive.chunk.id} ends with {naive.chunk.text[-40:]!r}",
        detected="10 working days" not in naive.chunk.text,
        signal="gold fact present in corpus but split across chunks; the retrieved chunk lacks it",
        fixed="10 working days" in fixed.chunk.text,
        details={"chunk_size": size, "fix": "overlap=80 (structure-aware chunking in Chapter 11)"},
    )


# --------------------------------------------------------------------------- 2. coverage
UNCOVERED_QUESTION = "How many weeks of paid sabbatical do employees get after seven years?"


def missing_evidence() -> FailureCase:
    rag = corpus_rag()
    hits = rag.index.search(UNCOVERED_QUESTION, k=4, user_groups={"all"})
    covered = any(re.search(r"\bsabbatical", c.text, re.I) for c in rag.index.chunks)
    return FailureCase(
        "missing evidence", "ingest", UNCOVERED_QUESTION,
        observed=f"retriever still returned {len(hits)} chunks, top={hits[0].chunk.id} score={hits[0].score:.3f}",
        detected=not covered and len(hits) == 4,
        signal="no indexed chunk mentions the topic, yet top-k is always full; scores are relative, not calibrated",
        fixed=None,
        details={"top_ids": [h.chunk.id for h in hits], "fix": "coverage tests on the gold set (Chapter 14)"},
    )


# --------------------------------------------------------------------------- 3. ranking
def distractor() -> FailureCase:
    rag = corpus_rag(k=4)
    question = "What is the return window for my old laptop?"
    hits = rag.index.search(question, k=4, user_groups={"all"})
    gold = "it-laptop-replacement-runbook"
    gold_rank = next((i + 1 for i, h in enumerate(hits) if h.chunk.doc_id == gold), None)
    # Tight budget (k=1) or a model that anchors on the first block turns rank 2 into a wrong answer.
    tight = MinimalRAG(FakeLLM(handler=extractive_fake_handler), rag.index.embedder, k=1)
    tight.index = rag.index
    answer = tight.answer(question, user_groups={"all"})
    return FailureCase(
        "distractor outranks evidence", "retrieve/rank", question,
        observed=f"rank 1 = {hits[0].chunk.id}; gold doc at rank {gold_rank}; k=1 answer: {answer.text[:70]!r}",
        detected=hits[0].chunk.doc_id != gold and gold_rank is not None and gold_rank > 1,
        signal="MRR < 1 on a gold question; top chunk from another domain (retail product API, not IT)",
        fixed=None,
        details={"ranking": [(h.chunk.id, round(h.score, 3)) for h in hits], "fix": "hybrid + rerank (Chapter 12)"},
    )


# --------------------------------------------------------------------------- 4. freshness
def stale_version() -> FailureCase:
    rag = corpus_rag()
    question = "How many unused PTO days can I carry over into next year?"
    answer = rag.answer(question, user_groups={"all"})
    prompt = rag.llm.last_request.messages[-1].text
    versions = {h.chunk.doc_id: (h.chunk.version, h.chunk.updated_at) for h in answer.evidence}
    return FailureCase(
        "stale document version", "ingest/pack", question,
        observed=f"answer: {answer.text[:110]!r}",
        detected="5 unused days" in answer.text and "hr-pto-policy" in versions,
        signal="evidence from two docs that disagree; the older one (hr-faq 2025-06-10) wins; "
               "the prompt carries no version or date",
        fixed=None,
        details={"versions_in_context": versions, "prompt_has_version_metadata": 'version="' in prompt,
                 "fix": "version metadata in evidence + supersession at ingest (Chapters 11, 13, 15)"},
    )


# --------------------------------------------------------------------------- 5. generation
def eager_model(req: CompletionRequest) -> str:
    """Scripted behavior: answers from prior knowledge unless the prompt offers an explicit way out."""
    evidence = req.messages[-1].text.split("Question:")[0]
    if "INSUFFICIENT_EVIDENCE" in req.messages[0].text and "sabbatical" not in evidence.lower():
        return "INSUFFICIENT_EVIDENCE"
    return "Employees receive 11 weeks of paid sabbatical after 7 years of service."


def no_abstention() -> FailureCase:
    rag = corpus_rag(llm=FakeLLM(handler=eager_model))
    hits = rag.index.search(UNCOVERED_QUESTION, k=4, user_groups={"all"})
    texts = [h.chunk.text for h in hits]
    naive_text = rag.llm.complete(naive_request(UNCOVERED_QUESTION, hits)).text
    contract_text = rag.llm.complete(build_request(UNCOVERED_QUESTION, hits)).text
    unsupported = unsupported_claims(naive_text, texts)
    return FailureCase(
        "no abstention", "generate", UNCOVERED_QUESTION,
        observed=f"naive prompt answered: {naive_text!r}",
        detected=bool(unsupported),
        signal=f"answer states figures absent from all evidence: {unsupported}",
        fixed=contract_text.startswith("INSUFFICIENT_EVIDENCE") and not unsupported_claims(contract_text, texts),
        details={"with_contract": contract_text, "fix": "abstention contract + grounding check (Chapter 13)"},
    )


# --------------------------------------------------------------------------- 6. validation
def _sentences_citing(text: str) -> list[tuple[str, str]]:
    pairs = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        pairs.extend((cid, sentence) for cid in CITATION_RE.findall(sentence))
    return pairs


def hallucinated_citation() -> FailureCase:
    question = "How far in advance must I request a two-week vacation?"
    probe = corpus_rag()
    hits = probe.index.search(question, k=4, user_groups={"all"})
    real = next(h.chunk.id for h in hits if h.chunk.doc_id == "hr-pto-policy")
    off_topic = next(h.chunk.id for h in hits if h.chunk.doc_id != "hr-pto-policy")
    scripted = (f"Submit the request at least 14 calendar days in advance [hr-pto-policy#c9]. "
                f"Your manager must answer within 5 working days [{off_topic}]. See also [{real}].")
    rag = corpus_rag(llm=FakeLLM(responses=[scripted]))
    answer = rag.answer(question, user_groups={"all"})
    by_id = {h.chunk.id: h.chunk.text for h in answer.evidence}
    unsupported = sorted({cid for cid, sentence in _sentences_citing(answer.text)
                          if cid in by_id and unsupported_claims(sentence, [by_id[cid]])})
    return FailureCase(
        "hallucinated citation", "validate", question,
        observed=f"cited {answer.cited_ids}",
        detected=bool(answer.invalid_citations) and bool(unsupported),
        signal=f"ids never shown to the model: {answer.invalid_citations}; "
               f"shown but not supporting the cited sentence: {unsupported}",
        fixed=None,
        details={"off_topic": off_topic, "fix": "drop or repair invalid citations; span-level support check (Chapter 13)"},
    )


# --------------------------------------------------------------------------- 7. permissions
def permission_leak() -> FailureCase:
    secret = load_docs(FIXTURES_DIR)                   # hr-compensation-bands, acl_groups ["hr"]
    rag = corpus_rag(extra=secret)
    question = "What is the salary band for a senior software engineer?"
    user_groups = {"all"}                              # an ordinary employee
    naive = rag.index.search(question, k=4)            # user_groups=None: no filter
    leaked = [h.chunk.id for h in naive if not set(h.chunk.acl_groups) & user_groups]
    filtered = rag.index.search(question, k=4, user_groups=user_groups)
    in_prompt = "104,000" in pack_evidence(naive)
    return FailureCase(
        "permission leak", "retrieve (authorization)", question,
        observed=f"unfiltered top-k includes {leaked}; salary figures in prompt: {in_prompt}",
        detected=bool(leaked),
        signal="a packed chunk whose acl_groups do not intersect the caller's groups",
        fixed=not any(not set(h.chunk.acl_groups) & user_groups for h in filtered),
        details={"fix": "filter inside retrieval, never after generation (Chapter 15)"},
    )


ALL_DEMOS = [wrong_chunk_boundary, missing_evidence, distractor, stale_version,
             no_abstention, hallucinated_citation, permission_leak]


def run_all() -> list[FailureCase]:
    return [demo() for demo in ALL_DEMOS]


if __name__ == "__main__":
    for case in run_all():
        fixed = {True: "fixed here", False: "NOT fixed", None: "fix in later chapter"}[case.fixed]
        print(f"[{'DETECTED' if case.detected else 'missed'}] {case.name:30s} stage={case.stage:22s} {fixed}")
        print(f"    observed: {case.observed}")
        print(f"    signal:   {case.signal}")
