# path: book/projects/ragkit/tests/test_retrieval_parent_contextual.py
from __future__ import annotations

from retrieval_fixtures import EMPLOYEE, corpus, gold

from aie_core.llm.errors import ProviderUnavailableError
from aie_core.llm.providers import FakeLLM
from ragkit.chunking import MarkdownSectionChunker
from ragkit.eval.compare_retrievers import GoldQuestion, main, score_question
from ragkit.parsers import MarkdownParser
from ragkit.retrieval.bm25 import BM25Index
from ragkit.retrieval.common import indexed_text
from ragkit.retrieval.contextual import ContextualEnricher, JsonFileCache
from ragkit.retrieval.parent import ParentDocumentRetriever, parent_child_index
from ragkit.retrieval.types import RetrievalQuery

# ----------------------------------------------------------------------------- parent-document retrieval


def parent_retriever() -> tuple[ParentDocumentRetriever, dict]:
    docs, _ = corpus()
    children, parents = parent_child_index(docs)
    child_index = BM25Index()
    child_index.add(children)
    return ParentDocumentRetriever(child_index, parents, fanout=3), parents


def test_parent_retrieval_returns_whole_sections_with_child_evidence():
    retriever, parents = parent_retriever()
    result = retriever.retrieve(RetrievalQuery(text="How long after I return my old laptop is it wiped?", principal=EMPLOYEE, k=3))
    top = result.hits[0]
    assert top.chunk.role == "parent" and top.chunk.id in parents and top.stage == "parent"
    assert top.chunk.doc_id == "it-laptop-replacement-runbook" and "14 days" in top.chunk.text
    assert top.signals["child_hits"] >= 1 and top.signals["child_rank"] >= 1
    assert len(result.chunk_ids) == len(set(result.chunk_ids)) <= 3  # several children collapse into one parent
    assert result.trace["child_k"] == 9 and len(result.trace["child_ids"]) >= len(result.hits)
    child_texts = [c for c in result.trace["child_ids"] if c.startswith("it-laptop-replacement-runbook")]
    assert child_texts  # the trace keeps which children matched


def test_parent_retrieval_respects_acl():
    retriever, _ = parent_retriever()
    for g in [g for g in gold() if g.forbidden]:
        result = retriever.retrieve(RetrievalQuery(text=g.question, principal=g.principal, k=5))
        assert not set(g.required_doc_ids) & set(result.doc_ids), g.id


# ----------------------------------------------------------------------------- contextual retrieval
RUNBOOK = """---
id: it-vpn-test
title: NorthGate VPN Access Runbook
version: "2.0"
tenant: shared
acl_groups: ["all"]
---

# NorthGate VPN Access Runbook

## Error 412

The client shows error 412 when the device certificate expired.

## Fix

Click Renew certificate in the client; if that fails, run Repair. <!-- note -->
"""


def vpn_doc_and_chunks():
    doc = MarkdownParser().parse(RUNBOOK, source_uri="docs/vpn.md")[0]
    return doc, MarkdownSectionChunker(300).chunk(doc)


def context_handler(req):
    chunk = req.messages[-1].text.split("<chunk>")[1]
    if "Renew certificate" in chunk:
        return "<b>From the NorthGate VPN runbook:</b> the fix for error 412, an expired device certificate."
    return "From the NorthGate VPN runbook: overview of the error."


def test_contextual_prefix_is_indexed_but_text_is_unchanged():
    doc, chunks = vpn_doc_and_chunks()
    llm = FakeLLM(handler=context_handler)
    enriched = ContextualEnricher(llm).enrich(chunks, [doc])
    fix = next(c for c in enriched if "Renew certificate" in c.text)
    assert fix.metadata["context_prefix"] == "From the NorthGate VPN runbook: the fix for error 412, an expired device certificate."
    assert fix.text == next(c for c in chunks if c.id == fix.id).text  # citations still quote the source
    assert indexed_text(fix).startswith("From the NorthGate VPN runbook")
    assert "<untrusted_data>" in llm.requests[0].messages[-1].text


def test_contextual_prefix_improves_lexical_retrieval_of_a_terse_chunk():
    doc, chunks = vpn_doc_and_chunks()
    fix_id = next(c.id for c in chunks if "Renew certificate" in c.text)
    plain, ctx = BM25Index(), BM25Index()
    plain.add(chunks)
    ctx.add(ContextualEnricher(FakeLLM(handler=context_handler)).enrich(chunks, [doc]))
    q = "how do I fix error 412"
    assert plain.search(q, EMPLOYEE, 1)[0].chunk.id != fix_id
    assert ctx.search(q, EMPLOYEE, 1)[0].chunk.id == fix_id


def test_contextual_cache_by_content_hash(tmp_path):
    doc, chunks = vpn_doc_and_chunks()
    cache = JsonFileCache(tmp_path / "ctx.json")
    llm = FakeLLM(handler=context_handler)
    first = ContextualEnricher(llm, cache=cache)
    first.enrich(chunks, [doc])
    calls = len(llm.requests)
    assert calls == len(chunks) and first.stats["misses"] == calls
    # a new process with the same cache file makes no calls at all
    second = ContextualEnricher(llm, cache=JsonFileCache(tmp_path / "ctx.json"))
    again = second.enrich(chunks, [doc])
    assert len(llm.requests) == calls and second.stats == {"hits": calls, "misses": 0, "failures": 0}
    assert all("context_prefix" in c.metadata for c in again)
    # with a narrow window, editing one section recomputes only the chunks whose window changed
    narrow = ContextualEnricher(llm, cache=cache, window_chars=40)
    narrow.enrich(chunks, [doc])
    edited = MarkdownParser().parse(RUNBOOK.replace("run Repair", "run Repair from Self-Service"), source_uri="docs/vpn.md")[0]
    third = ContextualEnricher(llm, cache=cache, window_chars=40)
    third.enrich(MarkdownSectionChunker(300).chunk(edited), [edited])
    assert third.stats["misses"] == 1 and third.stats["hits"] == len(chunks) - 1
    # a prompt-version bump recomputes everything
    fourth = ContextualEnricher(llm, cache=cache, prompt_version="ctx-v2")
    fourth.enrich(chunks, [doc])
    assert fourth.stats["hits"] == 0


def test_contextual_failure_is_not_cached_and_falls_back():
    doc, chunks = vpn_doc_and_chunks()
    llm = FakeLLM(responses=[ProviderUnavailableError("down", provider="fake")] * len(chunks))
    enricher = ContextualEnricher(llm, max_concurrency=1)
    out = enricher.enrich(chunks, [doc])
    assert enricher.stats["failures"] == len(chunks) and len(enricher.cache) == 0
    assert all("context_prefix" not in c.metadata for c in out)


def test_contextual_output_is_capped():
    doc, chunks = vpn_doc_and_chunks()
    enricher = ContextualEnricher(FakeLLM(handler=lambda req: "word " * 500), max_prefix_chars=60)
    out = enricher.enrich(chunks, [doc])
    assert all(len(c.metadata["context_prefix"]) <= 60 for c in out)


# ----------------------------------------------------------------------------- comparison script
def test_inverted_scoring_for_forbidden_questions():
    forbidden = GoldQuestion("X", "q", ["secret"], ["all"], "shared", ["forbidden-doc", "abstain"])
    assert score_question(forbidden, ["a", "b"], [1, 5])["hit@5"] == 1.0
    leaked = score_question(forbidden, ["a", "secret"], [1, 5])
    assert leaked["hit@1"] == 0.0 and leaked["leak"] == 1.0
    normal = GoldQuestion("Y", "q", ["d1", "d2"], ["all"], "shared", [])
    row = score_question(normal, ["x", "d2", "d2", "d1"], [1, 3, 5], candidate_docs=["d1"])
    assert row["hit@1"] == 0.0 and row["hit@3"] == 1.0 and row["recall@3"] == 0.5 and row["recall@5"] == 1.0
    assert row["mrr"] == 0.5 and row["cand_recall"] == 0.5


def test_compare_retrievers_runs_offline_without_leaks():
    lines: list[str] = []
    summaries = main(["--k", "1", "5", "--by-tag"], out=lines.append)
    assert set(summaries) == {"bm25", "dense", "hybrid", "hybrid+rerank"}
    assert all(s["leaks"] == 0 and s["n"] == 40 for s in summaries.values())
    assert summaries["hybrid"]["mrr"] >= min(summaries["bm25"]["mrr"], summaries["dense"]["mrr"])
    assert summaries["hybrid+rerank"]["cand_recall"] >= summaries["hybrid+rerank"]["recall@5"]
    text = "\n".join(lines)
    assert "| hybrid+rerank |" in text and "forbidden-doc" in text
