# path: book/projects/examples/ch37/tests/test_graphrag.py
from __future__ import annotations

import pytest

from aie_core.llm.providers import FakeLLM
from corpus import EMPLOYEE, ONCALL
from graphrag import DictGraph, EntityResolver, GraphRAG, fixture_chunks, fixture_handler


@pytest.fixture()
def rag() -> GraphRAG:
    g = GraphRAG(FakeLLM(handler=fixture_handler()), store=DictGraph())
    g.build(fixture_chunks())
    return g


def test_build_uses_one_extraction_call_per_chunk(rag: GraphRAG):
    assert rag.stats.chunks == len(fixture_chunks()) == 5
    assert rag.stats.llm_calls == rag.stats.chunks


def test_resolver_merges_surface_variants_through_normalization_and_aliases():
    r = EntityResolver()
    a = r.resolve("PayBridge Adapter")
    assert r.resolve("paybridge adapter") == a
    assert r.resolve("the `PayBridge Adapter`") == a
    team = r.resolve("Retail Systems team", aliases=["Retail Systems"])
    assert r.resolve("Retail Systems") == team
    cert = r.resolve("PayBridge client certificate")
    assert r.resolve("PayBridge certificate", aliases=["PayBridge client certificate"]) == cert


def test_resolver_override_forces_a_merge():
    r = EntityResolver(overrides={"IC": "Incident Commander"})
    assert r.resolve("IC") == r.resolve("Incident Commander")


def test_unverifiable_evidence_is_flagged_not_silently_trusted(rag: GraphRAG):
    flagged = [d for _, _, d in rag.store.edges() if not d["verified"]]
    assert rag.stats.unverified_relations == len(flagged) == 1
    assert flagged[0]["relation"] == "monitored_by"


def test_verbatim_evidence_does_not_prove_a_correct_reading(rag: GraphRAG):
    # "owned by a single engineer who had left Retail Systems" is verbatim, so it verifies,
    # yet the relation "Retail Systems owned the certificate" misreads it. Span checks catch
    # fabrication, not misinterpretation; that needs sampled human review of edges.
    owned = [d for u, v, d in rag.store.edges() if d["relation"] == "owned"]
    assert owned and owned[0]["verified"] is True


def test_local_query_returns_neighborhood_facts_with_provenance(rag: GraphRAG):
    facts = rag.local_query("What does the PayBridge Adapter depend on?", ONCALL, hops=1)
    rendered = [f.render() for f in facts]
    assert any("authenticates_with" in r and "PayBridge client certificate" in r for r in rendered)
    assert {f.doc_id for f in facts} == {"prod-retail-pos-overview", "inc-2025-11-pos-outage"}


def test_two_hops_reach_the_certificate_inventory(rag: GraphRAG):
    one = {f.target for f in rag.local_query("PayBridge Adapter", ONCALL, hops=1)}
    two = {f.target for f in rag.local_query("PayBridge Adapter", ONCALL, hops=2)}
    assert "certificate inventory" not in one
    assert "certificate inventory" in two


def test_local_query_filters_edges_by_acl(rag: GraphRAG):
    facts = rag.local_query("What does the PayBridge Adapter depend on?", EMPLOYEE, hops=1)
    assert facts and {f.doc_id for f in facts} == {"prod-retail-pos-overview"}


def test_communities_and_summaries_carry_their_sources_acl(rag: GraphRAG):
    summaries = rag.summarize_communities(min_size=3)
    assert len(summaries) == 2
    paybridge = next(s for s in summaries if "paybridge adapter" in s.members)
    assert "incident commander" not in paybridge.members
    # it mixes a restricted incident report with a public overview: hidden from ordinary employees
    assert paybridge.visible_to(ONCALL) and not paybridge.visible_to(EMPLOYEE)


def test_global_query_maps_over_visible_summaries_and_cites_communities(rag: GraphRAG):
    rag.summarize_communities()
    answer, used = rag.global_query("Who posts updates to the incidents channel?", ONCALL)
    assert used and answer.startswith("Synthesis:")
    assert rag.global_query("Who posts updates to the incidents channel?", EMPLOYEE) == ("INSUFFICIENT_EVIDENCE", [])
