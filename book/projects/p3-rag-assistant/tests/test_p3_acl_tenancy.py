# path: book/projects/p3-rag-assistant/tests/test_p3_acl_tenancy.py
"""Permissions: forbidden documents never reach candidates, prompts or caches; tenants stay apart."""
from __future__ import annotations

import pytest
from conftest import PTO_Q
from guardrails import GuardContext
from guardrails.tenancy import TenantIsolationError
from ragkit.eval.rag_dataset import load_gold_dataset, rag_input
from ragkit.retrieval import Principal, visible

from rag_assistant.caching.caches import guard_context
from rag_assistant.config import DEFAULT_GOLD_PATH


def _forbidden_cases():  # type: ignore[no-untyped-def]
    return [c for c in load_gold_dataset(DEFAULT_GOLD_PATH).cases if "forbidden-doc" in c.tags]


def test_forbidden_doc_gold_questions_never_see_the_document(container):
    cases = _forbidden_cases()
    assert len(cases) == 3
    for case in cases:
        inp = rag_input(case)
        forbidden = set(case.expected["forbidden_doc_ids"])
        out = container.answers.ask(inp.question, inp.principal)
        assert not forbidden & set(out.retrieval.doc_ids), case.id
        assert all(visible(h.chunk, inp.principal) for h in out.retrieval.hits)
        assert out.retrieval.trace["acl_violations"] == []
        if out.qa is not None:
            assert not forbidden & {b.doc_id for b in out.qa.packed.blocks}
            assert not forbidden & {c.doc_id for c in out.response.answer.citations}


def test_forbidden_doc_questions_abstain_except_the_known_label_error(container):
    """RQ-037 is answerable from hr-faq, which everyone may read (see Chapter 14): answering it
    is correct behavior and not a leak. The other forbidden-doc questions must abstain."""
    for case in _forbidden_cases():
        inp = rag_input(case)
        action = container.answers.ask(inp.question, inp.principal).response.answer.action
        if case.id != "RQ-037":
            assert action == "abstain", case.id


def test_oncall_group_sees_the_restricted_runbook(container, employee, oncall):
    q = "What are the RTO and RPO for the production PostgreSQL clusters?"
    assert "it-database-failover-runbook" not in container.answers.ask(q, employee).retrieval.doc_ids
    assert "it-database-failover-runbook" in container.answers.ask(q, oncall).retrieval.doc_ids


@pytest.mark.parametrize("mode", ["shared", "namespace"])
def test_cross_tenant_isolation(make_container, mode, employee, logistics_employee):
    c = make_container(tenancy_mode=mode)
    q = "From June 2026, which weekday do deliveries to North region depots move to?"
    retail = c.answers.ask(q, employee)
    assert all(h.chunk.tenant in ("retail", "shared") for h in retail.retrieval.hits)
    assert "ext-vendor-newsletter-brightline" not in retail.retrieval.doc_ids
    logistics = c.answers.ask(q, logistics_employee)
    assert "ext-vendor-newsletter-brightline" in logistics.retrieval.doc_ids
    if mode == "namespace":  # a retail query does not even open the logistics partition
        lists = [s["name"] for s in retail.retrieval.trace["stages"] if s["kind"] == "retrieve"]
        assert lists and not any("logistics" in n for n in lists)


def test_cache_keys_include_the_authorization_scope(container, employee, oncall):
    a = container.answers.ask(PTO_Q, employee)
    b = container.answers.ask(PTO_Q, oncall)  # same question, different groups
    assert b.response.cache == "miss"  # never served from the employee's entry
    ka = container.caches.answers.key(guard_context(employee), "x")
    kb = container.caches.answers.key(guard_context(oncall), "x")
    kc = container.caches.answers.key(guard_context(Principal(user_id="z", tenant="logistics", groups=["all"])), "x")
    assert len({ka, kb, kc}) == 3
    assert a.response.answer.text  # both answered independently
    # a hand-built context that collides on the key still cannot read across scopes
    entry_key_parts = ("probe",)
    container.caches.answers.put(guard_context(employee), a, *entry_key_parts, doc_ids=["hr-pto-policy"])
    forged = GuardContext(tenant="logistics", user_id="x", groups=frozenset({"all"}))
    container.caches.answers._store[container.caches.answers.key(forged, *entry_key_parts)] = \
        container.caches.answers._store[container.caches.answers.key(guard_context(employee), *entry_key_parts)]
    container.caches.answers._meta[container.caches.answers.key(forged, *entry_key_parts)] = \
        container.caches.answers._meta[container.caches.answers.key(guard_context(employee), *entry_key_parts)]
    with pytest.raises(TenantIsolationError):
        container.caches.answers.get(forged, *entry_key_parts)


def test_retrieval_cache_key_includes_index_generation(container, employee, docs_dir):
    container.answers.ask(PTO_Q, employee)
    assert container.answers.ask(PTO_Q, employee).response.cache == "hit"
    container.ingestion.delete("hr-travel-policy")  # any change in a scope the caller can read
    assert container.answers.ask(PTO_Q, employee).response.cache == "miss"
