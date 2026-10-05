# path: book/projects/p6-research-team/tests/test_contracts_and_ledger.py
from __future__ import annotations

import json
from datetime import date

import pytest
from pydantic import ValidationError

from research_team import BudgetLedger, BudgetSlice, Claim, EvidenceRef, Role, TaskEnvelope, TeamBudget
from research_team.checks import deterministic_support, find_conflicts
from research_team.contracts import objective_key


def env(task_id: str = "t.r1-sq1", objective: str = "What is the hotel cap?", depth: int = 1,
        tokens: int = 12_000) -> TaskEnvelope:
    return TaskEnvelope(task_id=task_id, parent_id="t", trace_id="t", sender="supervisor",
                        recipient=Role.RESEARCHER, depth=depth, objective=objective,
                        output_schema="ResearchFindings", budget=BudgetSlice(max_tokens=tokens),
                        principal={"groups": ["all", "hr"], "tenant": "retail"})


def test_envelope_renders_task_and_schema_but_never_the_principal():
    text = env().render()
    body = json.loads(text.split("TASK\n", 1)[1].split("\nOUTPUT JSON SCHEMA", 1)[0])
    assert body["task_id"] == "t.r1-sq1" and body["output_schema"] == "ResearchFindings"
    assert "OUTPUT JSON SCHEMA" in text and '"claims"' in text
    assert "hr" not in body.get("inputs", {}) and "principal" not in text


def test_envelope_rejects_unsafe_task_ids_and_claims_need_evidence():
    with pytest.raises(ValidationError):
        env(task_id="../etc/passwd")
    with pytest.raises(ValidationError):
        Claim(text="Hotels are capped.", evidence=[])


def test_objective_key_ignores_case_and_punctuation():
    assert objective_key("What is the HOTEL cap?") == objective_key("what is the hotel cap")
    assert objective_key("hotel cap") != objective_key("per diem")


def test_reserve_then_settle_never_lets_reservations_exceed_the_pool():
    ledger = BudgetLedger(TeamBudget(max_tokens=30_000, min_child_tokens=4_000, max_children=10))
    a = ledger.admit(env("t.a", "q a"))
    b = ledger.admit(env("t.b", "q b"))
    c = ledger.admit(env("t.c", "q c"))
    assert a.admitted and b.admitted and a.granted.max_tokens == 12_000
    assert c.admitted and c.granted.max_tokens == 6_000          # what is left, not what was asked
    assert not ledger.admit(env("t.d", "q d")).admitted           # pool empty: below min_child_tokens
    ledger.settle("t.a", tokens=3_000, cost_usd=0.0)              # unused 9k returns to the pool
    assert ledger.available_tokens() == 9_000
    assert ledger.admit(env("t.e", "q e")).granted.max_tokens == 9_000


def test_admission_refusals_have_reasons():
    ledger = BudgetLedger(TeamBudget(max_children=1, max_depth=1))
    assert ledger.admit(env("t.a", "same")).reason == "ok"
    assert ledger.admit(env("t.b", "other")).reason == "spawn_cap"
    ledger2 = BudgetLedger(TeamBudget(max_children=5))
    ledger2.admit(env("t.a", "same question"))
    assert ledger2.admit(env("t.b", "Same question!")).reason == "duplicate"
    assert ledger2.admit(env("t.c", "deeper", depth=2)).reason == "max_depth"


def test_deadline_propagates_to_children():
    now = [0.0]
    ledger = BudgetLedger(TeamBudget(deadline_s=30.0), clock=lambda: now[0])
    now[0] = 20.0
    granted = ledger.admit(env("t.a", "q")).granted
    assert granted.deadline_s == pytest.approx(10.0)              # child inherits the remaining time
    now[0] = 29.5
    assert ledger.admit(env("t.b", "q2")).reason == "deadline"


def test_hold_protects_synthesis_budget():
    ledger = BudgetLedger(TeamBudget(max_tokens=20_000, min_child_tokens=4_000))
    ledger.hold("synthesis", 10_000)
    assert ledger.admit(env("t.a", "q")).granted.max_tokens == 10_000
    assert ledger.admit(env("t.b", "q2")).reason == "budget"
    ledger.release("synthesis")
    assert ledger.available_tokens() == 10_000


def test_deterministic_support_is_strict_on_numbers():
    src = "From 1 January 2026, employees may carry over up to 10 unused PTO days into the next calendar year."
    assert deterministic_support("Employees may carry over up to 10 unused PTO days.", src)[0]
    ok, why = deterministic_support("Employees may carry over up to 17 unused PTO days.", src)
    assert not ok and "17" in why
    assert not deterministic_support("Employees get a free laptop every year.", src)[0]


def test_conflicts_between_documents_prefer_the_newer_one():
    policy = Claim(claim_id="a", text="From 1 January 2026, employees may carry over up to 10 unused PTO days "
                   "into the next calendar year.", evidence=[EvidenceRef(doc_id="hr-pto-policy", passage_id="p#1")])
    faq = Claim(claim_id="b", text="Under the PTO Policy in force since 1 January 2024, you may carry over up to 5 "
                "unused days into the next calendar year.", evidence=[EvidenceRef(doc_id="hr-faq", passage_id="f#1")])
    perdiem = Claim(claim_id="c", text="International travel: 95 USD per day.",
                    evidence=[EvidenceRef(doc_id="hr-expense-policy", passage_id="e#1")])
    dates = {"hr-pto-policy": date(2026, 1, 15), "hr-faq": date(2025, 6, 10), "hr-expense-policy": date(2025, 1, 1)}
    found = find_conflicts([policy, faq, perdiem], dates.get)
    assert len(found) == 1 and found[0].preferred_doc == "hr-pto-policy" and found[0].unit == "day"
