# path: book/projects/examples/ch01/test_lineage.py
from __future__ import annotations

from lineage import (
    EvalOutcome,
    EvidenceRef,
    ModelRef,
    PolicyGate,
    PromptRef,
    RequestLineage,
    ToolEvent,
    Usage,
    content_hash,
)


def northwind_request() -> RequestLineage:
    """A retail employee asks about the parental-leave policy. RAG only, no tools called."""
    return RequestLineage(
        request_id="req-0001",
        tenant="retail",
        principal="emp-4471",
        principal_groups=("all", "retail"),
        model=ModelRef(provider="fake", name="fake-model", version="2026-01"),
        prompt=PromptRef(name="policy_answer", version="7", content_hash=content_hash("SYSTEM v7")),
        usage=Usage(input_tokens=1850, output_tokens=210, cached_input_tokens=1200),
        latency_ms=2340.0,
        output_hash=content_hash("Parental leave is 16 weeks... [1]"),
        index_version="kb-2026-03-01",
        evidence=[
            EvidenceRef("hr/parental-leave.md", "c3", 0.91, ("all",)),
            EvidenceRef("hr/leave-faq.md", "c1", 0.74, ("all",)),
        ],
        tools=[ToolEvent(name="create_ticket", offered=True, called=False)],
        gates=[
            PolicyGate("tenant_filter", "allow"),
            PolicyGate("evidence_gate", "allow", reason="2 chunks above threshold"),
        ],
        eval_outcome=EvalOutcome.PASS,
    )


def test_complete_record_answers_every_question() -> None:
    lineage = northwind_request()
    assert lineage.unanswered_questions() == []
    assert lineage.consistency_violations() == []
    assert lineage.is_complete()


def test_missing_gates_and_usage_are_reported_by_name() -> None:
    lineage = northwind_request()
    lineage.gates = []
    lineage.usage = Usage()
    missing = lineage.unanswered_questions()
    assert "which policy gates ran" in missing
    assert "how many tokens it consumed" in missing


def test_evidence_outside_callers_groups_is_a_violation() -> None:
    lineage = northwind_request()
    lineage.evidence.append(EvidenceRef("it/oncall-runbook.md", "c9", 0.80, ("it-oncall",)))
    violations = lineage.consistency_violations()
    assert len(violations) == 1
    assert "outside the caller's groups" in violations[0]


def test_cross_tenant_evidence_is_a_violation_even_when_groups_match() -> None:
    lineage = northwind_request()
    # Visible to group "all", but tagged for the other tenant: groups alone would admit it.
    lineage.evidence.append(EvidenceRef("hr/logistics-shift-pay.md", "c2", 0.77, ("all",), "logistics"))
    violations = lineage.consistency_violations()
    assert len(violations) == 1
    assert "belongs to tenant logistics" in violations[0]


def test_evidence_without_index_version_is_unanswerable() -> None:
    lineage = northwind_request()
    lineage.index_version = None
    assert "which index version served the evidence" in lineage.unanswered_questions()


def test_tool_called_without_being_offered_is_a_violation() -> None:
    lineage = northwind_request()
    lineage.tools.append(ToolEvent(name="send_reply", offered=False, called=True, outcome="ok"))
    assert any("never offered" in v for v in lineage.consistency_violations())


def test_denied_gate_with_model_answer_shown_is_a_violation() -> None:
    lineage = northwind_request()
    lineage.gates.append(PolicyGate("output_pii_scan", "deny", reason="employee SSN in answer"))
    assert any("denied but the model's answer was shown" in v for v in lineage.consistency_violations())


def test_denied_gate_followed_by_fallback_message_is_consistent() -> None:
    lineage = northwind_request()
    lineage.gates[-1] = PolicyGate("evidence_gate", "deny", reason="no chunk above threshold")
    lineage.output_kind = "fallback"
    lineage.output_hash = content_hash("I could not find a current policy document for this...")
    assert lineage.consistency_violations() == []


def test_json_round_trip_preserves_every_field() -> None:
    lineage = northwind_request()
    restored = RequestLineage.from_dict(__import__("json").loads(lineage.to_json()))
    assert restored == lineage
    assert restored.to_json() == lineage.to_json()
