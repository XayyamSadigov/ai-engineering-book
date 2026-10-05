# path: book/projects/examples/ch37/tests/test_toc_navigation.py
from __future__ import annotations

from aie_core.llm.providers import FakeLLM
from corpus import EMPLOYEE, ONCALL, load_corpus
from toc_navigation import NavBudget, TocNavigator, build_toc, keyword_picker


def _doc(doc_id: str):
    return next(d for d in load_corpus() if d.id == doc_id)


def test_toc_spans_nest_and_cover_subsections():
    toc = build_toc(_doc("hr-pto-policy"))
    titles = [toc.nodes[r].title for r in toc.roots]
    assert titles[0].startswith("1. Purpose") and any(t.startswith("3. Carryover") for t in titles)
    for node in toc.nodes.values():
        for child in node.children:
            c = toc.nodes[child]
            assert node.start <= c.start and c.end <= node.end


def test_navigates_to_the_carryover_section_of_the_current_policy():
    result = TocNavigator(FakeLLM(handler=keyword_picker())).navigate("How many unused PTO days can I carry over?", EMPLOYEE)
    assert result.trace[0].picked[0] == "hr-pto-policy"
    assert any("Carryover" in s.path[-1] and "10 unused PTO days" in s.text for s in result.sections)
    assert result.llm_calls == 2


def test_catalog_hides_documents_outside_the_principals_acl():
    nav = TocNavigator(FakeLLM(handler=keyword_picker()))
    result = nav.navigate("INC-2025-1142 PayBridge outage root cause", EMPLOYEE)
    assert "inc-2025-11-pos-outage" not in result.trace[0].offered
    oncall = nav.navigate("INC-2025-1142 PayBridge outage root cause", ONCALL)
    assert "inc-2025-11-pos-outage" in oncall.trace[0].offered


def test_hallucinated_and_forbidden_ids_are_rejected_and_recorded():
    responses = [
        {"ids": ["inc-2025-11-pos-outage", "made-up-doc", "hr-pto-policy"]},
        {"ids": ["hr-pto-policy#s99", "hr-pto-policy#s3"]},
    ]
    result = TocNavigator(FakeLLM(responses=responses)).navigate("carryover", EMPLOYEE)
    assert result.trace[0].rejected == ["inc-2025-11-pos-outage", "made-up-doc"]
    assert result.trace[1].rejected == ["hr-pto-policy#s99"]
    assert [s.node_id for s in result.sections] == ["hr-pto-policy#s3"]


def test_read_budget_limits_tokens_read():
    nav = TocNavigator(FakeLLM(handler=keyword_picker()), budget=NavBudget(max_read_tokens=100))
    result = nav.navigate("How many unused PTO days can I carry over?", EMPLOYEE)
    assert sum(s.tokens for s in result.sections) <= 100
    assert result.stop_reason == "read_budget"


def test_vocabulary_mismatch_is_a_visible_failure_not_a_wrong_answer():
    # headings say "Payments", the question says "renewal process": a keyword navigator picks
    # the wrong section. The trace shows exactly where navigation went astray.
    result = TocNavigator(FakeLLM(handler=keyword_picker())).navigate("What is the PayBridge certificate renewal process?", EMPLOYEE)
    assert result.trace[1].picked == ["prod-retail-pos-overview#s6"]  # "Release process"
    assert "Payments" not in result.sections[0].path[-1]
