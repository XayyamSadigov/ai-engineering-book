# path: book/projects/examples/ch32/tests/test_parsing_properties.py
"""Property-based tests for the output parser and the business rules.

Example-based tests check the cases you thought of. Property tests state what must hold
for every input and let hypothesis search for the cases you did not think of, then shrink
a failure to the smallest counterexample.
"""
from __future__ import annotations

import json

from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from northwind_triage.domain import (
    Category,
    ParseError,
    Priority,
    Ticket,
    TriageDecision,
    apply_business_rules,
    parse_triage,
    render_decision,
)
from northwind_triage.domain.parsing import (
    extract_json_object,
    normalize_category,
    normalize_priority,
)

decisions = st.builds(
    TriageDecision,
    category=st.sampled_from(list(Category)),
    priority=st.sampled_from(list(Priority)),
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    rationale=st.text(max_size=200),
)
# Prose that models put around JSON. It may not contain '{' (otherwise it IS a JSON candidate).
noise = st.text(alphabet=st.characters(blacklist_characters="{}", blacklist_categories=("Cs",)), max_size=80)
wrappers = st.sampled_from(["{}", "```json\n{}\n```", "Here is the result:\n{}", "{}\nHope this helps!",
                            "```\n{}\n```\nLet me know."])
tickets = st.builds(Ticket, id=st.just("T-1"), tenant=st.sampled_from(["retail", "logistics", "shared"]),
                    subject=st.text(max_size=40), body=st.text(max_size=200))


@given(st.text())
@example("")
@example("{")
@example('{"category": "hardware", "priority": "P2", "confidence": NaN}')
@example('{"category": "hardware", "priority": true, "confidence": 0.5}')
@example("[" * 5000)
def test_parser_total_function(text: str) -> None:
    """Any string: a decision or ParseError. Never KeyError, TypeError, RecursionError."""
    try:
        result = parse_triage(text)
    except ParseError:
        return
    assert isinstance(result, TriageDecision)


@given(decisions, noise, noise, wrappers)
def test_round_trip_through_prose_and_fences(decision: TriageDecision, before: str, after: str,
                                             wrapper: str) -> None:
    text = before + wrapper.replace("{}", render_decision(decision)) + after
    assert parse_triage(text) == decision


@given(st.dictionaries(st.text(max_size=10),
                       st.recursive(st.none() | st.booleans() | st.integers() | st.text(max_size=10),
                                    lambda inner: st.lists(inner, max_size=3)
                                    | st.dictionaries(st.text(max_size=5), inner, max_size=3),
                                    max_leaves=10),
                       max_size=5), noise)
def test_extract_finds_any_embedded_object(obj: dict, prefix: str) -> None:
    """Braces and quotes inside string values must not confuse the scanner."""
    assert extract_json_object(prefix + json.dumps(obj)) == obj


@given(st.sampled_from(list(Priority)), st.sampled_from(["", " ", "  "]), st.booleans(), st.booleans())
def test_priority_normalization_is_idempotent(p: Priority, pad: str, lower: bool, drop_p: bool) -> None:
    raw = p.value[1:] if drop_p else p.value
    raw = raw.lower() if lower else raw
    once = normalize_priority(pad + raw + pad)
    assert once == p.value
    assert normalize_priority(once) == once


@given(st.sampled_from(list(Category)), st.sampled_from(["upper", "title_spaces", "dashes"]))
def test_category_normalization_accepts_spellings(c: Category, style: str) -> None:
    spelled = {"upper": c.value.upper(), "title_spaces": c.value.replace("_", " ").title(),
               "dashes": c.value.replace("_", "-")}[style]
    assert normalize_category(spelled) == c.value


@settings(suppress_health_check=[HealthCheck.too_slow])
@given(tickets, decisions)
def test_business_rules_never_lower_urgency(ticket: Ticket, decision: TriageDecision) -> None:
    final = apply_business_rules(ticket, decision)
    assert final.priority.urgency >= decision.priority.urgency
    if decision.category == Category.SECURITY_REPORT:
        assert final.route == "human_review"
        assert final.priority.urgency >= Priority.P2.urgency
    if decision.confidence < 0.6:
        assert final.route == "human_review"
