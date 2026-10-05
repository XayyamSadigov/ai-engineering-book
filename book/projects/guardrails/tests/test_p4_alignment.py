# path: book/projects/guardrails/tests/test_p4_alignment.py
"""The preset's tool rules must match Project 4's real tool schemas, and PII tokens from the input
guardrail must survive the trip to a real tool (Chapter 27, "PII tokens at the tool boundary")."""
from __future__ import annotations

import re

import pytest
from aie_core.llm.types import ToolCall

from guardrails import GuardContext, GuardrailPipeline, PIIVault, ToolPolicyCheck
from guardrails.pii import redact_pii
from guardrails.presets import agent_pipeline, support_tool_rules
from guardrails.tools import approval_token, guard_tool_call, rehydrate_arguments, rehydrate_kinds, token_tolerant_schema

wiring = pytest.importorskip("support_assistant.wiring", reason="Project 4 (support_assistant) not installed")

SAMPLE_VALUES = {
    "query": "vpn drops store", "status": "open", "limit": 5, "service": "vpn",
    "subject": "VPN drops at store 0412", "body": "The VPN drops every ten minutes since Monday morning.",
    "category": "vpn_network", "priority": "P3", "ticket_id": "TCK-2026-0001", "to": "jane@northwind.example",
}


@pytest.fixture(scope="module")
def specs():
    return {s.name: s for s in wiring.build_container().registry.specs()}


def _valid_args(spec) -> dict:
    props = spec.parameters.get("properties", {})
    return {k: SAMPLE_VALUES[k] for k in props if k in SAMPLE_VALUES}


def test_every_project4_tool_has_a_rule_with_its_real_argument_names(specs):
    rules = {r.name: r for r in support_tool_rules()}
    assert set(rules) == set(specs), "preset and Project 4 must cover the same tools"
    for name, spec in specs.items():
        props = set(spec.parameters.get("properties", {}))
        required = set(spec.parameters.get("required", []))
        rule = rules[name]
        assert set(rule.required_args) == required, name
        assert set(rule.constraints) <= props, (name, set(rule.constraints) - props)
        assert set(rule.rehydrate_args) <= props, name


def test_valid_project4_calls_pass_the_preset(specs, ctx):
    p = GuardrailPipeline([ToolPolicyCheck(support_tool_rules(), max_calls_per_request=50)])
    for name, spec in specs.items():
        c = ToolCall(id=f"c-{name}", name=name, arguments=_valid_args(spec))
        res = p.check_tool(c, ctx)
        if name == "send_reply":   # approval-gated: blocked until the exact arguments are approved
            assert res.verdicts[-1].metadata.get("needs_approval"), res.reasons()
            ctx.approvals.add(approval_token(c))
            res = p.check_tool(c, ctx)
        assert res.allowed, (name, res.reasons())


def test_preset_constraints_mirror_project4_limits(specs, ctx):
    p = GuardrailPipeline([ToolPolicyCheck(support_tool_rules())])
    bad_ticket = dict(_valid_args(specs["draft_reply"]), ticket_id="TICKET-1")
    assert not p.check_tool(ToolCall(id="x", name="draft_reply", arguments=bad_ticket), ctx).allowed
    bad_priority = dict(_valid_args(specs["create_ticket"]), priority="urgent")
    assert not p.check_tool(ToolCall(id="y", name="create_ticket", arguments=bad_priority), ctx).allowed


# ---------------------------------------------------------------- PII tokens at the tool boundary
def _tokenized_ctx() -> tuple[GuardContext, str]:
    vault = PIIVault(tenant="retail", scope_id="req-1")
    ctx = GuardContext(tenant="retail", user_id="u-42", groups=frozenset({"all", "support"}), request_id="req-1",
                       vault=vault)
    text = redact_pii("Please reply to jane@northwind.example about TCK-2026-0001", mode="tokenize", vault=vault).text
    token = re.search(r"<PII:email:[0-9a-f]{10}>", text).group(0)
    return ctx, token


def test_raw_token_fails_the_tool_schema_and_the_recipient_rule(specs):
    ctx, token = _tokenized_ctx()
    args = dict(_valid_args(specs["send_reply"]), to=token)
    # Project 4's own validation rejects a token as an e-mail address ...
    with pytest.raises(Exception):
        wiring.build_container().registry.get("send_reply").args_model.model_validate(args)
    # ... and so does the guardrail when the token is not re-hydrated first.
    res = GuardrailPipeline([ToolPolicyCheck(support_tool_rules())]).check_tool(
        ToolCall(id="c", name="send_reply", arguments=args), ctx)
    assert not res.allowed and "not an email address" in res.reasons()[0]


def test_guard_tool_call_rehydrates_then_checks_and_approval_binds_to_the_real_value(specs):
    ctx, token = _tokenized_ctx()
    pipeline = agent_pipeline()
    model_call = ToolCall(id="c", name="send_reply", arguments=dict(_valid_args(specs["send_reply"]), to=token))
    prepared, res = guard_tool_call(pipeline, model_call, ctx)
    assert prepared.arguments["to"] == "jane@northwind.example" and model_call.arguments["to"] == token
    assert res.verdicts[-1].metadata["approval_token"] == approval_token(prepared)   # human approves the real recipient
    ctx.approvals.add(approval_token(prepared))
    prepared, res = guard_tool_call(pipeline, model_call, ctx)
    assert res.allowed, res.reasons()
    # the call that executes satisfies Project 4's real schema
    wiring.build_container().registry.get("send_reply").args_model.model_validate(prepared.arguments)


def test_rehydration_is_scoped_to_listed_arguments_kinds_and_tenant(specs):
    ctx, token = _tokenized_ctx()
    c = ToolCall(id="c", name="send_reply", arguments={"to": token, "body": f"cc {token}"})
    only_to = rehydrate_arguments(c, ctx, rehydrate_kinds("email"), ["to"])
    assert only_to.arguments["to"] == "jane@northwind.example" and token in only_to.arguments["body"]
    assert rehydrate_arguments(c, ctx, rehydrate_kinds("phone"), ["to"]).arguments["to"] == token
    forged = ToolCall(id="c", name="send_reply", arguments={"to": "<PII:email:0123456789>"})
    assert rehydrate_arguments(forged, ctx, rehydrate_kinds("email")).arguments["to"] == "<PII:email:0123456789>"
    other_tenant = GuardContext(tenant="logistics", user_id="u", groups=frozenset({"all"}), request_id="r",
                                vault=ctx.vault)
    with pytest.raises(PermissionError):
        rehydrate_arguments(c, other_tenant, rehydrate_kinds("email"), ["to"])


def test_token_tolerant_schema_accepts_tokens_and_keeps_the_original_pattern(specs):
    _, token = _tokenized_ctx()
    original = specs["send_reply"].parameters
    tolerant = token_tolerant_schema(original)
    pattern = tolerant["properties"]["to"]["pattern"]
    assert re.fullmatch(pattern, token) and re.fullmatch(pattern, "jane@northwind.example")
    assert not re.fullmatch(pattern, "not an address")
    assert original["properties"]["to"]["pattern"] != pattern     # the original schema is not mutated
