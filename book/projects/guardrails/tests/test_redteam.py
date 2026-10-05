# path: book/projects/guardrails/tests/test_redteam.py
"""The Chapter 26 attack corpus, end to end. A case passes when the harmful effect is blocked even
though the simulated model fully complies with every injected instruction."""
from __future__ import annotations

import pytest

from guardrails import GuardrailPipeline, agent_pipeline, rag_pipeline
from guardrails.eval import measure
from guardrails.eval.datasets import load_ch26
from guardrails.eval.redteam import PROMPT_CANARY, default_context, run_all, run_scenario

ac = load_ch26()
SENSITIVE = ac.sensitive_documents(seed=0)
CANARIES = [d.canary for d in SENSITIVE] + [PROMPT_CANARY]


def test_without_guardrails_every_effect_occurs():
    # Proves the harness can detect each effect; otherwise "0 effects" below would mean nothing.
    results = run_all(GuardrailPipeline([]))
    assert len(results) == len(ac.Variant)
    assert all(r.effect_occurred for r in results), [(r.scenario, r.evidence) for r in results]


@pytest.mark.parametrize("variant", list(ac.Variant), ids=lambda v: v.value)
def test_agent_pipeline_blocks_effect_per_variant(variant):
    doc = next(d for d in ac.adversarial_documents() if d.variant is variant)
    r = run_scenario(agent_pipeline(canaries=CANARIES), doc, SENSITIVE, ac, default_context())
    assert not r.effect_occurred, r.evidence
    assert r.blocked_by, "an effect-level control, not luck, stopped it"


def test_which_layer_stops_what():
    pipe = agent_pipeline(canaries=CANARIES)
    by_variant = {}
    for doc in ac.adversarial_documents():
        r = run_scenario(pipe, doc, SENSITIVE, ac, default_context())
        by_variant[doc.variant] = r.blocked_by
    assert any(b.startswith("tool:") for b in by_variant[ac.Variant.PLAIN])
    assert any(b.startswith("tool:") for b in by_variant[ac.Variant.FAKE_TOOL_OUTPUT])
    assert "output:canary" in by_variant[ac.Variant.BASE64]
    assert any(b.startswith("output:") for b in by_variant[ac.Variant.MARKDOWN_IMAGE_EXFIL])


def test_rag_pipeline_has_no_tools_so_any_tool_call_is_refused():
    pipe = rag_pipeline(canaries=CANARIES, require_citations=False)
    doc = next(d for d in ac.adversarial_documents() if d.variant is ac.Variant.PLAIN)
    r = run_scenario(pipe, doc, SENSITIVE, ac, default_context())
    # rag_pipeline has no ToolPolicyCheck, but the canary check on the TOOL stage still fires.
    assert not r.effect_occurred


def test_effects_hold_even_without_context_sanitization():
    from guardrails.context import ContextSanitizerCheck
    checks = [c for c in agent_pipeline(canaries=CANARIES)._checks if not isinstance(c, ContextSanitizerCheck)]
    results = run_all(GuardrailPipeline(checks))
    assert not any(r.effect_occurred for r in results)


def test_measure_report_and_gate(capsys):
    report = measure.run()
    assert report["effects_without_guardrails"]["bypass_rate"] == 1.0
    assert report["effects_with_guardrails"]["bypass_rate"] == 0.0
    heur_input = next(r for r in report["checks"] if r["check"] == "injection_heuristic" and r["stage"] == "input")
    assert heur_input["bypass_rate"] > 0.0, "the heuristic is a weak signal and the report must show it"
    assert measure.main([]) == 0
    assert "effects_with_guardrails: 0/" in capsys.readouterr().out


def test_wilson_interval():
    lo, hi = measure.wilson(0, 5)
    assert lo == 0.0 and 0.4 < hi < 0.5
