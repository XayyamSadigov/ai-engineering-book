# path: book/projects/examples/ch26/test_ch26.py
"""Tests for the Chapter 26 attack corpus and threat-modeling helpers.

These tests assert *effects*, in the spirit of the chapter: a red-team case
passes when the harmful effect is blocked, even if the model would comply.
They run offline with no model and no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import attack_corpus as ac
import threat_model as tm


# --------------------------------------------------------------------------- #
# attack_corpus
# --------------------------------------------------------------------------- #


def test_every_variant_is_generated():
    docs = ac.adversarial_documents()
    assert {d.variant for d in docs} == set(ac.Variant)
    # Each adversarial document still contains its benign carrier text.
    assert all(len(d.body) > 50 for d in docs)


def test_sensitive_canaries_are_unique_and_well_formed():
    docs = ac.sensitive_documents(seed=1)
    canaries = [d.canary for d in docs]
    assert len(set(canaries)) == len(canaries)
    assert all(ac.CANARY_RE.fullmatch(c) for c in canaries)


def test_canaries_are_deterministic_for_a_seed():
    a = [d.canary for d in ac.sensitive_documents(seed=7)]
    b = [d.canary for d in ac.sensitive_documents(seed=7)]
    assert a == b


def test_html_comment_variant_hides_the_instruction_from_rendered_text():
    doc = next(d for d in ac.adversarial_documents() if d.variant is ac.Variant.HTML_COMMENT)
    assert "<!--" in doc.body and "-->" in doc.body
    # A reviewer who reads only the text outside comments sees the benign carrier.
    import re

    visible = re.sub(r"<!--.*?-->", "", doc.body, flags=re.DOTALL)
    assert "send_reply" not in visible


def test_base64_variant_conceals_then_decodes_to_an_instruction():
    doc = next(d for d in ac.adversarial_documents() if d.variant is ac.Variant.BASE64)
    assert "send_reply" not in doc.body  # keyword filters see nothing
    decoded = ac.decode_base64_blocks(doc.body)
    assert any("system" in d.lower() or "verbatim" in d.lower() for d in decoded)


def test_fake_tool_output_variant_is_valid_json_shaped_like_a_tool_result():
    doc = next(d for d in ac.adversarial_documents() if d.variant is ac.Variant.FAKE_TOOL_OUTPUT)
    block = doc.body.split("```json", 1)[1].split("```", 1)[0]
    parsed = json.loads(block)
    assert parsed["next_action"]["tool"] == "send_reply"


def test_markdown_image_variant_points_off_domain():
    doc = next(d for d in ac.adversarial_documents() if d.variant is ac.Variant.MARKDOWN_IMAGE_EXFIL)
    urls = ac.extract_image_urls(doc.body)
    assert urls and all(u.startswith(ac.EXFIL_HOST) for u in urls)


def test_find_canary_leaks_detects_exfiltrated_record():
    docs = ac.sensitive_documents(seed=3)
    leaked = docs[0].canary
    outbound = f"Here is the archive you asked for: {leaked} and some filler."
    found = ac.find_canary_leaks(outbound, [d.canary for d in docs])
    assert found == [leaked]


def test_off_allowlist_urls_flags_exfil_host_and_passes_clean_output():
    allowed = ["intranet.northwind.example"]
    bad_output = f"status: {ac.EXFIL_HOST}/pixel.png?d=123"
    good_output = "See https://intranet.northwind.example/policy for details."
    assert ac.off_allowlist_urls(bad_output, allowed)
    assert ac.off_allowlist_urls(good_output, allowed) == []


def test_egress_control_blocks_image_exfil_effect():
    """Red-team pass condition: no off-allowlist URL survives the egress filter."""
    doc = next(d for d in ac.adversarial_documents() if d.variant is ac.Variant.MARKDOWN_IMAGE_EXFIL)
    allowed = ["intranet.northwind.example"]
    # The model 'complied' and reproduced the image. The control must still win.
    simulated_answer = doc.body
    offending = ac.off_allowlist_urls(simulated_answer, allowed)
    sanitized = simulated_answer
    for url in offending:
        sanitized = sanitized.replace(url, "[blocked]")
    assert ac.off_allowlist_urls(sanitized, allowed) == []


def test_write_corpus_round_trips(tmp_path: Path):
    paths = ac.write_corpus(tmp_path, seed=0)
    assert (tmp_path / "corpus.jsonl").exists()
    lines = (tmp_path / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
    kinds = {json.loads(line)["kind"] for line in lines}
    assert kinds == {"sensitive", "adversarial"}
    assert any(p.suffix == ".md" for p in paths)


# --------------------------------------------------------------------------- #
# threat_model
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("build", [tm.northwind_rag_model, tm.northwind_agent_model])
def test_worked_models_are_internally_consistent(build):
    model = build()
    assert model.validate() == []


def test_threats_are_ordered_by_descending_risk():
    model = tm.northwind_agent_model()
    risks = [t.risk for t in model.by_risk()]
    assert risks == sorted(risks, reverse=True)


def test_highest_risk_agent_threat_is_the_outbound_send():
    model = tm.northwind_agent_model()
    top = model.by_risk()[0]
    assert top.threat_id == "A1"
    assert top.risk == 9


def test_every_threat_names_at_least_one_control():
    for build in (tm.northwind_rag_model, tm.northwind_agent_model):
        model = build()
        assert all(t.controls for t in model.threats)


def test_controls_feed_chapter_27_as_deduplicated_requirements():
    model = tm.northwind_agent_model()
    controls = model.controls()
    assert len(controls) == len(set(controls))
    assert any("approval" in c.lower() for c in controls)


def test_validate_catches_a_dangling_asset_reference():
    model = tm.northwind_rag_model()
    model.threats.append(
        tm.Threat("X1", "bad", tm.Stride.TAMPERING, "chat-message", ("no-such-asset",),
                  "m", "e", tm.Level.LOW, tm.Level.LOW, ("c",))
    )
    problems = model.validate()
    assert any("no-such-asset" in p for p in problems)


def test_render_markdown_includes_table_and_control_section():
    md = tm.render_markdown(tm.northwind_rag_model())
    assert "## Threat model:" in md
    assert "| ID | STRIDE |" in md
    assert "### Control requirements" in md


def test_render_threat_table_escapes_pipes():
    model = tm.ThreatModel(system="t")
    model.assets = [tm.Asset("a", "d", "internal")]
    model.boundaries = [tm.Boundary("b", "s", "t", "x")]
    model.entry_points = [tm.EntryPoint("e", "b", "d")]
    model.threats = [
        tm.Threat("T1", "pipe|title", tm.Stride.SPOOFING, "e", ("a",),
                  "a|b mechanism", "effect", tm.Level.LOW, tm.Level.LOW, ("c1",))
    ]
    table = tm.render_threat_table(model)
    assert "a\\|b mechanism" in table
