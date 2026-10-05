# path: book/projects/examples/ch04/tests/test_registry.py
from __future__ import annotations

import json
import shutil

import pytest

from aie_core.llm.types import Role
from prompts import PromptDefinitionError, PromptNotFoundError, PromptRegistry, load_prompt_text

MINIMAL = """+++
id = "demo.echo"
version = "1.0.0"
[variables.text]
+++
=== system ===
Echo.
=== user ===
{{ text }}
"""


def test_directory_loads_every_version(registry):
    assert registry.ids() == ["assist.answer", "judge.groundedness", "ticket.classify"]
    assert registry.versions("ticket.classify") == ["1.0.0", "1.1.0", "1.2.0"]


def test_lookup_rules(registry):
    assert registry.get("ticket.classify").spec.version == "1.1.0"  # 1.2.0 is a draft
    assert registry.get("ticket.classify", "1.2.0").spec.status == "draft"
    assert registry.get("ticket.classify", "prod").spec.version == "1.0.0"
    assert registry.get("ticket.classify", "canary").spec.version == "1.1.0"
    with pytest.raises(PromptNotFoundError):
        registry.get("ticket.classify", "9.9.9")
    with pytest.raises(PromptNotFoundError):
        registry.get("no.such.prompt")


def test_committed_lock_matches_the_files(registry, root):
    """The CI check: a published prompt cannot change without a version bump."""
    lock = json.loads((root / "prompts.lock").read_text())
    assert registry.verify_lock(lock) == []
    assert set(lock) == set(registry.lock())


def test_editing_a_published_version_is_detected(tmp_path, root):
    shutil.copytree(root / "prompt_files", tmp_path / "p")
    lock = PromptRegistry.from_directory(tmp_path / "p").lock()
    f = tmp_path / "p" / "ticket.classify" / "1.0.0.md"
    f.write_text(f.read_text().replace("exactly one category", "one category"))
    problems = PromptRegistry.from_directory(tmp_path / "p").verify_lock(lock)
    assert problems == ["ticket.classify@1.0.0: content changed without a version bump"]


def test_schema_edit_changes_the_hash(tmp_path, root):
    shutil.copytree(root / "prompt_files", tmp_path / "p")
    before = PromptRegistry.from_directory(tmp_path / "p").lock()
    schema = tmp_path / "p" / "schemas" / "ticket_classification.json"
    data = json.loads(schema.read_text())
    data["properties"]["evidence"]["maxLength"] = 50
    schema.write_text(json.dumps(data))
    after = PromptRegistry.from_directory(tmp_path / "p").lock()
    changed = {k for k in before if before[k] != after[k]}
    assert changed == {"ticket.classify@1.0.0", "ticket.classify@1.1.0"}  # 1.2.0 is a draft, not locked


def test_drafts_are_editable_and_not_servable_by_alias(tmp_path, root):
    shutil.copytree(root / "prompt_files", tmp_path / "p")
    lock = PromptRegistry.from_directory(tmp_path / "p").lock()
    assert "ticket.classify@1.2.0" not in lock
    draft = tmp_path / "p" / "ticket.classify" / "1.2.0.md"
    draft.write_text(draft.read_text().replace("Task:", "Task (draft edit):"))
    assert PromptRegistry.from_directory(tmp_path / "p").verify_lock(lock) == []
    aliases = tmp_path / "p" / "aliases.toml"
    aliases.write_text(aliases.read_text().replace('canary = "1.1.0"', 'canary = "1.2.0"'))
    with pytest.raises(PromptDefinitionError, match="draft"):
        PromptRegistry.from_directory(tmp_path / "p")


def test_whitespace_only_hash_is_stable_across_line_endings():
    assert load_prompt_text(MINIMAL).content_hash == load_prompt_text(MINIMAL.replace("\n", "\r\n")).content_hash


def test_front_matter_is_validated():
    with pytest.raises(PromptDefinitionError, match="invalid front matter"):
        load_prompt_text(MINIMAL.replace('version = "1.0.0"', 'version = "1.0"'))
    with pytest.raises(PromptDefinitionError, match="invalid front matter"):
        load_prompt_text(MINIMAL.replace('version = "1.0.0"\n', 'version = "1.0.0"\ntemprature = 0.2\n'))
    with pytest.raises(PromptDefinitionError, match="missing front matter"):
        load_prompt_text("=== user ===\nhi")
    with pytest.raises(PromptDefinitionError, match="before the first section"):
        load_prompt_text(MINIMAL.replace("=== system ===", "stray\n=== system ==="))


def test_yaml_front_matter():
    text = "---\nid: demo.echo\nversion: 1.0.0\nvariables:\n  text: {max_chars: 5}\n---\n=== user ===\n{{ text }}\n"
    try:
        import yaml  # noqa: F401
    except ImportError:
        with pytest.raises(PromptDefinitionError, match="PyYAML"):
            load_prompt_text(text)
        return
    pv = load_prompt_text(text)
    assert pv.spec.variables["text"].max_chars == 5


def test_file_path_must_match_id_and_version(tmp_path):
    (tmp_path / "demo.echo").mkdir()
    (tmp_path / "demo.echo" / "2.0.0.md").write_text(MINIMAL)
    with pytest.raises(PromptDefinitionError, match="path must be"):
        PromptRegistry.from_directory(tmp_path)


def test_duplicate_versions_rejected():
    pv = load_prompt_text(MINIMAL)
    with pytest.raises(PromptDefinitionError, match="duplicate"):
        PromptRegistry([pv, pv])


def test_output_schema_cannot_escape_root(tmp_path):
    text = MINIMAL.replace('version = "1.0.0"\n', 'version = "1.0.0"\noutput_schema = "../secrets.json"\n')
    with pytest.raises(PromptDefinitionError, match="escapes"):
        load_prompt_text(text, root=tmp_path)


def test_to_request_carries_the_decoding_policy_and_identity(registry, ticket_cases):
    pv = registry.get("ticket.classify", "1.1.0")
    req = pv.render(ticket_cases[0].variables).to_request()
    assert req.temperature == 0.0 and req.max_tokens == 120
    assert req.response_schema["required"] == ["category", "evidence"]
    assert req.metadata["prompt.id"] == "ticket.classify"
    assert req.metadata["prompt.version"] == "1.1.0"
    assert req.metadata["prompt.hash"] == pv.content_hash[:16]
    overridden = pv.render(ticket_cases[0].variables).to_request(temperature=0.7)
    assert overridden.temperature == 0.7 and overridden.metadata["prompt.overrides"] == ["temperature"]


@pytest.mark.parametrize("version", ["1.0.0", "1.1.0", "1.2.0"])
def test_system_prefix_is_stable_across_inputs(registry, ticket_cases, version):
    """Cache-friendly layout: nothing request-specific may leak into the leading messages."""
    pv = registry.get("ticket.classify", version)
    prefixes = {
        tuple(m.text for m in pv.render(c.variables).messages[:-1]) for c in ticket_cases
    }
    assert len(prefixes) == 1
    last = pv.render(ticket_cases[0].variables).messages[-1]
    assert last.role == Role.USER and "<untrusted_data" in last.text
