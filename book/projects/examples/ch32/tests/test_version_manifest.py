# path: book/projects/examples/ch32/tests/test_version_manifest.py
from __future__ import annotations

from northwind_triage.version_manifest import VersionManifest, content_hash, versioned


def base() -> VersionManifest:
    return VersionManifest(app="northwind-triage", app_version="0.3.0", git_sha="abc1234",
                           environment="prod", prompts={"triage.classify": "1.0.0#aaa"},
                           models={"classifier": "model-a"}, tool_schemas={"create_ticket": "2.0.0#bbb"})


def test_fingerprint_is_stable_and_order_independent() -> None:
    a = base()
    b = VersionManifest(**{**a.model_dump(), "models": {"classifier": "model-a"},
                           "prompts": {"triage.classify": "1.0.0#aaa"}})
    assert a.fingerprint() == b.fingerprint()
    assert len(a.fingerprint()) == 16


def test_any_component_change_changes_fingerprint() -> None:
    a = base()
    assert a.fingerprint() != a.with_updates(prompts={"triage.classify": "1.1.0#ccc"}).fingerprint()
    assert a.fingerprint() != a.with_updates(index_version="idx-2026-03-01").fingerprint()


def test_with_updates_merges_dicts() -> None:
    m = base().with_updates(models={"judge": "model-j"})
    assert m.models == {"classifier": "model-a", "judge": "model-j"}


def test_span_attributes_are_flat_strings() -> None:
    semconv = base().as_semconv_attributes()
    assert semconv["prompt.id"] == "triage.classify" and semconv["prompt.version"] == "1.0.0#aaa"
    assert not set(semconv) & set(base().as_span_attributes())  # one span can carry both sets
    attrs = base().as_span_attributes()
    assert attrs["version.prompts.triage.classify"] == "1.0.0#aaa"
    assert attrs["version.git_sha"] == "abc1234"
    assert "version.fingerprint" in attrs
    assert all(isinstance(v, str) for v in attrs.values())


def test_diff_supports_one_change_at_a_time() -> None:
    a = base()
    one = a.with_updates(prompts={"triage.classify": "1.1.0#ccc"}, git_sha="def5678")
    assert a.changed_components(one) == {"prompts"}         # git sha ignored by default
    two = one.with_updates(models={"classifier": "model-b"})
    assert a.changed_components(two) == {"prompts", "models"}
    assert a.diff(two)["models.classifier"] == ("model-a", "model-b")


def test_versioned_label_binds_name_to_content() -> None:
    assert versioned("1.0.0", "hello") == f"1.0.0#{content_hash('hello')}"
    assert versioned("1.0.0", "hello") != versioned("1.0.0", "hello!")
