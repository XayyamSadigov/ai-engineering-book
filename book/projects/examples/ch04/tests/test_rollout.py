# path: book/projects/examples/ch04/tests/test_rollout.py
from __future__ import annotations

import json
import shutil

import pytest

from prompts import PromptRegistry, PromptRollout, bucket


@pytest.fixture()
def prompt_dir(tmp_path, root):
    target = tmp_path / "prompt_files"
    shutil.copytree(root / "prompt_files", target)
    return target


def test_assignment_is_sticky_and_close_to_the_requested_fraction(registry):
    rollout = PromptRollout(lambda: registry, canary_percent={"ticket.classify": 10})
    arms = [rollout.arm("ticket.classify", f"user-{i}") for i in range(5000)]
    assert arms == [rollout.arm("ticket.classify", f"user-{i}") for i in range(5000)]
    share = arms.count("canary") / len(arms)
    assert 0.08 < share < 0.12
    assert rollout.select("ticket.classify", "user-1").spec.version in {"1.0.0", "1.1.0"}


def test_buckets_are_salted_by_prompt_id():
    a = [bucket("ticket.classify", f"u{i}") < 10 for i in range(2000)]
    b = [bucket("assist.answer", f"u{i}") < 10 for i in range(2000)]
    assert a != b


def test_no_canary_alias_means_everyone_gets_prod(registry):
    rollout = PromptRollout(lambda: registry, canary_percent={"assist.answer": 100})
    assert {rollout.arm("assist.answer", f"u{i}") for i in range(200)} == {"prod"}


def test_broken_alias_push_keeps_last_known_good(prompt_dir):
    rollout = PromptRollout(lambda: PromptRegistry.from_directory(prompt_dir))
    before = rollout.registry.get("ticket.classify", "prod").ref
    (prompt_dir / "aliases.toml").write_text('["ticket.classify"]\nprod = "9.9.9"\n')
    result = rollout.reload()
    assert not result.ok and "missing version" in result.error
    assert rollout.reload_failures == 1
    assert rollout.registry.get("ticket.classify", "prod").ref == before


def test_good_alias_push_is_picked_up(prompt_dir):
    rollout = PromptRollout(lambda: PromptRegistry.from_directory(prompt_dir))
    (prompt_dir / "aliases.toml").write_text('["ticket.classify"]\nprod = "1.1.0"\n')
    assert rollout.reload().ok and rollout.last_error is None
    assert rollout.registry.get("ticket.classify", "prod").spec.version == "1.1.0"


def test_reload_refuses_an_edited_published_version(prompt_dir, root):
    lock = json.loads((root / "prompts.lock").read_text())
    rollout = PromptRollout(lambda: PromptRegistry.from_directory(prompt_dir), lock=lock)
    path = prompt_dir / "ticket.classify" / "1.0.0.md"
    path.write_text(path.read_text().replace("exactly one category", "one category"))
    result = rollout.reload()
    assert not result.ok and "content changed without a version bump" in result.error


def test_startup_fails_fast_on_a_broken_registry(prompt_dir):
    (prompt_dir / "aliases.toml").write_text('["ticket.classify"]\nprod = "9.9.9"\n')
    with pytest.raises(Exception):
        PromptRollout(lambda: PromptRegistry.from_directory(prompt_dir))
