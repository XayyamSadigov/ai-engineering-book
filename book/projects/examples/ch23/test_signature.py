# path: book/projects/examples/ch23/test_signature.py
from __future__ import annotations

import pytest

from signature import BootstrapFewShot, Field, Predict, Signature

TRIAGE = Signature(
    instruction="Classify a Northwind support ticket.",
    inputs=(Field("ticket", "the ticket text"),),
    outputs=(Field("category", "one of: access, billing, hardware"),
             Field("urgent", "true if the user is blocked", bool)),
)


def scripted_llm(prompt: str) -> str:
    """A fake model: answers correctly only when the prompt carries at least
    one demonstration. This makes the optimizer's effect observable."""
    has_demo = prompt.count("\ncategory:") > 1   # field list uses "- category:", so only labels count
    ticket = prompt.rsplit("ticket:", 1)[1].split("\n", 1)[0].lower()
    if "invoice" in ticket:
        cat = "billing"
    elif "laptop" in ticket:
        cat = "hardware" if has_demo else "access"
    else:
        cat = "access"
    return f"category: {cat}\nurgent: {'true' if 'cannot' in ticket else 'false'}"


def test_render_lists_fields_demos_and_inputs() -> None:
    text = TRIAGE.render({"ticket": "VPN down"}, demos=[{"ticket": "Pay?", "category": "billing", "urgent": False}])
    assert text.startswith("Classify a Northwind support ticket.")
    assert "- category: one of: access, billing, hardware" in text
    assert "ticket: Pay?\ncategory: billing\nurgent: False" in text
    assert text.endswith("ticket: VPN down\ncategory:")


def test_parse_is_typed_and_strict() -> None:
    assert TRIAGE.parse("category: billing\nurgent: true") == {"category": "billing", "urgent": True}
    # first output may come back unlabeled; later ones must be labeled
    assert TRIAGE.parse("hardware\nurgent: no") == {"category": "hardware", "urgent": False}
    bad = Signature("x", (Field("a", "a"),), (Field("n", "a number", int),))
    with pytest.raises(ValueError, match="cannot parse"):
        bad.parse("n: twelve")


def test_optimizer_selects_demos_that_raise_dev_score() -> None:
    def metric(example: dict, pred: dict) -> float:
        return float(example["category"] == pred["category"])

    train = [{"ticket": "Invoice is wrong", "category": "billing"},
             {"ticket": "I cannot log in", "category": "access"}]
    dev = [{"ticket": "My laptop screen died", "category": "hardware"},
           {"ticket": "Invoice shows double charge", "category": "billing"}]

    base = Predict(TRIAGE, scripted_llm)
    opt = BootstrapFewShot(metric, max_demos=2)
    assert opt.evaluate(base, dev) == 0.5          # laptop misclassified without demos

    compiled = opt.compile(base, train, dev)
    assert len(compiled.demos) >= 1
    assert compiled.signature is TRIAGE          # spec unchanged; only demos changed
    assert opt.evaluate(compiled, dev) == 1.0


def test_optimizer_keeps_empty_demos_when_nothing_improves() -> None:
    def metric(example: dict, pred: dict) -> float:
        return float(example["category"] == pred["category"])

    dev = [{"ticket": "Invoice missing", "category": "billing"}]       # already perfect
    base = Predict(TRIAGE, scripted_llm)
    compiled = BootstrapFewShot(metric).compile(base, train=dev, dev=dev)
    assert compiled.demos == []
