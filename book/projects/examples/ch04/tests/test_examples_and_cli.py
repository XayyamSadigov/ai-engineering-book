# path: book/projects/examples/ch04/tests/test_examples_and_cli.py
from __future__ import annotations

from aie_core.embeddings import FakeEmbeddings
from prompts import FewShotExample, select_examples

POOL = [
    FewShotExample(input="vpn error 412 when connecting", output='{"category":"vpn_network"}', label="vpn_network"),
    FewShotExample(input="vpn very slow on video calls", output='{"category":"vpn_network"}', label="vpn_network"),
    FewShotExample(input="locked out after password change", output='{"category":"password_mfa"}', label="password_mfa"),
    FewShotExample(input="register declines every card", output='{"category":"pos_payments"}', label="pos_payments"),
]
VOCAB = ["vpn", "error", "slow", "locked", "password", "register", "card", "connecting", "calls"]


def test_selects_similar_examples_with_label_diversity():
    emb = FakeEmbeddings(vocabulary=VOCAB)
    chosen = select_examples(POOL, "vpn error after connecting", emb, k=2, max_per_label=1)
    assert chosen[0].input == "vpn error 412 when connecting"
    assert len({ex.label for ex in chosen}) == 2  # the second vpn example was skipped


def test_token_budget_and_eval_leak_exclusion():
    emb = FakeEmbeddings(vocabulary=VOCAB)
    assert select_examples(POOL, "vpn error", emb, k=3, token_budget=5) == []
    chosen = select_examples(POOL, "vpn error 412 when connecting", emb, k=1, exclude_inputs={"vpn error 412 when connecting"})
    assert chosen and chosen[0].input != "vpn error 412 when connecting"


def test_cli_verify_and_compare(monkeypatch, root, capsys):
    import prompts_cli

    monkeypatch.setenv("LLM_PROVIDER", "fake")
    assert prompts_cli.main(["verify"]) == 0
    cases = str(root / "cases" / "ticket_classify.jsonl")
    assert prompts_cli.main(["compare", "ticket.classify", "--baseline", "1.1.0", "--candidate", "1.2.0", "--cases", cases]) == 0
    assert prompts_cli.main(["compare", "ticket.classify", "--baseline", "1.0.0", "--candidate", "1.1.0", "--cases", cases]) == 1
    out = capsys.readouterr().out
    assert "critical cases regressed: TCK-2026-0026" in out
