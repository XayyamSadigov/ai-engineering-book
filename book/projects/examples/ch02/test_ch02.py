# path: book/projects/examples/ch02/test_ch02.py
"""Tests for the Chapter 2 experiments. Run offline; tiktoken is optional."""
from __future__ import annotations

import numpy as np
import pytest

import kv_cache_calc as kv
import sampling
import tokenizer_experiment as tok

# ---------------------------------------------------------------------------
# tokenizer_experiment
# ---------------------------------------------------------------------------


def test_heuristic_is_positive_and_monotone_in_length() -> None:
    short = tok.heuristic_token_count("hello world")
    longer = tok.heuristic_token_count("hello world " * 10)
    assert short >= 1
    assert longer > short
    assert tok.heuristic_token_count("") == 0


def test_heuristic_charges_more_for_non_latin_scripts() -> None:
    # Same information in Russian is estimated more expensive than in English,
    # which is the direction a real BPE tokenizer trained mostly on English shows.
    assert tok.heuristic_token_count(tok.RUSSIAN) > tok.heuristic_token_count(tok.ENGLISH)


def test_count_tokens_falls_back_when_encoding_is_unavailable() -> None:
    tc = tok.count_tokens("some text", encoding_name="definitely-not-an-encoding")
    assert tc.method == "heuristic"
    assert not tc.exact
    assert tc.tokens == tok.heuristic_token_count("some text")


def test_count_tokens_reports_its_method() -> None:
    tc = tok.count_tokens(tok.ENGLISH)
    assert tc.tokens > 0
    assert tc.method == "heuristic" or tc.method.startswith("tiktoken:")


def test_measure_produces_one_row_per_sample_with_consistent_fields() -> None:
    rows = tok.measure(tok.SAMPLES)
    assert len(rows) == len(tok.SAMPLES)
    for row in rows:
        assert row.tokens > 0
        assert row.utf8_bytes >= row.chars
        assert row.chars_per_token > 0
    table = tok.format_table(rows)
    assert "counting method" in table
    assert all(r.label in table for r in rows)


def test_pretty_json_costs_more_than_compact_json() -> None:
    rows = {r.label: r for r in tok.measure(tok.SAMPLES)}
    assert rows["JSON, indent=2"].tokens > rows["JSON, compact"].tokens


@pytest.mark.skipif(tok._load_tiktoken(tok.DEFAULT_ENCODING) is None, reason="tiktoken encoding not available offline")
def test_real_tokenizer_penalizes_cyrillic_and_splits_numbers() -> None:
    rows = {r.label: r for r in tok.measure(tok.SAMPLES)}
    assert rows["Russian prose (same text)"].tokens > rows["English prose"].tokens
    pieces = tok.token_pieces("1234567890")
    assert pieces is not None
    assert len(pieces) >= 2  # a ten-digit number is never one token in these vocabularies
    assert "".join(pieces) == "1234567890"


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------


def test_softmax_is_a_distribution_and_shift_invariant() -> None:
    logits = np.array([2.0, 1.0, 0.0])
    p = sampling.softmax(logits)
    assert p.sum() == pytest.approx(1.0)
    assert np.all(p > 0)
    np.testing.assert_allclose(p, sampling.softmax(logits + 100.0))
    # the Primer's hand-computed example
    np.testing.assert_allclose(p, [0.665, 0.245, 0.090], atol=0.001)


def test_temperature_sharpens_below_one_and_flattens_above_one() -> None:
    base = sampling.softmax(sampling.TOY_LOGITS)
    cold = sampling.softmax(sampling.apply_temperature(sampling.TOY_LOGITS, 0.3))
    hot = sampling.softmax(sampling.apply_temperature(sampling.TOY_LOGITS, 2.0))
    assert cold.max() > base.max() > hot.max()
    assert sampling.entropy_bits(cold) < sampling.entropy_bits(base) < sampling.entropy_bits(hot)
    # temperature never changes the ranking of tokens
    assert np.argmax(cold) == np.argmax(base) == np.argmax(hot)


def test_temperature_rejects_zero_and_negative() -> None:
    with pytest.raises(ValueError):
        sampling.apply_temperature(sampling.TOY_LOGITS, 0.0)
    with pytest.raises(ValueError):
        sampling.apply_temperature(sampling.TOY_LOGITS, -1.0)


def test_top_k_keeps_exactly_k_candidates_and_renormalizes() -> None:
    p = sampling.softmax(sampling.TOY_LOGITS)
    filtered = sampling.top_k_filter(p, 3)
    assert sampling.support_size(filtered) == 3
    assert filtered.sum() == pytest.approx(1.0)
    kept = set(np.flatnonzero(filtered))
    assert kept == set(np.argsort(-p)[:3])
    # k larger than the vocabulary is a no-op
    np.testing.assert_allclose(sampling.top_k_filter(p, 100), p)


def test_top_p_adapts_support_to_confidence() -> None:
    peaked = sampling.softmax(np.array([10.0, 0.0, 0.0, 0.0]))
    flat = sampling.softmax(np.array([0.0, 0.0, 0.0, 0.0]))
    assert sampling.support_size(sampling.top_p_filter(peaked, 0.9)) == 1
    assert sampling.support_size(sampling.top_p_filter(flat, 0.9)) == 4
    # the top token always survives, even for a tiny threshold
    assert sampling.support_size(sampling.top_p_filter(flat, 0.01)) == 1


def test_top_p_keeps_smallest_prefix_reaching_threshold() -> None:
    p = np.array([0.5, 0.3, 0.15, 0.05])
    filtered = sampling.top_p_filter(p, 0.8)
    assert set(np.flatnonzero(filtered)) == {0, 1}
    np.testing.assert_allclose(filtered[:2], [0.5 / 0.8, 0.3 / 0.8])


def test_greedy_is_deterministic_and_sample_with_t0_matches_it() -> None:
    assert sampling.greedy(sampling.TOY_LOGITS) == 0
    rng = np.random.default_rng(0)
    assert all(sampling.sample(sampling.TOY_LOGITS, temperature=0.0, rng=rng) == 0 for _ in range(50))


def test_sample_respects_filters_and_tracks_distribution() -> None:
    rng = np.random.default_rng(123)
    draws = np.array([sampling.sample(sampling.TOY_LOGITS, temperature=1.0, top_k=3, rng=rng) for _ in range(4000)])
    assert set(np.unique(draws)) <= {0, 1, 2}
    expected = sampling.top_k_filter(sampling.softmax(sampling.TOY_LOGITS), 3)
    freq = np.bincount(draws, minlength=len(sampling.TOY_LOGITS)) / draws.size
    np.testing.assert_allclose(freq[:3], expected[:3], atol=0.03)


def test_demo_table_renders_every_setting() -> None:
    table = sampling.distribution_table(sampling.TOY_LOGITS, sampling.TOY_VOCAB, sampling.DEMO_SETTINGS)
    for s in sampling.DEMO_SETTINGS:
        assert s.name in table
    assert "entropy (bits)" in table


# ---------------------------------------------------------------------------
# kv_cache_calc
# ---------------------------------------------------------------------------


def test_kv_cache_matches_worked_example() -> None:
    # 32 layers, 8 KV heads, head_dim 128, 16k tokens, BF16: about 2 GiB per sequence.
    b = kv.kv_cache_bytes(layers=32, kv_heads=8, head_dim=128, tokens=16_000, bytes_per_elem=2)
    assert b == 2 * 32 * 8 * 128 * 16_000 * 2
    assert 1.9 * kv.GIB < b < 2.1 * kv.GIB


def test_kv_cache_is_linear_in_tokens_and_zero_for_empty_context() -> None:
    one = kv.kv_cache_bytes(32, 8, 128, 1000, 2)
    two = kv.kv_cache_bytes(32, 8, 128, 2000, 2)
    assert two == 2 * one
    assert kv.kv_cache_bytes(32, 8, 128, 0, 2) == 0
    assert kv.bytes_per_token(32, 8, 128, 2) == one // 1000


def test_fewer_kv_heads_and_smaller_dtype_shrink_the_cache_proportionally() -> None:
    full = kv.kv_cache_bytes(32, 32, 128, 16_000, 2)
    gqa = kv.kv_cache_bytes(32, 8, 128, 16_000, 2)
    fp8 = kv.kv_cache_bytes(32, 8, 128, 16_000, 1)
    assert full == 4 * gqa
    assert gqa == 2 * fp8


def test_max_concurrent_sequences_floors() -> None:
    per_seq = kv.kv_cache_bytes(32, 8, 128, 16_000, 2)
    budget = 40 * kv.GIB
    assert kv.max_concurrent_sequences(budget, per_seq) == budget // per_seq
    assert kv.max_concurrent_sequences(per_seq - 1, per_seq) == 0


def test_kv_cache_rejects_nonsense() -> None:
    with pytest.raises(ValueError):
        kv.kv_cache_bytes(0, 8, 128, 10, 2)
    with pytest.raises(ValueError):
        kv.kv_cache_bytes(32, 8, 128, -1, 2)
    with pytest.raises(ValueError):
        kv.max_concurrent_sequences(10, 0)


def test_cli_prints_report_and_gqa_comparison(capsys: pytest.CaptureFixture[str]) -> None:
    rc = kv.main(
        [
            "--layers", "32", "--kv-heads", "8", "--head-dim", "128", "--tokens", "16000",
            "--concurrency", "16", "--memory-gb", "40", "--query-heads", "32",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "per sequence" in out
    assert "16 concurrent sequences" in out
    assert "fits" in out
    assert "4.0x more" in out
