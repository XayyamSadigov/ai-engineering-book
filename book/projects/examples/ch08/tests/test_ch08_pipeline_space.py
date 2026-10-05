# path: book/projects/examples/ch08/tests/test_ch08_pipeline_space.py
from __future__ import annotations

import os

import numpy as np
import pytest

from aie_core.embeddings import FakeEmbeddings, OpenAICompatibleEmbeddings
from aie_core.llm.errors import InvalidRequestError, RateLimitError
from aie_core.llm.gateway import RetryPolicy
from aie_core.observability import InMemoryTracer
from aie_core.settings import Settings
from embedlab.corpus import make_client
from embedlab.pipeline import BatchingEmbeddings, EmbeddingPipeline, NamespacedStore, prepare_text
from embedlab.space import EmbeddingSpace, SpaceMismatchError, VectorIndex, plan_reembed

VOCAB = ["refund", "return", "receipt", "vpn", "network", "tunnel", "pto", "vacation", "days"]


def fake() -> FakeEmbeddings:
    return FakeEmbeddings(vocabulary=VOCAB, model="fake-bow-v1")


def test_prepare_text_is_deterministic():
    assert prepare_text("  Refund\n\n policy\t") == "Refund policy"
    assert prepare_text("x" * 50, max_chars=10) == "x" * 10


def test_query_and_passage_prefixes_are_applied():
    f = fake()
    pipe = EmbeddingPipeline(f, query_prefix="query: ", passage_prefix="passage: ")
    pipe.embed_queries(["refund window"])
    pipe.embed_passages(["refund policy"])
    sent = [t for call in f.calls for t in call]
    assert sent == ["query: refund window", "passage: refund policy"]


def test_cache_serves_repeats_and_meters_only_misses():
    f = fake()
    pipe = EmbeddingPipeline(f, price_per_million_tokens=1_000_000.0)  # 1 USD per token, easy to check
    texts = ["refund receipt", "vpn tunnel", "pto days"]
    first = pipe.embed_passages(texts)
    tokens_after_first = pipe.stats.tokens_sent
    second = pipe.embed_passages(texts)
    assert np.allclose(first, second)
    assert pipe.stats.tokens_sent == tokens_after_first
    assert len(f.calls) == 1
    assert pipe.stats.cost_usd == pytest.approx(tokens_after_first)


def test_identical_texts_in_one_call_are_embedded_once():
    f = fake()
    pipe = EmbeddingPipeline(f)
    out = pipe.embed_passages(["vpn tunnel", "vpn  tunnel", "pto"])
    assert f.calls == [["vpn tunnel", "pto"]]
    assert np.allclose(out[0], out[1])


def test_changing_the_space_never_returns_stale_cached_vectors():
    store: dict[str, bytes] = {}
    a = EmbeddingPipeline(fake(), store=store)
    b = EmbeddingPipeline(fake(), store=store, passage_prefix="passage: ")
    assert a.space.fingerprint != b.space.fingerprint
    a.embed_passages(["refund receipt"])
    b.embed_passages(["refund receipt"])
    assert b.stats.texts_sent == 1  # miss: different namespace
    assert len(NamespacedStore(store, a.space.fingerprint)) == 1
    assert len(NamespacedStore(store, b.space.fingerprint)) == 1


def test_batches_respect_count_and_token_budget():
    f = fake()
    pipe = EmbeddingPipeline(f, max_batch_texts=2, max_batch_tokens=10_000)
    pipe.embed_passages([f"refund {i}" for i in range(5)])
    assert [len(c) for c in f.calls] == [2, 2, 1]

    f2 = fake()
    pipe2 = EmbeddingPipeline(f2, max_batch_texts=100, max_batch_tokens=12)
    pipe2.embed_passages(["refund " * 5 + str(i) for i in range(4)])  # about 6-7 tokens each
    assert all(len(c) == 1 for c in f2.calls)


def test_embedding_spans_record_cache_behaviour():
    tracer = InMemoryTracer()
    pipe = EmbeddingPipeline(fake(), tracer=tracer)
    pipe.embed_passages(["vpn"])
    pipe.embed_passages(["vpn"])
    attrs = [s.attributes for s in tracer.spans]
    assert attrs[0]["cache_misses"] == 1 and attrs[1]["cache_hits"] == 1
    assert attrs[0]["model"] == "fake-bow-v1"
    assert attrs[0]["space"] == pipe.space.fingerprint and attrs[0]["retries"] == 0
    assert attrs[0]["zero_vectors"] == 0


def test_index_refuses_mixed_spaces_and_wrong_dimensions():
    pipe = EmbeddingPipeline(fake())
    index = VectorIndex(pipe.space)
    index.add(["a", "b"], pipe.embed_passages(["refund receipt", "vpn tunnel"]), pipe.space, [{"tenant": "retail"}, {"tenant": "shared"}])
    other = EmbeddingSpace(**{**pipe.space.model_dump(), "model": "other-model"})
    with pytest.raises(SpaceMismatchError):
        index.add(["c"], pipe.embed_passages(["pto"]), other)
    with pytest.raises(SpaceMismatchError):
        index.search(pipe.embed_query("refund"), other)
    with pytest.raises(SpaceMismatchError):
        index.add(["d"], np.ones((1, 3)), pipe.space)
    hits = index.search(pipe.embed_query("refund"), pipe.space, k=2)
    assert hits[0].id == "a"
    shared_only = index.search(pipe.embed_query("refund"), pipe.space, k=2, where=lambda m: m["tenant"] == "shared")
    assert [h.id for h in shared_only] == ["b"]


def test_reembed_plan_lists_what_changed():
    pipe = EmbeddingPipeline(fake())
    index = VectorIndex(pipe.space)
    texts = {"a": "refund receipt policy", "b": "vpn tunnel"}
    index.add(list(texts), pipe.embed_passages(list(texts.values())), pipe.space)
    assert plan_reembed(index, pipe.space, texts, 0.02) is None
    new = EmbeddingSpace(**{**pipe.space.model_dump(), "model": "fake-bow-v2", "dimensions": 12})
    plan = plan_reembed(index, new, texts, 0.02)
    assert plan is not None and plan.items == 2 and plan.estimated_tokens > 0
    assert "model" in plan.reason and "dimensions" in plan.reason


class FlakyClient:
    """Fails the first `failures` calls with a given error, then delegates to the fake."""

    def __init__(self, failures: int, error: Exception) -> None:
        self.inner, self.failures, self.error, self.calls = fake(), failures, error, 0
        self.model, self.dimensions = self.inner.model, self.inner.dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return self.inner.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


def test_transient_provider_errors_are_retried_per_batch():
    sleeps: list[float] = []
    client = FlakyClient(2, RateLimitError("slow down", provider="fake", retry_after_s=0.25))
    batcher = BatchingEmbeddings(client, retry=RetryPolicy(max_attempts=3, jitter=False), sleep=sleeps.append)
    assert len(batcher.embed(["refund", "vpn"])) == 2
    assert batcher.stats.retries == 2 and sleeps == [0.25, 0.25]  # provider hint honored
    assert batcher.stats.requests == 1  # only the successful call is metered as sent


def test_non_retryable_errors_surface_immediately_so_the_caller_can_degrade():
    client = FlakyClient(5, InvalidRequestError("input too long", provider="fake"))
    batcher = BatchingEmbeddings(client, retry=RetryPolicy(max_attempts=3), sleep=lambda _: None)
    with pytest.raises(InvalidRequestError):
        batcher.embed(["refund"])
    assert client.calls == 1


def test_delete_removes_vectors_for_erasure():
    pipe = EmbeddingPipeline(fake())
    index = VectorIndex(pipe.space)
    index.add(["a", "b"], pipe.embed_passages(["refund receipt", "vpn tunnel"]), pipe.space)
    assert index.delete(["a", "missing"]) == 1
    assert index.ids == ["b"] and index.matrix.shape[0] == 1
    assert [h.id for h in index.search(pipe.embed_query("refund"), pipe.space, k=5)] == ["b"]


def test_reembed_plan_estimates_the_migration_window():
    pipe = EmbeddingPipeline(fake())
    index = VectorIndex(pipe.space)
    texts = {"a": "refund receipt policy", "b": "vpn tunnel"}
    index.add(list(texts), pipe.embed_passages(list(texts.values())), pipe.space)
    new = EmbeddingSpace(**{**pipe.space.model_dump(), "passage_prefix": "passage: "})
    plan = plan_reembed(index, new, texts, 0.02, tokens_per_minute=1.0)
    assert plan is not None and plan.estimated_minutes == plan.estimated_tokens
    assert plan_reembed(index, new, texts, 0.02).estimated_minutes is None


def test_settings_select_the_model_without_code_changes():
    fake_client = make_client(Settings(embedding_provider="fake"), corpus_texts=["refund policy", "refund window"])
    assert isinstance(fake_client, FakeEmbeddings) and fake_client.vocabulary == ["refund"]
    real = make_client(Settings(embedding_provider="openai", embedding_model="example-embedding-model", openai_api_key="sk-test"))
    assert isinstance(real, OpenAICompatibleEmbeddings) and real.model == "example-embedding-model"


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("EMBEDDING_PROVIDER", "fake") == "fake", reason="set EMBEDDING_PROVIDER and credentials")
def test_real_model_separates_paraphrases():
    pipe = EmbeddingPipeline(make_client())
    v = pipe.embed_passages(["my computer was taken from my car", "laptop stolen from vehicle", "PTO carryover rules"])
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    assert v[0] @ v[1] > v[0] @ v[2]
