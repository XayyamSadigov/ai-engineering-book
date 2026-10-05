# path: book/projects/aie_core/tests/test_embeddings.py
import json
import math

import httpx
import numpy as np
import pytest

from aie_core.embeddings import (
    CachedEmbeddings,
    EmbeddingClient,
    FakeEmbeddings,
    OpenAICompatibleEmbeddings,
    cosine_similarity,
    normalize,
    top_k,
)
from aie_core.llm.errors import MalformedResponseError, RateLimitError


def test_math_helpers():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)
    assert cosine_similarity([0, 0], [1, 0]) == 0.0
    n = normalize([3, 4])
    assert n == pytest.approx([0.6, 0.8]) and math.isclose(np.linalg.norm(n), 1.0)
    assert normalize([0, 0]) == [0.0, 0.0]


def test_top_k_orders_and_bounds():
    matrix = [[1, 0], [0.9, 0.1], [0, 1], [-1, 0]]
    result = top_k([1, 0], matrix, k=2)
    assert [i for i, _ in result] == [0, 1]
    assert result[0][1] == pytest.approx(1.0)
    assert top_k([1, 0], matrix, k=10) == sorted(top_k([1, 0], matrix, k=4), key=lambda t: -t[1])
    assert top_k([1, 0], [], k=3) == [] and top_k([1, 0], matrix, k=0) == []


def test_fake_embeddings_hashing_is_deterministic_and_unit_length():
    emb = FakeEmbeddings(dimensions=32)
    a1, a2 = emb.embed(["refund policy", "refund policy"])
    assert a1 == a2 and len(a1) == 32 and math.isclose(np.linalg.norm(a1), 1.0)
    assert emb.embed_query("refund policy") == a1
    assert isinstance(emb, EmbeddingClient)


def test_fake_embeddings_vocabulary_mode_gives_controllable_similarity():
    emb = FakeEmbeddings(vocabulary=["refund", "policy", "deadline", "vpn", "outage"])
    assert emb.dimensions == 5
    q = emb.embed_query("refund policy")
    close = emb.embed_query("refund policy deadline")
    far = emb.embed_query("vpn outage")
    assert cosine_similarity(q, close) > 0.8 > cosine_similarity(q, far) == pytest.approx(0.0)
    assert emb.embed_query("unrelated words only") == [0.0] * 5


def test_cached_embeddings_hits_store():
    inner = FakeEmbeddings(dimensions=8)
    store: dict[str, bytes] = {}
    cached = CachedEmbeddings(inner, store)
    first = cached.embed(["a", "b"])
    second = cached.embed(["b", "c", "a"])
    assert second[2] == first[0] and second[0] == first[1]
    assert cached.hits == 2 and cached.misses == 3
    assert inner.calls == [["a", "b"], ["c"]]
    assert cached.dimensions == 8 and len(store) == 3


def test_openai_compatible_embeddings_mapping():
    def handler(r: httpx.Request):
        payload = json.loads(r.content)
        assert payload["model"] == "emb-1" and payload["dimensions"] == 3
        n = len(payload["input"])
        rows = [{"index": 1, "embedding": [0, 1, 0]}, {"index": 0, "embedding": [1, 0, 0]}][:n]
        if n == 1:
            rows = [{"index": 0, "embedding": [1, 0, 0]}]
        return httpx.Response(200, json={"data": rows, "usage": {"prompt_tokens": 4}})

    client = OpenAICompatibleEmbeddings("emb-1", base_url="http://test/v1", api_key="k", dimensions=3, transport=httpx.MockTransport(handler))
    vectors = client.embed(["first", "second"])
    assert vectors == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]  # re-ordered by index
    assert client.embed_query("x") == [1.0, 0.0, 0.0]


def test_openai_compatible_embeddings_batches_and_errors():
    seen: list[int] = []

    def handler(r: httpx.Request):
        n = len(json.loads(r.content)["input"])
        seen.append(n)
        return httpx.Response(200, json={"data": [{"index": i, "embedding": [1.0, 2.0]} for i in range(n)]})

    client = OpenAICompatibleEmbeddings("e", base_url="http://test/v1", batch_size=2, transport=httpx.MockTransport(handler))
    out = client.embed(["a", "b", "c"])
    assert seen == [2, 1] and len(out) == 3 and client.dimensions == 2

    bad = OpenAICompatibleEmbeddings("e", base_url="http://test/v1", transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"data": []})))
    with pytest.raises(MalformedResponseError):
        bad.embed(["a"])
    limited = OpenAICompatibleEmbeddings("e", base_url="http://test/v1", transport=httpx.MockTransport(lambda r: httpx.Response(429, json={"error": {"message": "rl"}})))
    with pytest.raises(RateLimitError):
        limited.embed(["a"])


def test_cached_embeddings_key_includes_embedding_space():
    store: dict[str, bytes] = {}
    a = CachedEmbeddings(FakeEmbeddings(dimensions=8, model="e"), store)
    b = CachedEmbeddings(FakeEmbeddings(dimensions=16, model="e"), store)  # same model, other space
    c = CachedEmbeddings(FakeEmbeddings(dimensions=8, model="e"), store, instruction="query: ")
    d = CachedEmbeddings(FakeEmbeddings(dimensions=8, model="e"), store, text_prep_version="v2")
    fingerprints = {x.space_fingerprint for x in (a, b, c, d)}
    assert len(fingerprints) == 4 and all(len(f) == 16 for f in fingerprints)
    assert CachedEmbeddings(FakeEmbeddings(dimensions=8, model="e"), {}).space_fingerprint == a.space_fingerprint
    a.embed(["refund policy"])
    b.embed(["refund policy"])
    c.embed(["refund policy"])
    assert len(store) == 3 and b.misses == 1 and c.misses == 1
    assert len(a.embed(["refund policy"])[0]) == 8 and len(b.embed(["refund policy"])[0]) == 16
    assert c.inner.calls == [["query: refund policy"]]  # the instruction is applied on the miss path


def test_cached_embeddings_fingerprint_resolves_innermost_provider():
    real = FakeEmbeddings(dimensions=8, model="e")
    direct = CachedEmbeddings(real, {})
    double = CachedEmbeddings(CachedEmbeddings(real, {}), {})
    assert double.innermost is real
    assert double.space_fingerprint == direct.space_fingerprint
    other = CachedEmbeddings(CachedEmbeddings(FakeEmbeddings(dimensions=8, model="other"), {}), {})
    assert other.space_fingerprint != direct.space_fingerprint


def test_cached_embeddings_rejects_partial_inner_response():
    class Partial:
        provider = "partial"
        model = "p"
        dimensions = 2

        def embed(self, texts):
            return [[1.0, 0.0]] * (len(texts) - 1)

        def embed_query(self, text):
            return [1.0, 0.0]

    cached = CachedEmbeddings(Partial(), {})
    with pytest.raises(MalformedResponseError):
        cached.embed(["a", "b", "c"])


def test_openai_compatible_embeddings_validates_indices():
    def dup(r):
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}, {"index": 0, "embedding": [2.0]}]})

    client = OpenAICompatibleEmbeddings("e", base_url="http://test/v1", transport=httpx.MockTransport(dup))
    with pytest.raises(MalformedResponseError) as info:
        client.embed(["a", "b"])
    assert "indices" in str(info.value)

    def short(r):
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

    client = OpenAICompatibleEmbeddings("e", base_url="http://test/v1", transport=httpx.MockTransport(short))
    with pytest.raises(MalformedResponseError) as info:
        client.embed(["a", "b"])
    assert "sent 2 inputs, got 1" in str(info.value)
