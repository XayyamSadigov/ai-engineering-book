# path: book/projects/examples/ch08/tests/test_ch08_quality.py
from __future__ import annotations

import pytest

from aie_core.embeddings import FakeEmbeddings
from embedlab.corpus import load_docs, load_jsonl, make_client, parse_front_matter, split_sections
from embedlab.pipeline import EmbeddingPipeline
from embedlab.quality import (
    LabeledQuery,
    evaluate_retrieval,
    nearest_neighbors,
    recall_at_k,
    reciprocal_rank,
    truncation_report,
)

VOCAB = ["refund", "return", "receipt", "vpn", "network", "tunnel", "pto", "vacation", "days", "carryover"]
CORPUS = {
    "returns": "Refund and return rules. Bring the receipt for a refund.",
    "vpn": "VPN network tunnel setup and VPN troubleshooting.",
    "pto": "PTO vacation days and carryover of PTO days.",
}


def pipe() -> EmbeddingPipeline:
    return EmbeddingPipeline(FakeEmbeddings(vocabulary=VOCAB))


def test_rank_metrics():
    rel = frozenset({"b", "d"})
    assert recall_at_k(["a", "b", "c", "d"], rel, 2) == 0.5
    assert recall_at_k(["a", "b", "c", "d"], rel, 4) == 1.0
    assert reciprocal_rank(["a", "b"], rel) == 0.5
    assert reciprocal_rank(["a", "c"], rel) == 0.0


def test_evaluate_retrieval_on_a_tiny_corpus():
    queries = [
        LabeledQuery("refund without receipt", frozenset({"returns"})),
        LabeledQuery("vpn tunnel down", frozenset({"vpn"})),
        LabeledQuery("holiday allowance", frozenset({"pto"})),  # no shared words: bag-of-words misses it
    ]
    rep = evaluate_retrieval(pipe(), queries, list(CORPUS), list(CORPUS.values()), k=1)
    assert rep.recall == pytest.approx(2 / 3)
    assert rep.misses == ["holiday allowance"]


def test_sections_collapse_to_documents():
    ids = ["returns#s1", "returns#s2", "vpn#s1"]
    texts = ["refund receipt", "return refund", "vpn tunnel"]
    q = [LabeledQuery("refund", frozenset({"returns"})), LabeledQuery("vpn", frozenset({"vpn"}))]
    rep = evaluate_retrieval(pipe(), q, ids, texts, k=1, group_of=lambda i: i.split("#")[0])
    assert rep.recall == 1.0 and rep.mrr == 1.0


def test_truncation_report_full_dimension_is_identity():
    p = pipe()
    ids, texts = list(CORPUS), list(CORPUS.values())
    queries = [LabeledQuery("refund receipt", frozenset({"returns"})), LabeledQuery("pto days", frozenset({"pto"}))]
    rows = truncation_report(p.embed_queries([q.query for q in queries]), p.embed_passages(texts), ids, queries, [len(VOCAB), 3], k=1)
    assert rows[0].neighbor_overlap == 1.0 and rows[0].recall == 1.0
    # the first three coordinates only know refund/return/receipt: the PTO query loses its signal
    assert rows[1].recall == 0.5


def test_nearest_neighbors_excludes_the_probe_itself():
    p = pipe()
    m = p.embed_passages(["refund receipt", "refund return", "vpn tunnel"])
    nn = nearest_neighbors(m, ["a", "b", "c"], ["a"], k=1)
    assert nn["a"][0][0] == "b"


def test_front_matter_and_sections_from_shared_data():
    meta, body = parse_front_matter('---\nid: x\nacl_groups: ["hr", "all"]\n---\n# T\n## A\ntext')
    assert meta == {"id": "x", "acl_groups": ["hr", "all"]} and body.startswith("# T")
    docs = load_docs()
    assert len(docs) >= 20
    for d in docs:
        sections = split_sections(d)
        assert len(sections) >= 3, d.id
        assert all(s.text.startswith(d.title) for s in sections)


def test_regression_floor_on_lexical_queries_with_the_default_model():
    """A recall floor on your own labeled queries is the test that catches a bad model swap."""
    docs = load_docs()
    rows = [r for r in load_jsonl("retrieval_pairs.jsonl") if r["style"] == "lexical"]
    queries = [LabeledQuery(r["query"], frozenset(r["relevant"])) for r in rows]
    p = EmbeddingPipeline(make_client())
    rep = evaluate_retrieval(p, queries, [d.id for d in docs], [d.text for d in docs], k=3)
    assert rep.recall >= 0.9, rep.misses
