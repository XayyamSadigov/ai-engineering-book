# path: book/projects/p2-semantic-search/tests/test_metrics_eval.py
from __future__ import annotations

import json

import pytest
from aie_core.embeddings import FakeEmbeddings

from semsearch.adapters.numpy_store import NumpyVectorStore
from semsearch.domain.metrics import overlap_at_k, precision_at_k, recall_at_k, reciprocal_rank, unique_in_order
from semsearch.eval.run_eval import LOCAL_FIXTURE, evaluate, load_gold
from semsearch.ingest.loader import load_corpus
from semsearch.ingest.pipeline import ingest
from semsearch.service import SearchService

from .conftest import SHARED_DOCS


def test_worked_example_two_relevant_one_found_at_rank_2():
    ranked = ["x", "rel-1", "y", "z", "w"]
    relevant = ["rel-1", "rel-2"]
    assert recall_at_k(ranked, relevant, 5) == 0.5
    assert precision_at_k(ranked, relevant, 5) == 0.2
    assert reciprocal_rank(ranked, relevant) == 0.5


def test_recall_requires_relevant_items():
    with pytest.raises(ValueError):
        recall_at_k(["a"], [], 1)


def test_unique_in_order_and_overlap():
    assert unique_in_order(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]
    assert overlap_at_k(["a", "b", "q"], ["a", "b", "c"], 3) == pytest.approx(2 / 3)


def test_fixture_is_valid_gold():
    rows = load_gold(LOCAL_FIXTURE)
    assert any(r.forbidden for r in rows)
    assert all(r.required_doc_ids for r in rows)


def _service(tmp_docs=None):
    emb = FakeEmbeddings(dimensions=256)
    store = NumpyVectorStore(emb.dimensions)
    ns = "knowledge:fake-embedding:v1"
    ingest(load_corpus(tmp_docs or SHARED_DOCS), store, emb, ns)
    return SearchService(store, emb, ns, max_k=50)


@pytest.mark.skipif(not SHARED_DOCS.exists(), reason="shared-data not present")
def test_eval_on_corpus_has_no_leaks_and_sane_recall():
    report = evaluate(_service(), load_gold(LOCAL_FIXTURE), gold_path=str(LOCAL_FIXTURE))
    assert report.leaks == []
    assert report.forbidden_rows == 1
    assert report.ingestion_gaps == []
    assert report.recall[10] >= 0.8  # hashing embedder behaves like bag-of-words on this corpus
    assert 0.0 < report.mrr <= 1.0


def test_eval_detects_leak_and_ingestion_gap(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    # The 'restricted' doc is mislabeled as public: the evaluator must flag the leak.
    (docs / "r.md").write_text('---\nid: restricted\ntitle: R\nacl_groups: ["all"]\n---\nincident sev1 target\n', encoding="utf-8")
    gold = tmp_path / "gold.jsonl"
    rows = [
        {"id": "L1", "question": "incident sev1 target", "required_doc_ids": ["restricted"], "user_groups": ["all"],
         "tenant": "shared", "tags": ["forbidden-doc"]},
        {"id": "G1", "question": "anything", "required_doc_ids": ["never-ingested"], "user_groups": ["all"],
         "tenant": "shared", "tags": []},
    ]
    gold.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    report = evaluate(_service(docs), load_gold(gold))
    assert report.leaks == ["L1"]
    assert report.ingestion_gaps == ["G1:never-ingested"]
    assert report.rows == 0
