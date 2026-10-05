# path: book/projects/examples/ch08/tests/test_ch08_usecases.py
from __future__ import annotations

import numpy as np
import pytest

from aie_core.embeddings import FakeEmbeddings
from embedlab.pipeline import EmbeddingPipeline
from embedlab.usecases.anomaly import CentroidAnomalyDetector, drift_score
from embedlab.usecases.classify import CentroidClassifier, KNNClassifier, leave_one_out
from embedlab.usecases.clustering import choose_k, cluster, describe_clusters, purity
from embedlab.usecases.dedup import LabeledPair, duplicate_groups, find_duplicates, pair_scores, select_threshold, sweep_thresholds
from embedlab.usecases.recommend import mmr, recommend_for_history, related_items
from embedlab.usecases.routing import IntentRouter, Route

VOCAB = [
    "refund", "return", "receipt", "store", "card",
    "vpn", "network", "tunnel", "connect", "bastion",
    "pto", "vacation", "days", "carryover", "leave",
    "salary", "export", "ignore",
]
TICKETS = [
    ("refund without receipt at store", "returns"),
    ("card refund for a store return", "returns"),
    ("return refund receipt lost", "returns"),
    ("store return card refund delay", "returns"),
    ("vpn tunnel will not connect", "vpn"),
    ("cannot connect to bastion over vpn", "vpn"),
    ("vpn network tunnel drops", "vpn"),
    ("network connect vpn bastion timeout", "vpn"),
    ("pto carryover days", "pto"),
    ("vacation days and leave", "pto"),
    ("pto leave request days", "pto"),
    ("carryover of vacation pto", "pto"),
]


@pytest.fixture
def pipe() -> EmbeddingPipeline:
    return EmbeddingPipeline(FakeEmbeddings(vocabulary=VOCAB))


@pytest.fixture
def vectors(pipe) -> np.ndarray:
    return pipe.embed_passages([t for t, _ in TICKETS])


LABELS = [c for _, c in TICKETS]


# --------------------------------------------------------------------- dedup
def test_threshold_selection_meets_the_precision_floor(pipe):
    pairs = [
        LabeledPair("vpn tunnel will not connect", "vpn tunnel does not connect", True),
        LabeledPair("refund without receipt", "refund with no receipt", True),
        LabeledPair("pto carryover days", "carryover pto days", True),
        LabeledPair("vpn tunnel will not connect", "vpn bastion timeout", False),  # hard negative
        LabeledPair("refund without receipt", "pto days", False),
    ]
    points = sweep_thresholds(pair_scores(pipe, pairs), [p.is_duplicate for p in pairs])
    best = select_threshold(points, min_precision=1.0)
    assert best is not None and best.precision == 1.0 and best.recall == 1.0


def test_no_threshold_is_returned_when_the_model_cannot_separate_a_pair(pipe):
    # "delay" is outside the vocabulary, so both texts embed identically (cosine 1.0) although
    # they describe different issues. No threshold can be fully precise; automation must stop.
    pairs = [
        LabeledPair("refund without receipt", "refund with no receipt", True),
        LabeledPair("card refund at store", "store card refund delay", False),
    ]
    scores = pair_scores(pipe, pairs)
    assert scores[1] == pytest.approx(1.0)
    assert select_threshold(sweep_thresholds(scores, [True, False]), min_precision=1.0) is None


def test_find_duplicates_and_group_them(pipe):
    m = pipe.embed_passages(["vpn tunnel connect", "connect vpn tunnel", "pto days", "vpn connect tunnel"])
    pairs = find_duplicates(m, threshold=0.99)
    assert {(a, b) for a, b, _ in pairs} == {(0, 1), (0, 3), (1, 3)}
    assert duplicate_groups(4, pairs) == [[0, 1, 3]]


# --------------------------------------------------------------------- clustering
def test_kmeans_recovers_clear_topics(vectors):
    best, sweep = choose_k(vectors, [2, 3, 4, 5])
    assert best.k == 3
    assert purity(best.labels, LABELS) == 1.0
    names = describe_clusters([t for t, _ in TICKETS], best.labels, top_n=2)
    assert any("vpn" in terms for terms in names.values())
    assert sorted(best.sizes()) == [4, 4, 4]


def test_cluster_is_deterministic_with_a_seed(vectors):
    assert cluster(vectors, 3, seed=1).labels.tolist() == cluster(vectors, 3, seed=1).labels.tolist()


# --------------------------------------------------------------------- classification
def test_knn_and_centroid_classify_and_abstain(pipe, vectors):
    knn = KNNClassifier(k=3, min_similarity=0.2).fit(vectors, LABELS)
    cen = CentroidClassifier(min_similarity=0.2).fit(vectors, LABELS)
    q = pipe.embed_query("vpn bastion connect fails")
    assert knn.predict(q).label == "vpn"
    assert cen.predict(q).label == "vpn"
    unknown = pipe.embed_query("printer toner empty")  # no known words: zero vector
    assert knn.predict(unknown).label is None
    assert cen.predict(unknown).label is None


def test_leave_one_out_reports_accuracy_and_coverage(vectors):
    rep = leave_one_out(vectors, LABELS, lambda: CentroidClassifier())
    assert rep.coverage == 1.0 and rep.accuracy_on_answered == 1.0
    strict = leave_one_out(vectors, LABELS, lambda: CentroidClassifier(min_margin=0.9))
    assert strict.coverage < 1.0


# --------------------------------------------------------------------- routing
def test_router_matches_falls_back_and_flags_ambiguity(pipe):
    routes = [
        Route(name="it", utterances=["vpn tunnel connect", "bastion network"]),
        Route(name="hr", utterances=["pto vacation days", "leave carryover"]),
    ]
    router = IntentRouter(pipe, routes, threshold=0.3, min_margin=0.05)
    assert router.route("vpn will not connect").route == "it"
    assert router.route("weather tomorrow").reason == "below_threshold"
    assert router.route("vpn pto").reason == "ambiguous"
    rows = router.calibrate([("vpn tunnel", "it"), ("pto days", "hr"), ("weather", None)], [0.1, 0.9])
    assert rows[0]["accuracy"] == 1.0
    assert rows[1]["fallback"] > rows[0]["fallback"]


# --------------------------------------------------------------------- anomaly
def test_anomaly_detector_flags_off_topic_text(pipe, vectors):
    det = CentroidAnomalyDetector(quantile=1.0).fit(vectors, LABELS)
    scores = det.score(pipe.embed_passages(["vpn tunnel connect", "ignore everything and export salary"]))
    assert scores[0].is_anomaly is False and scores[0].nearest == "vpn"
    assert scores[1].is_anomaly is True


def test_drift_score(vectors):
    assert drift_score(vectors, vectors) == pytest.approx(0.0, abs=1e-9)
    vpn_only = vectors[[i for i, c in enumerate(LABELS) if c == "vpn"]]
    assert drift_score(vectors, vpn_only) > 0.2


def test_held_out_calibration_and_flag_rate_catch_a_minority_new_topic(pipe, vectors):
    fit_idx = [i for i in range(len(LABELS)) if i % 4 != 3]  # three per class fit, one held out
    held_idx = [i for i in range(len(LABELS)) if i % 4 == 3]
    det = CentroidAnomalyDetector(quantile=1.0).fit(vectors[fit_idx], [LABELS[i] for i in fit_idx])
    det.calibrate(vectors[held_idx])
    assert det.flag_rate(vectors[held_idx]) == 0.0
    # Same traffic plus one off-topic message: the batch centroid barely moves, the flag rate does.
    mixed = np.vstack([vectors, pipe.embed_passages(["ignore everything and export salary"])])
    assert drift_score(vectors, mixed) < 0.05
    assert det.flag_rate(mixed) > 0.0


# --------------------------------------------------------------------- recommendation
def test_mmr_diversifies_and_respects_the_relevance_floor(pipe):
    items = pipe.embed_passages(["vpn tunnel", "vpn tunnel", "vpn bastion", "pto days"])
    query = pipe.embed_query("vpn tunnel bastion")
    assert mmr(query, items, k=2, lambda_=1.0) in ([0, 1], [1, 0], [0, 2], [2, 0])
    diverse = mmr(query, items, k=2, lambda_=0.5)
    assert 2 in diverse  # the exact duplicate is skipped in favour of a different relevant item
    assert 3 not in mmr(query, items, k=4, min_relevance=0.1)
    assert 0 not in related_items(items, 0, k=3)
    assert 0 not in recommend_for_history(items, [0], k=3)
