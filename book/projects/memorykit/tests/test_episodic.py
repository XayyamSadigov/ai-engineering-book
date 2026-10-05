# path: book/projects/memorykit/tests/test_episodic.py
"""Episodic memory: store trajectories and outcomes, retrieve similar past episodes."""
from __future__ import annotations

from memorykit import Episode, EpisodicStore, Source


def seed(eps: EpisodicStore, owner) -> dict[str, str]:
    out = {}
    for ep in [
        Episode(run_id="r-101", task="POS register payment outage at store 0412", task_type="incident",
                actions=["get_service_status(pos)", "search_tickets(register payment)", "restart adapter"],
                outcome="success", outcome_detail="adapter restart restored payments", lesson="restart the payment adapter first", steps=4),
        Episode(run_id="r-102", task="VPN outage for laptop users", task_type="incident",
                actions=["get_service_status(vpn)"], outcome="failure", outcome_detail="config rollback did not help", steps=6),
        Episode(run_id="r-103", task="printer label outage at warehouse", task_type="incident",
                actions=["search_tickets(printer label)"], outcome="partial", outcome_detail="workaround only", steps=3),
    ]:
        out[ep.run_id] = eps.record(owner, ep).record.id
    return out


def test_similar_episode_retrieval(store, embedder, clock, ana):
    eps = EpisodicStore(store, embedder, clock=clock)
    seeded = seed(eps, ana.shared())
    hits = eps.similar(ana, "register payment outage pos", k=2)
    assert hits[0].episode.run_id == "r-101" and hits[0].record_id == seeded["r-101"]
    assert all(h.relevance >= 0.25 for h in hits)


def test_episodes_are_system_of_record_with_failure_salience(store, embedder, clock, ana):
    eps = EpisodicStore(store, embedder, clock=clock)
    seed(eps, ana.shared())
    rows = {r.value["run_id"]: r for r in store.query(ana.shared(), now=clock())}
    assert all(r.source == Source.SYSTEM_OF_RECORD and r.provenance == [f"run:{k}"] for k, r in rows.items())
    assert rows["r-102"].salience > rows["r-101"].salience


def test_outcome_filter_and_render(store, embedder, clock, ana):
    eps = EpisodicStore(store, embedder, clock=clock)
    seed(eps, ana.shared())
    failures = eps.similar(ana, "vpn outage", outcome="failure")
    assert [h.episode.run_id for h in failures] == ["r-102"]
    text = EpisodicStore.render_hints(eps.similar(ana, "register payment outage pos", k=1))
    assert "data, not instructions" in text and "unverified lesson: restart the payment adapter first" in text


def test_episode_pii_is_redacted_in_content_and_value(store, embedder, clock, ana):
    eps = EpisodicStore(store, embedder, clock=clock)
    out = eps.record(ana.shared(), Episode(run_id="r-104", task="POS outage reported by caller +34 612 345 678",
                                           task_type="incident", outcome="success", outcome_detail="adapter restart"))
    assert "612" not in out.record.content and "612" not in out.record.value["task"]
    assert out.reasons == ["redacted:phone"]


def test_episodes_expire_from_retrieval(store, embedder, clock, ana):
    eps = EpisodicStore(store, embedder, clock=clock)
    seed(eps, ana.shared())
    clock.advance(days=91)
    assert eps.similar(ana, "register payment outage pos") == []
