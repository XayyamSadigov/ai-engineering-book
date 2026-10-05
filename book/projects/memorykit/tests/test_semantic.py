# path: book/projects/memorykit/tests/test_semantic.py
"""Semantic memory: ranking, filtering, dedupe, isolation, re-embedding, recall evaluation."""
from __future__ import annotations

from aie_core.embeddings import FakeEmbeddings

from memorykit import ConsolidationAction, MemoryKind, RecallCase, SemanticMemory, Source, evaluate_recall

from conftest import VOCAB


def ids(result) -> list[str]:
    return [m.record.id for m in result.memories]


def test_relevance_gate_beats_recency_and_salience(store, embedder, clock, ana):
    mem = SemanticMemory(store, embedder, clock=clock)
    old = mem.remember(ana, "Ana prefers Spanish language", source=Source.USER_STATED).record
    clock.advance(days=90)
    recent = mem.remember(ana, "Ana uses a docking station", source=Source.USER_STATED, salience=1.0).record
    result = mem.recall(ana, "which language does Ana prefer")
    assert ids(result) == [old.id]
    assert recent.id not in ids(result)


def test_recency_breaks_ties_between_equally_relevant(store, embedder, clock, ana):
    mem = SemanticMemory(store, embedder, clock=clock)
    older = mem.remember(ana, "night shift Lisbon", source=Source.USER_STATED).record
    clock.advance(days=60)
    newer = mem.remember(ana, "night shift Berlin", source=Source.USER_STATED).record
    result = mem.recall(ana, "night shift")
    assert ids(result) == [newer.id, older.id]
    assert result.memories[0].relevance == result.memories[1].relevance
    assert result.memories[1].recency < result.memories[0].recency


def test_salience_lifts_equally_relevant_memories(store, embedder, clock, ana):
    mem = SemanticMemory(store, embedder, clock=clock)
    low = mem.remember(ana, "laptop replacement Lisbon", source=Source.USER_STATED, salience=0.1).record
    high = mem.remember(ana, "laptop replacement Berlin", source=Source.USER_STATED, salience=0.9).record
    assert ids(mem.recall(ana, "laptop replacement")) == [high.id, low.id]


def test_inferred_memories_rank_below_stated_ones(store, embedder, clock, ana):
    mem = SemanticMemory(store, embedder, clock=clock)
    guess = mem.remember(ana, "keyboard scanner Lisbon", source=Source.MODEL_INFERRED, confidence=0.9).record
    stated = mem.remember(ana, "keyboard scanner Berlin", source=Source.USER_STATED).record
    assert ids(mem.recall(ana, "keyboard scanner")) == [stated.id, guess.id]


def test_near_duplicates_are_merged(store, embedder, clock, ana):
    mem = SemanticMemory(store, embedder, clock=clock)
    first = mem.remember(ana, "Ana prefers the Spanish language", source=Source.USER_STATED, provenance=["turn:s1#2"])
    clock.advance(days=5)
    second = mem.remember(ana, "Ana prefers Spanish language", source=Source.USER_STATED, provenance=["turn:s4#7"])
    assert first.action == ConsolidationAction.INSERTED and second.action == ConsolidationAction.MERGED
    rows = store.query(ana, kinds=[MemoryKind.SEMANTIC], now=clock())
    assert len(rows) == 1 and rows[0].provenance == ["turn:s1#2", "turn:s4#7"]
    assert rows[0].content == "Ana prefers Spanish language"  # newer wording kept


def test_recall_is_scoped_to_owner_plus_tenant_shared(store, embedder, clock, ana, ben, ana_logistics):
    mem = SemanticMemory(store, embedder, clock=clock)
    mine = mem.remember(ana, "POS register adapter restart", source=Source.USER_STATED).record
    team = mem.remember(ana.shared(), "POS register payment adapter", source=Source.SYSTEM_OF_RECORD).record
    mem.remember(ben, "POS register adapter outage", source=Source.USER_STATED)
    mem.remember(ana_logistics, "POS register adapter restart", source=Source.USER_STATED)
    got = set(ids(mem.recall(ana, "pos register adapter", k=10)))
    assert got == {mine.id, team.id}
    assert set(ids(mem.recall(ana, "pos register adapter", k=10, include_shared=False))) == {mine.id}


def test_deleted_memory_is_not_recalled(store, embedder, clock, ana):
    mem = SemanticMemory(store, embedder, clock=clock)
    r = mem.remember(ana, "Ana works night shift", source=Source.USER_STATED).record
    store.delete(ana, r.id, reason="user request", now=clock())
    assert mem.recall(ana, "night shift").memories == []


def test_embedding_model_change_is_detected_and_backfilled(store, embedder, clock, ana):
    SemanticMemory(store, embedder, clock=clock).remember(ana, "Ana prefers Spanish language", source=Source.USER_STATED)
    v2 = SemanticMemory(store, FakeEmbeddings(vocabulary=VOCAB, model="fake-embedding-v2"), clock=clock)
    result = v2.recall(ana, "language")
    assert result.memories == [] and len(result.needs_reembedding) == 1
    assert v2.reembed(ana, result.needs_reembedding) == 1
    assert len(v2.recall(ana, "language").memories) == 1


def test_recall_evaluation_reports_hits_and_forbidden(store, embedder, clock, ana, ben):
    mem = SemanticMemory(store, embedder, clock=clock)
    lang = mem.remember(ana, "Ana prefers Spanish language", source=Source.USER_STATED).record
    shift = mem.remember(ana, "Ana works night shift warehouse", source=Source.USER_STATED).record
    bens = mem.remember(ben, "Ben prefers Spanish language", source=Source.USER_STATED).record
    cases = [
        RecallCase(name="language", owner=ana, query="preferred language", expected_ids={lang.id}, forbidden_ids={bens.id}),
        RecallCase(name="shift", owner=ana, query="which shift", expected_ids={shift.id}),
        RecallCase(name="vpn", owner=ana, query="vpn", expected_ids={"mem_none"}),
    ]
    report = evaluate_recall(mem, cases, k=3)
    assert report.n == 3 and report.forbidden_rate == 0.0
    assert abs(report.hit_rate - 2 / 3) < 1e-9 and abs(report.mrr - 2 / 3) < 1e-9
