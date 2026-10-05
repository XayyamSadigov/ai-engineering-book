# path: book/projects/memorykit/tests/test_profile.py
"""Profile memory: confirmation flow, consolidation, conflicts, forgetting."""
from __future__ import annotations

from datetime import timedelta

import pytest

from memorykit import ConsolidationAction, Decision, MemoryKind, MemoryStatus, ProfileError, Source, UserProfileMemory


def test_inference_waits_for_confirmation(store, clock, ana):
    profile = UserProfileMemory(store, clock=clock)
    out = profile.propose(ana, "preferred_language", "es", source=Source.MODEL_INFERRED, confidence=0.8,
                          provenance=["turn:s1#5"])
    assert out.decision == Decision.PENDING
    assert profile.facts(ana) == {}
    assert profile.render(ana) == ""
    assert [p.key for p in profile.pending(ana)] == ["preferred_language"]


def test_confirmation_promotes_to_user_stated(store, clock, ana):
    profile = UserProfileMemory(store, clock=clock)
    pending = profile.propose(ana, "preferred_language", "es", source=Source.MODEL_INFERRED, confidence=0.8).record
    clock.advance(days=1)
    out = profile.confirm(ana, pending.id, provenance="turn:s2#1")
    fact = profile.facts(ana)["preferred_language"]
    assert out.decision == Decision.ACCEPT and fact.id == pending.id
    assert fact.source == Source.USER_STATED and "turn:s2#1" in fact.provenance
    assert fact.expires_at == clock() + timedelta(days=365)
    assert profile.pending(ana) == []
    # Confirming must not suppress the value it just confirmed.
    again = profile.propose(ana, "preferred_language", "es", source=Source.USER_STATED)
    assert again.action == ConsolidationAction.REFRESHED


def test_unconfirmed_inference_expires(store, clock, ana):
    profile = UserProfileMemory(store, clock=clock)
    profile.propose(ana, "preferred_language", "es", source=Source.MODEL_INFERRED, confidence=0.8)
    clock.advance(days=15)
    assert profile.pending(ana) == []


def test_rejected_proposal_is_not_proposed_again(store, clock, ana):
    profile = UserProfileMemory(store, clock=clock)
    p = profile.propose(ana, "timezone", "Europe/Berlin", source=Source.MODEL_INFERRED, confidence=0.9).record
    profile.reject(ana, p.id)
    again = profile.propose(ana, "timezone", "Europe/Berlin", source=Source.MODEL_INFERRED, confidence=0.9)
    assert again.decision == Decision.REJECT and again.reasons == ["suppressed_by_user_deletion"]
    # A different value is a different proposal.
    other = profile.propose(ana, "timezone", "Europe/Lisbon", source=Source.MODEL_INFERRED, confidence=0.9)
    assert other.decision == Decision.PENDING


def test_newer_system_of_record_supersedes_user_statement(store, clock, ana):
    profile = UserProfileMemory(store, clock=clock)
    old = profile.propose(ana, "office", "Lisbon", source=Source.USER_STATED, provenance=["turn:s1#2"]).record
    clock.advance(days=30)
    out = profile.propose(ana, "office", "Berlin", source=Source.SYSTEM_OF_RECORD, provenance=["hris:emp-2231"])
    assert out.action == ConsolidationAction.SUPERSEDED and out.conflict and out.replaced == [old.id]
    assert profile.facts(ana)["office"].value == "Berlin"
    history = store.query(ana, kinds=[MemoryKind.PROFILE], key="office", statuses=(MemoryStatus.SUPERSEDED,), now=clock())
    assert [h.id for h in history] == [old.id]


def test_user_statement_does_not_override_system_of_record(store, clock, ana):
    profile = UserProfileMemory(store, clock=clock)
    profile.propose(ana, "manager", "ben", source=Source.SYSTEM_OF_RECORD, provenance=["hris:emp-2231"])
    clock.advance(days=1)
    out = profile.propose(ana, "manager", "carla", source=Source.USER_STATED)
    assert out.action == ConsolidationAction.KEPT_EXISTING and out.conflict and not out.stored
    assert profile.facts(ana)["manager"].value == "ben"


def test_newer_user_statement_replaces_older_one(store, clock, ana):
    profile = UserProfileMemory(store, clock=clock)
    profile.propose(ana, "preferred_language", "es", source=Source.USER_STATED)
    clock.advance(days=10)
    out = profile.propose(ana, "preferred_language", "pt", source=Source.USER_STATED)
    assert out.action == ConsolidationAction.SUPERSEDED
    assert profile.facts(ana)["preferred_language"].value == "pt"


def test_inference_never_displaces_a_stated_fact(store, clock, ana):
    profile = UserProfileMemory(store, clock=clock)
    profile.propose(ana, "preferred_language", "es", source=Source.USER_STATED)
    out = profile.propose(ana, "preferred_language", "en", source=Source.MODEL_INFERRED, confidence=0.95)
    assert out.decision == Decision.PENDING
    assert profile.facts(ana)["preferred_language"].value == "es"


def test_duplicate_refreshes_instead_of_inserting(store, clock, ana):
    profile = UserProfileMemory(store, clock=clock)
    first = profile.propose(ana, "preferred_language", "es", source=Source.USER_STATED, provenance=["turn:s1#2"]).record
    clock.advance(days=200)
    out = profile.propose(ana, "preferred_language", " ES ", source=Source.USER_STATED, provenance=["turn:s9#1"])
    assert out.action == ConsolidationAction.REFRESHED and out.record.id == first.id
    assert out.record.version == 2 and out.record.provenance == ["turn:s1#2", "turn:s9#1"]
    assert out.record.expires_at == clock() + timedelta(days=365)
    assert len(store.query(ana, kinds=[MemoryKind.PROFILE], statuses=tuple(MemoryStatus), now=clock())) == 1


def test_forget_removes_every_version(store, clock, ana):
    profile = UserProfileMemory(store, clock=clock)
    profile.propose(ana, "office", "Lisbon", source=Source.USER_STATED)
    clock.advance(days=1)
    profile.propose(ana, "office", "Berlin", source=Source.SYSTEM_OF_RECORD)
    stones = profile.forget(ana, "office")
    assert len(stones) == 2
    assert store.query(ana, key="office", statuses=tuple(MemoryStatus), include_expired=True, now=clock()) == []


def test_profile_requires_a_user(store, clock, ana):
    with pytest.raises(ProfileError):
        UserProfileMemory(store, clock=clock).facts(ana.shared())
