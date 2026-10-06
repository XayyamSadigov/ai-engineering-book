"""Edge cases in expiry, deletion, redaction, and ownership that the main suites do not reach."""
from __future__ import annotations

import pytest

from memorykit import SemanticMemory, UserProfileMemory
from memorykit.conversation import ConversationMemory, ExtractedFact, Turn
from memorykit.models import MemoryKind, MemoryRecord, Owner, Sensitivity, Source
from memorykit.profile import ProfileError


def test_a_guess_restating_a_fact_does_not_shorten_its_life(store, clock, ana):
    profile = UserProfileMemory(store, clock=clock)
    stated = profile.propose(ana, "preferred_language", "es", source=Source.USER_STATED, provenance=["turn:s#1"])
    profile.propose(ana, "preferred_language", "es", source=Source.MODEL_INFERRED, confidence=0.8)
    assert profile.facts(ana)["preferred_language"].expires_at == stated.record.expires_at


def test_a_deleted_memory_with_pii_stays_deleted(store, clock, embedder, ana):
    sem = SemanticMemory(store, embedder, clock=clock)
    text = "VPN outage: call ana on +1 555 123 4567"
    first = sem.remember(ana, text, source=Source.USER_STATED)
    store.delete(ana, first.record.id, reason="forget")
    again = sem.remember(ana, text, source=Source.USER_STATED)
    assert again.decision.value == "reject" and "suppressed_by_user_deletion" in again.reasons


def test_account_erasure_cascades_into_derived_records(store, clock, embedder, ana):
    sem = SemanticMemory(store, embedder, clock=clock)
    base = sem.remember(ana, "ana prefers the berlin office", source=Source.USER_STATED).record
    store.put(MemoryRecord(owner=ana.shared(), kind=MemoryKind.SEMANTIC, content="team note: ana prefers berlin",
                           source=Source.SYSTEM_OF_RECORD, provenance=[f"mem:{base.id}"]))
    assert store.delete_owner(ana) == 1
    assert store.query(ana.shared()) == []
    assert store.tombstones(ana.shared()) == []   # no fingerprint of Ana's data survives erasure


def test_an_expired_guess_cannot_be_confirmed(store, clock, ana):
    profile = UserProfileMemory(store, clock=clock)
    guess = profile.propose(ana, "shift", "night", source=Source.MODEL_INFERRED, confidence=0.8).record
    clock.advance(days=30)
    with pytest.raises(ProfileError):
        profile.confirm(ana, guess.id, provenance="turn:x#1")
    assert "shift" not in profile.facts(ana)


def test_a_restatement_never_lowers_sensitivity(store, clock, embedder, ana):
    sem = SemanticMemory(store, embedder, clock=clock)
    sem.remember(ana, "vpn outage at warehouse", source=Source.USER_STATED)
    clock.advance(hours=1)
    again = sem.remember(ana, "vpn outage at warehouse", source=Source.USER_STATED, sensitivity=Sensitivity.RESTRICTED)
    assert again.record.sensitivity is Sensitivity.RESTRICTED


def test_pii_inside_a_structured_value_is_redacted(store, clock, embedder, ana):
    sem = SemanticMemory(store, embedder, clock=clock)
    out = sem.remember(ana, "contact info", source=Source.USER_STATED, value={"phone": "+1 555 123 4567"})
    assert "555" not in str(store.get(ana, out.record.id).value)


def test_a_record_cannot_change_owner_inside_a_tenant(store, clock, embedder, ana):
    rec = SemanticMemory(store, embedder, clock=clock).remember(ana, "printer label refund",
                                                                source=Source.USER_STATED).record
    with pytest.raises(PermissionError):
        store.put(rec.model_copy(update={"owner": Owner(tenant="retail", user="mallory")}))
    assert store.get(ana, rec.id) is not None


def test_session_facts_must_match_whole_tokens():
    memory = ConversationMemory.__new__(ConversationMemory)
    memory.session_id = "s"
    turn = {0: Turn(index=0, role="user", content="My ticket is INC-4821 and the amount is $1500")}
    assert memory._verify(ExtractedFact(key="ticket_id", value="INC-482", turn_index=0), turn) is None
    assert memory._verify(ExtractedFact(key="amount", value="150", turn_index=0), turn) is None
    assert memory._verify(ExtractedFact(key="ticket_id", value="INC-4821", turn_index=0), turn) is not None
    assert memory._verify(ExtractedFact(key="amount", value="$1500", turn_index=0), turn) is not None
