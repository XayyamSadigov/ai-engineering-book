# path: book/projects/examples/ch32/tests/test_record_replay.py
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from aie_core.llm.providers import OpenAICompatibleClient

from northwind_triage.adapters.llm_classifier import LLMClassifier
from northwind_triage.application import ClassifierUnavailable
from northwind_triage.domain import Category, parse_triage
from northwind_triage.record_replay import CassetteMiss, RecordReplayTransport

from .conftest import FakeOpenAIUpstream

CASSETTES = Path(__file__).parent / "cassettes"
REPLY = '{"category": "pos_payments", "priority": "P1", "confidence": 0.9, "rationale": "all cards"}'


def make_classifier(transport: RecordReplayTransport, api_key: str = "sk-test-not-a-real-key") -> LLMClassifier:
    client = OpenAICompatibleClient(base_url="https://llm.example.test/v1", api_key=api_key,
                                    default_model="model-a", transport=transport)
    return LLMClassifier(client)


def test_record_then_replay_without_network(tmp_path: Path) -> None:
    cassette = tmp_path / "c.json"
    upstream = FakeOpenAIUpstream(REPLY)
    recorder = make_classifier(RecordReplayTransport(cassette, "record", upstream.transport()))
    first = recorder.classify(system="sys", user="ticket text", model="model-a")
    assert upstream.calls == 1

    replayer = make_classifier(RecordReplayTransport(cassette, "replay"))  # no upstream at all
    second = replayer.classify(system="sys", user="ticket text", model="model-a")
    assert upstream.calls == 1
    assert second.text == first.text
    assert second.served_model == "model-a-2026-01-15"   # adapter decoding ran for real
    assert parse_triage(second.text).category == Category.POS_PAYMENTS


def test_changed_request_is_a_loud_miss(tmp_path: Path) -> None:
    cassette = tmp_path / "c.json"
    make_classifier(RecordReplayTransport(cassette, "record", FakeOpenAIUpstream(REPLY).transport())
                    ).classify(system="sys", user="ticket text", model="model-a")
    replayer = make_classifier(RecordReplayTransport(cassette, "replay"))
    with pytest.raises(CassetteMiss, match="no recording"):
        replayer.classify(system="sys v2", user="ticket text", model="model-a")


def test_credentials_never_reach_the_cassette(tmp_path: Path) -> None:
    cassette = tmp_path / "c.json"
    upstream = FakeOpenAIUpstream(REPLY)
    make_classifier(RecordReplayTransport(cassette, "record", upstream.transport()),
                    api_key="sk-live-SUPERSECRET").classify(system="s", user="u", model="model-a")
    assert upstream.seen_auth == ["Bearer sk-live-SUPERSECRET"]  # it was sent upstream...
    text = cassette.read_text()
    assert "SUPERSECRET" not in text                            # ...but not stored
    assert "org-secret" not in text and "req_secret_123" not in text


def test_scrub_removes_pii_before_disk(tmp_path: Path) -> None:
    cassette = tmp_path / "c.json"
    transport = RecordReplayTransport(cassette, "record", FakeOpenAIUpstream(REPLY).transport(),
                                      scrub=lambda s: s.replace("jordan@northwind.example", "<email>"))
    make_classifier(transport).classify(system="s", user="from jordan@northwind.example", model="model-a")
    assert "jordan@northwind.example" not in cassette.read_text()


def test_volatile_fields_do_not_break_matching(tmp_path: Path) -> None:
    import httpx

    cassette = tmp_path / "c.json"
    rec = RecordReplayTransport(cassette, "record", FakeOpenAIUpstream(REPLY).transport())
    req1 = httpx.Request("POST", "https://x/v1/chat/completions",
                         json={"model": "m", "messages": [], "metadata": {"request_id": "1"}})
    req2 = httpx.Request("POST", "https://x/v1/chat/completions",
                         json={"messages": [], "model": "m", "metadata": {"request_id": "2"}})
    assert rec.key_for(req1) == rec.key_for(req2)


def test_upstream_errors_are_recorded_and_mapped(tmp_path: Path) -> None:
    import httpx

    cassette = tmp_path / "c.json"
    failing = httpx.MockTransport(lambda r: httpx.Response(503, json={"error": {"message": "overloaded"}}))
    with pytest.raises(ClassifierUnavailable):
        make_classifier(RecordReplayTransport(cassette, "record", failing)).classify(
            system="s", user="u", model="model-a")
    with pytest.raises(ClassifierUnavailable):   # the 503 replays, error mapping is exercised offline
        make_classifier(RecordReplayTransport(cassette, "replay")).classify(
            system="s", user="u", model="model-a")


def test_committed_cassette_replays_end_to_end() -> None:
    """The fixture in tests/cassettes is committed. In this book it was recorded against
    FakeOpenAIUpstream because tests must not need keys; in a real project record it once
    against the provider (CASSETTE_MODE=record) and review the diff like code."""
    transport = RecordReplayTransport(CASSETTES / "triage_openai_compat.json", "replay")
    out = make_classifier(transport).classify(
        system="You are the ticket triage component.", user="Register 3 declines every card",
        model="model-a")
    assert parse_triage(out.text).category == Category.POS_PAYMENTS
    assert transport.unused_keys() == []


@pytest.mark.integration
def test_rerecord_against_real_provider(tmp_path: Path) -> None:  # pragma: no cover
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("needs OPENAI_API_KEY")
    client = OpenAICompatibleClient(api_key=key, default_model=os.environ.get("LLM_MODEL", "set-me"),
                                    transport=RecordReplayTransport(tmp_path / "live.json", "record"))
    out = LLMClassifier(client).classify(system="Reply with JSON.", user="VPN drops hourly",
                                         model=os.environ.get("LLM_MODEL", "set-me"))
    assert json.loads((tmp_path / "live.json").read_text())["interactions"]
    assert out.text
