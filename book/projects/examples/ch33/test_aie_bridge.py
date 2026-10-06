# path: book/projects/examples/ch33/test_aie_bridge.py
"""The Chapter 33 pipeline driven through aie_core clients, offline."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("aie_core")

from aie_core import FakeLLM, InvalidRequestError, ModelGateway, PricingTable, RetryPolicy
from aie_core.llm.errors import ProviderUnavailableError
from aie_core.llm.gateway import InMemoryResponseCache
from aie_core.embeddings import CachedEmbeddings, FakeEmbeddings

from aie_bridge import (
    INVALID_LABEL,
    embed_fn_from_client,
    embedding_space,
    logprob_confidence,
    prompt_sha256,
    run_classifier,
)
from dataset_builder import RawExample, dedupe_near
from eval_protocol import HoldoutExample, Prediction, SystemRun, cascade, evaluate

T0 = datetime(2026, 1, 5, tzinfo=timezone.utc)
SYSTEM = "Classify the Northwind support ticket into exactly one category. Reply with the category id only."
LABELS = ["vpn", "password", "billing"]


def _raw(i: int, text: str, label: str) -> RawExample:
    return RawExample(id=f"t{i}", text=text, label=label, entity_id=f"acct-{i}", created_at=T0 + timedelta(hours=i))


def test_vocabulary_embedder_collapses_reworded_duplicates_through_aie_core():
    rows = [
        _raw(1, "vpn will not connect from home office", "vpn"),
        _raw(2, "home office: cannot connect to the vpn", "vpn"),
        _raw(3, "invoice total looks wrong this month", "billing"),
    ]
    embedder = FakeEmbeddings(vocabulary=["vpn", "connect", "home", "office", "invoice", "total", "wrong", "month"])
    kept, removed, conflicts = dedupe_near(rows, embed_fn=embed_fn_from_client(embedder), threshold=0.9)
    assert [r.id for r in kept] == ["t1", "t3"] and removed == 1 and conflicts == []


def test_cached_embedder_records_its_space_and_embeds_each_text_once():
    inner = FakeEmbeddings(vocabulary=["vpn", "home"])
    cached = CachedEmbeddings(inner, instruction="passage: ")
    fn = embed_fn_from_client(cached)
    fn(["vpn home", "vpn"])
    fn(["vpn home", "vpn"])  # second build: served from cache
    assert len(inner.calls) == 1 and cached.hits == 2
    space = embedding_space(cached)
    assert space["space_fingerprint"] == cached.space_fingerprint and space["model"] == "fake-embedding"
    assert embedding_space(inner)["space_fingerprint"] == "uncached"


def _holdout() -> list[HoldoutExample]:
    return [
        HoldoutExample(id="h1", text="vpn drops every hour", label="vpn"),
        HoldoutExample(id="h2", text="locked out after password reset", label="password"),
        HoldoutExample(id="h3", text="charged twice for one order", label="billing"),
    ]


def _scripted(answers: dict[str, str]) -> FakeLLM:
    def handler(req):
        return answers[req.metadata["eval.example_id"]]

    return FakeLLM(handler=handler, model="nw-tickets-small-v3")


def test_run_classifier_through_gateway_carries_cost_and_invalid_labels():
    llm = _scripted({"h1": "vpn", "h2": "Password reset", "h3": "billing"})
    pricing = PricingTable({"nw-tickets-small": {"input_per_1m": 0.2, "output_per_1m": 0.8}})  # illustrative
    gw = ModelGateway(llm, pricing=pricing)
    run = run_classifier(gw, _holdout(), name="ft-small", model="nw-tickets-small-v3",
                         system_prompt=SYSTEM, labels=LABELS)
    by_id = run.by_id()
    assert by_id["h1"].label == "vpn" and by_id["h1"].confidence == 1.0
    assert by_id["h2"].label == INVALID_LABEL and by_id["h2"].confidence == 0.0  # outside the taxonomy
    assert all(p.cost_usd > 0 for p in run.predictions)  # priced by the gateway, prefix-matched model name
    # every request used the training-time rendering: same system prompt, raw text as the user turn
    assert {r.messages[0].text for r in llm.requests} == {SYSTEM}
    assert evaluate(_holdout(), run, LABELS).accuracy == pytest.approx(2 / 3)


def test_invalid_labels_escalate_in_a_cascade():
    small = run_classifier(_scripted({"h1": "vpn", "h2": "Password reset", "h3": "billing"}), _holdout(),
                           name="small", model="small", system_prompt=SYSTEM, labels=LABELS)
    strong = SystemRun(name="strong", predictions=[
        Prediction(example_id=h.id, label=h.label, confidence=0.9, cost_usd=0.006) for h in _holdout()
    ])
    composed = cascade(small, strong, threshold=0.5)
    assert evaluate(_holdout(), composed, LABELS).accuracy == 1.0
    assert composed.by_id()["h2"].cost_usd == pytest.approx(0.006)  # only the invalid row paid for the strong model


def test_prompt_mismatch_with_training_card_refuses_to_run():
    llm = _scripted({"h1": "vpn", "h2": "password", "h3": "billing"})
    with pytest.raises(ValueError, match="differs from the one the model was trained with"):
        run_classifier(llm, _holdout(), name="ft", model="ft", system_prompt=SYSTEM + " Be concise.",
                       labels=LABELS, expected_prompt_sha256=prompt_sha256(SYSTEM))
    assert llm.requests == []


def test_provider_errors_become_wrong_answers_not_crashes():
    def handler(req):
        if req.metadata["eval.example_id"] == "h3":
            raise InvalidRequestError("unknown model")
        return "vpn" if req.metadata["eval.example_id"] == "h1" else "password"

    gw = ModelGateway(FakeLLM(handler=handler), retry=RetryPolicy(max_attempts=1))
    run = run_classifier(gw, _holdout(), name="ft", model="ft", system_prompt=SYSTEM, labels=LABELS)
    assert run.by_id()["h3"].label == INVALID_LABEL


def test_logprob_confidence_reads_token_probabilities_from_raw():
    import math

    from aie_core import Completion, Message

    with_lp = Completion(message=Message.assistant("vpn"), raw={"choices": [{"logprobs": {"content": [
        {"token": "v", "logprob": -0.1}, {"token": "pn", "logprob": -0.2}]}}]})
    assert logprob_confidence(with_lp, "vpn") == pytest.approx(math.exp(-0.3))
    assert logprob_confidence(Completion(message=Message.assistant("vpn")), "vpn") == 1.0


def test_a_fallback_answer_does_not_count_for_the_model_under_test():
    primary = FakeLLM(responses=[ProviderUnavailableError("503")] * 9)
    backup = FakeLLM(handler=lambda r: "vpn")
    backup.provider = "backup"
    gw = ModelGateway(primary, fallbacks=[backup])
    run = run_classifier(gw, _holdout()[:1], name="ft", model="nw-tickets-small-v3", system_prompt=SYSTEM,
                         labels=LABELS, expected_provider=primary.provider)
    assert run.predictions[0].label == INVALID_LABEL


def test_a_cached_rerun_is_charged_at_full_price():
    llm = _scripted({"h1": "vpn", "h2": "password", "h3": "billing"})
    pricing = PricingTable({"nw-tickets-small": {"input_per_1m": 0.2, "output_per_1m": 0.8}})  # illustrative
    gw = ModelGateway(llm, pricing=pricing, cache=InMemoryResponseCache())
    first = run_classifier(gw, _holdout(), name="ft", model="nw-tickets-small-v3", system_prompt=SYSTEM, labels=LABELS)
    again = run_classifier(gw, _holdout(), name="ft", model="nw-tickets-small-v3", system_prompt=SYSTEM, labels=LABELS)
    assert [p.cost_usd for p in again.predictions] == [p.cost_usd for p in first.predictions]
