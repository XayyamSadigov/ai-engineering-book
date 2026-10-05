# path: book/projects/examples/ch33/test_ch33.py
"""Offline tests for the Chapter 33 examples. No network, no API keys, no GPUs."""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np
import pytest

from dataset_builder import (
    CleanConfig,
    RawExample,
    SplitConfig,
    assert_no_entity_overlap,
    build_dataset,
    clean,
    cohen_kappa,
    dedupe_exact,
    dedupe_near,
    detect_label_artifacts,
    read_jsonl,
    scrub_pii,
    split,
    to_chat_example,
)
from eval_protocol import (
    HoldoutExample,
    Prediction,
    ShipRule,
    SystemRun,
    calibration_bins,
    cascade,
    choose_threshold,
    decide_ship,
    escalation_rate,
    evaluate,
    macro_f1,
    per_class_prf,
)
from finetune_job import (
    FakeProvider,
    FineTuneFailed,
    Hyperparameters,
    OpenAICompatibleFineTuneProvider,
    ValidationError,
    run_fine_tune,
    validate_chat_jsonl,
)

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
SYSTEM = "Classify the Northwind support ticket into exactly one category. Reply with the category id only."


def ex(i: int, text: str, label: str, entity: str = "acct-1", days: int = 0, **kw) -> RawExample:
    return RawExample(id=f"t{i}", text=text, label=label, entity_id=entity, created_at=T0 + timedelta(days=days), **kw)


def synthetic_corpus(n_per_label: int = 30, labels: tuple[str, ...] = ("billing", "vpn", "password", "hardware")) -> list[RawExample]:
    """Deterministic corpus: distinct phrasings per label, spread over 12 accounts and 90 days."""
    phrases = {
        "billing": ["invoice total looks wrong for last month", "double charged on the retail subscription",
                    "need a copy of the receipt for order", "refund has not arrived after cancellation"],
        "vpn": ["cannot connect to the corporate vpn from home", "vpn client disconnects every ten minutes",
                "tunnel established but intranet pages do not load", "vpn certificate expired on my laptop"],
        "password": ["locked out after too many password attempts", "reset link never arrives in my inbox",
                     "need to change my password before it expires", "sso login loops back to the sign in page"],
        "hardware": ["laptop fan is loud and the machine overheats", "docking station does not detect the monitor",
                     "keyboard keys stopped responding after spill", "battery drains in under an hour"],
    }
    rows: list[RawExample] = []
    i = 0
    for label in labels:
        for k in range(n_per_label):
            base = phrases[label][k % len(phrases[label])]
            text = f"{base}, ticket ref {label[:2].upper()}{k:03d}, user in {'logistics' if k % 2 else 'retail'} unit"
            rows.append(ex(i, text, label, entity=f"acct-{k % 12}", days=k * 3))
            i += 1
    return rows


# ---------------------------------------------------------------------------- cleaning


def test_scrub_pii_replaces_emails_phones_ibans():
    text = "contact jane.doe@example.com or +994 50 123 45 67, iban AZ21NABZ00000000137010001944"
    out = scrub_pii(text)
    assert "<EMAIL>" in out and "<PHONE>" in out and "<IBAN>" in out
    assert "jane.doe" not in out and "137010001944" not in out


def test_clean_drops_by_reason():
    rows = [
        ex(1, "short", "vpn"),
        ex(2, "a perfectly fine ticket about the vpn client", "vpn"),
        ex(3, "customer withdrew consent for training use", "billing", consent=False),
        ex(4, "label not in the taxonomy at all", "zzz"),
        ex(5, "x" * 5000, "vpn"),
    ]
    kept, reasons = clean(rows, CleanConfig(allowed_labels={"vpn", "billing"}))
    assert [r.id for r in kept] == ["t2"]
    assert reasons == Counter({"too_short": 1, "no_consent": 1, "unknown_label": 1, "too_long": 1})


# ---------------------------------------------------------------------------- dedup


def test_dedupe_exact_keeps_earliest_and_reports_conflicts():
    rows = [
        ex(1, "VPN   drops every ten minutes", "vpn", days=2),
        ex(2, "vpn drops every ten minutes", "network", days=1),  # same text, earlier, different label
        ex(3, "unrelated billing question about an invoice", "billing"),
    ]
    kept, removed, conflicts = dedupe_exact(rows)
    assert removed == 1
    assert {r.id for r in kept} == {"t2", "t3"}
    assert conflicts == [("t2", "t1")]


def test_dedupe_near_removes_paraphrase_but_keeps_distinct():
    rows = [
        ex(1, "cannot connect to the corporate vpn from home office this morning", "vpn", days=0),
        ex(2, "cannot connect to the corporate vpn from home office this mornin", "vpn", days=1),  # typo variant
        ex(3, "invoice total looks wrong for last month on the retail account", "billing", days=2),
    ]
    kept, removed, _ = dedupe_near(rows, threshold=0.9)
    assert removed == 1
    assert {r.id for r in kept} == {"t1", "t3"}


def test_dedupe_near_accepts_pluggable_embedder():
    rows = [ex(1, "alpha", "a"), ex(2, "beta", "a"), ex(3, "gamma", "b")]

    def embed(texts: list[str]) -> np.ndarray:
        # alpha and gamma identical, beta orthogonal
        table = {"alpha": [1.0, 0.0], "beta": [0.0, 1.0], "gamma": [1.0, 0.0]}
        return np.array([table[t] for t in texts])

    kept, removed, conflicts = dedupe_near(rows, embed_fn=embed, threshold=0.99)
    assert removed == 1 and {r.id for r in kept} == {"t1", "t2"}
    assert conflicts == [("t1", "t3")]  # labels disagree: exactly the pair a reviewer must look at


# ---------------------------------------------------------------------------- splits


def test_entity_split_has_no_overlap_and_is_deterministic():
    rows = synthetic_corpus()
    s1 = split(rows, SplitConfig(seed=3))
    s2 = split(rows, SplitConfig(seed=3))
    assert_no_entity_overlap(s1)
    assert [r.id for r in s1.train] == [r.id for r in s2.train]
    assert len(s1.train) + len(s1.val) + len(s1.test) == len(rows)
    assert s1.train and s1.val and s1.test
    # fractions are approximate because whole entities move together
    assert abs(len(s1.train) / len(rows) - 0.8) < 0.1


def test_entity_split_is_independent_of_input_order():
    rows = synthetic_corpus()
    a = split(rows, SplitConfig(seed=11))
    b = split(list(reversed(rows)), SplitConfig(seed=11))
    assert {r.id for r in a.test} == {r.id for r in b.test}


def test_time_split_puts_future_in_test():
    rows = synthetic_corpus()
    cutoff = T0 + timedelta(days=60)
    s = split(rows, SplitConfig(strategy="time", time_cutoff=cutoff))
    assert all(r.created_at >= cutoff for r in s.test)
    assert all(r.created_at < cutoff for r in s.train + s.val)
    assert s.test


def test_assert_no_entity_overlap_raises():
    from dataset_builder import Splits

    bad = Splits(train=[ex(1, "a ticket about vpn access", "vpn", entity="acct-9")],
                 val=[], test=[ex(2, "another ticket about vpn access", "vpn", entity="acct-9")])
    with pytest.raises(ValueError, match="acct-9"):
        assert_no_entity_overlap(bad)


# ---------------------------------------------------------------------------- quality checks


def test_detect_label_artifacts_finds_leaked_phrase():
    rows = synthetic_corpus(n_per_label=25)
    # Simulate an agent macro that stamps "autorouted" on every billing ticket.
    rows = [r.model_copy(update={"text": r.text + " autorouted"}) if r.label == "billing" else r for r in rows]
    hits = detect_label_artifacts(rows, min_support=20, min_purity=0.98)
    tokens = {h.token: h for h in hits}
    assert "autorouted" in tokens and tokens["autorouted"].label == "billing"
    # Common tokens that appear across labels are not flagged.
    assert "ticket" not in tokens and "user" not in tokens


def test_cohen_kappa_bounds():
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == pytest.approx(1.0)
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        cohen_kappa(["a"], [])


# ---------------------------------------------------------------------------- end-to-end build


def test_build_dataset_writes_jsonl_and_card(tmp_path: Path):
    rows = synthetic_corpus()
    rows.append(ex(999, rows[0].text, rows[0].label, entity=rows[0].entity_id, days=50))  # exact dup
    result = build_dataset(rows, out_dir=tmp_path, system_prompt=SYSTEM)
    card = result.card
    assert card["dedup"]["exact_removed"] == 1
    assert sum(card["sizes"].values()) == len(rows) - 1 - card["dedup"]["near_removed"]
    assert set(card["labels"]) == {"billing", "vpn", "password", "hardware"}
    train_rows = read_jsonl(tmp_path / "train.jsonl")
    assert len(train_rows) == card["sizes"]["train"]
    first = train_rows[0]["messages"]
    assert [m["role"] for m in first] == ["system", "user", "assistant"]
    assert first[0]["content"] == SYSTEM and first[2]["content"] in card["labels"]
    assert set(first[0].keys()) == {"role", "content"}  # meta never leaks into the training payload
    assert (tmp_path / "data_card.json").exists()
    assert card["files"]["train"]["sha256"]
    assert validate_chat_jsonl(tmp_path / "train.jsonl")["examples"] == card["sizes"]["train"]


def test_to_chat_example_matches_inference_shape():
    chat = to_chat_example(ex(1, "vpn keeps dropping", "vpn"), SYSTEM, user_template="Ticket:\n{text}")
    assert chat.messages[1].content == "Ticket:\nvpn keeps dropping"
    assert chat.messages[-1].role == "assistant" and chat.messages[-1].content == "vpn"


# ---------------------------------------------------------------------------- eval protocol


def test_macro_f1_and_per_class_recall_known_values():
    y_true = ["a", "a", "b", "b", "c", "c"]
    y_pred = ["a", "a", "b", "a", "c", "b"]
    prf = per_class_prf(y_true, y_pred, ["a", "b", "c"])
    assert prf["a"]["recall"] == pytest.approx(1.0) and prf["a"]["precision"] == pytest.approx(2 / 3)
    assert prf["b"]["recall"] == pytest.approx(0.5) and prf["c"]["recall"] == pytest.approx(0.5)
    # F1: a=0.8, b=0.5, c=2/3 -> macro = 0.6556
    assert macro_f1(y_true, y_pred, ["a", "b", "c"]) == pytest.approx((0.8 + 0.5 + 2 / 3) / 3)


def test_calibration_bins_and_ece():
    confs = [0.95, 0.95, 0.95, 0.95, 0.55, 0.55]
    correct = [True, True, True, False, True, False]
    bins, ece = calibration_bins(confs, correct, n_bins=10)
    top = bins[-1]
    assert top.count == 4 and top.accuracy == pytest.approx(0.75) and top.mean_confidence == pytest.approx(0.95)
    # ECE = 4/6*|0.75-0.95| + 2/6*|0.5-0.55|
    assert ece == pytest.approx(4 / 6 * 0.2 + 2 / 6 * 0.05)


def holdout_and_runs():
    labels = ["billing", "vpn", "password", "hardware"]
    holdout = [HoldoutExample(id=f"h{i}", text="t", label=labels[i % 4], slices={"lang": "en" if i % 3 else "de"}) for i in range(40)]
    strong = SystemRun(name="strong-prompted", predictions=[
        Prediction(example_id=h.id, label=h.label, confidence=0.9, latency_ms=1800, cost_usd=0.0040) for h in holdout])
    # candidate: wrong on 2 of 40, cheap and fast
    cand_preds = []
    for i, h in enumerate(holdout):
        wrong = i in (5, 17)
        cand_preds.append(Prediction(example_id=h.id, label=("vpn" if h.label != "vpn" else "billing") if wrong else h.label,
                                     confidence=0.6 if wrong else 0.97, latency_ms=250, cost_usd=0.0003))
    candidate = SystemRun(name="ft-small", predictions=cand_preds)
    return labels, holdout, strong, candidate


def test_evaluate_report_fields_and_slices():
    labels, holdout, strong, candidate = holdout_and_runs()
    rep = evaluate(holdout, candidate, labels)
    assert rep.n == 40 and rep.coverage == 1.0
    assert 0.9 < rep.macro_f1 < 1.0
    assert set(rep.slice_macro_f1["lang"]) == {"en", "de"}
    assert rep.latency_p95_ms == 250 and rep.cost_per_1k_usd == pytest.approx(0.3)


def test_decide_ship_passes_and_fails_for_the_right_reasons():
    labels, holdout, strong, candidate = holdout_and_runs()
    ref = evaluate(holdout, strong, labels)
    cand = evaluate(holdout, candidate, labels)
    rule = ShipRule(max_macro_f1_drop_vs_reference=0.08, critical_labels=["password"], min_critical_recall=0.8,
                    min_cost_reduction=0.5, max_latency_p95_ms=500, max_ece=0.2, max_slice_drop=0.15)
    assert decide_ship(cand, ref, rule).ship is True
    strict = rule.model_copy(update={"max_macro_f1_drop_vs_reference": 0.0, "min_cost_reduction": 0.99})
    decision = decide_ship(cand, ref, strict, regression_pass_rate=0.9)
    assert decision.ship is False
    joined = " ".join(decision.reasons)
    assert "macro-F1" in joined and "cost ratio" in joined and "regression suite" in joined


def test_cascade_improves_quality_and_threshold_is_chosen_on_validation():
    labels, holdout, strong, candidate = holdout_and_runs()
    composed = cascade(candidate, strong, threshold=0.7)
    assert evaluate(holdout, composed, labels).accuracy == pytest.approx(1.0)
    assert escalation_rate(candidate, 0.7) == pytest.approx(2 / 40)
    escalated = composed.by_id()["h5"]
    assert escalated.cost_usd == pytest.approx(0.0043) and escalated.latency_ms == 2050
    t = choose_threshold(holdout, candidate, strong, labels, target_macro_f1=0.99)
    assert t is not None and 0.6 < t <= 0.7


# ---------------------------------------------------------------------------- fine-tune job


def write_train_file(path: Path, n: int = 12) -> Path:
    with path.open("w") as fh:
        for i in range(n):
            fh.write(json.dumps({"messages": [{"role": "system", "content": SYSTEM},
                                              {"role": "user", "content": f"ticket {i} about vpn"},
                                              {"role": "assistant", "content": "vpn"}]}) + "\n")
    return path


def test_validate_chat_jsonl_rejects_bad_rows(tmp_path: Path):
    good = write_train_file(tmp_path / "ok.jsonl")
    assert validate_chat_jsonl(good)["examples"] == 12
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"messages": [{"role": "user", "content": "no assistant turn"}]}\n' * 12)
    with pytest.raises(ValidationError, match="last message must be from the assistant"):
        validate_chat_jsonl(bad)
    few = write_train_file(tmp_path / "few.jsonl", n=3)
    with pytest.raises(ValidationError, match="at least 10"):
        validate_chat_jsonl(few)


def test_fake_provider_lifecycle_yields_model_id(tmp_path: Path):
    train = write_train_file(tmp_path / "train.jsonl")
    provider = FakeProvider()
    statuses: list[str] = []
    record = run_fine_tune(provider, train, "small-base", hyperparameters=Hyperparameters(n_epochs=2),
                           suffix="nw-tickets", sleep=lambda _: None, on_status=lambda j: statuses.append(j.status))
    assert record.fine_tuned_model.startswith("ft:small-base:nw-tickets:")
    assert statuses == ["queued", "running", "succeeded"] or statuses == ["validating", "queued", "running", "succeeded"]
    assert record.trained_tokens and record.training_file_sha256
    assert provider.calls.count("upload") == 1


def test_fake_provider_failure_raises_with_events(tmp_path: Path):
    train = write_train_file(tmp_path / "train.jsonl")
    provider = FakeProvider(fail_at="running")
    with pytest.raises(FineTuneFailed) as info:
        run_fine_tune(provider, train, "small-base", sleep=lambda _: None)
    assert info.value.job.status == "failed"
    assert any(e.level == "error" for e in info.value.events)


def test_openai_compatible_adapter_maps_endpoints_without_network(tmp_path: Path):
    train = write_train_file(tmp_path / "train.jsonl")
    seen: list[tuple[str, str]] = []
    polls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        assert request.headers["authorization"] == "Bearer test-key"
        if request.url.path == "/v1/files":
            return httpx.Response(200, json={"id": "file-abc", "purpose": "fine-tune"})
        if request.url.path == "/v1/fine_tuning/jobs" and request.method == "POST":
            body = json.loads(request.content)
            assert body["training_file"] == "file-abc" and body["model"] == "small-base"
            assert body["hyperparameters"] == {"n_epochs": 2}
            return httpx.Response(200, json={"id": "ftjob-1", "status": "validating_files", "model": "small-base",
                                             "training_file": "file-abc", "created_at": 1_767_225_600,
                                             "hyperparameters": {"n_epochs": 2}})
        if request.url.path == "/v1/fine_tuning/jobs/ftjob-1":
            polls["n"] += 1
            done = polls["n"] >= 2
            return httpx.Response(200, json={"id": "ftjob-1", "status": "succeeded" if done else "running",
                                             "model": "small-base", "training_file": "file-abc",
                                             "created_at": 1_767_225_600, "hyperparameters": {"n_epochs": 2},
                                             "fine_tuned_model": "ft:small-base:org:xyz" if done else None,
                                             "trained_tokens": 12345 if done else None})
        return httpx.Response(404, json={"error": "unexpected path"})

    provider = OpenAICompatibleFineTuneProvider("https://example.invalid/v1", "test-key",
                                                transport=httpx.MockTransport(handler))
    record = run_fine_tune(provider, train, "small-base", hyperparameters=Hyperparameters(n_epochs=2), sleep=lambda _: None)
    assert record.fine_tuned_model == "ft:small-base:org:xyz" and record.trained_tokens == 12345
    assert seen[0] == ("POST", "/v1/files") and seen[1] == ("POST", "/v1/fine_tuning/jobs")
    assert seen.count(("GET", "/v1/fine_tuning/jobs/ftjob-1")) == 2


# ----------------------------------------------------------------------------- job runner resilience
def _job_json(status: str, done: bool = False) -> dict:
    return {"id": "ftjob-9", "status": status, "model": "small-base", "training_file": "file-abc",
            "created_at": 1_767_225_600, "hyperparameters": {"n_epochs": 2},
            "fine_tuned_model": "ft:small-base:org:r9" if done else None, "trained_tokens": 99 if done else None}


def test_poll_survives_transient_errors_and_resumes_existing_job(tmp_path: Path):
    from aie_core.llm.errors import ProviderUnavailableError  # noqa: F401 - documents the mapped class

    train = write_train_file(tmp_path / "train.jsonl")
    script = iter([httpx.Response(200, json=_job_json("running")),
                   httpx.Response(503, json={"error": {"message": "overloaded"}}),
                   httpx.Response(429, headers={"retry-after": "7"}, json={"error": {"message": "slow down"}}),
                   httpx.Response(200, json=_job_json("succeeded", done=True))])
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        return next(script)

    provider = OpenAICompatibleFineTuneProvider("https://example.invalid/v1", "k", transport=httpx.MockTransport(handler))
    sleeps: list[float] = []
    record = run_fine_tune(provider, train, "small-base", resume_job_id="ftjob-9", sleep=sleeps.append, poll_interval_s=30)
    assert record.fine_tuned_model == "ft:small-base:org:r9" and record.training_file_id == "file-abc"
    assert all(method == "GET" for method, _ in seen)  # resumed: no upload, no second paid job
    assert sleeps == [30, 7.0]  # the 503 waited the poll interval, the 429 honored Retry-After


def test_non_retryable_poll_error_raises_immediately(tmp_path: Path):
    from aie_core.llm.errors import InvalidRequestError

    train = write_train_file(tmp_path / "train.jsonl")
    provider = OpenAICompatibleFineTuneProvider(
        "https://example.invalid/v1", "bad-key",
        transport=httpx.MockTransport(lambda r: httpx.Response(401, json={"error": {"message": "bad key"}})),
    )
    with pytest.raises(InvalidRequestError):
        run_fine_tune(provider, train, "small-base", resume_job_id="ftjob-9", sleep=lambda _: None)


def test_poll_timeout_keeps_job_id_and_does_not_cancel(tmp_path: Path):
    from finetune_job import FineTunePollTimeout

    train = write_train_file(tmp_path / "train.jsonl")
    provider = FakeProvider(polls_per_stage=50)
    with pytest.raises(FineTunePollTimeout) as info:
        run_fine_tune(provider, train, "small-base", sleep=lambda _: None, max_polls=3)
    assert info.value.job.id and info.value.job.status not in {"succeeded", "failed", "cancelled"}
    assert "cancel" not in provider.calls
