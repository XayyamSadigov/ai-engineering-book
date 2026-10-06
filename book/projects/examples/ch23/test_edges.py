# path: book/projects/examples/ch23/test_edges.py
"""Edge cases the main tests do not reach: failed retries, loose model output, fixture files."""
from __future__ import annotations

from pathlib import Path

import pytest

from ports import AnswerService, Passage, RecordingLLM, ReplayLLM
from runnable import Lambda, RunLog
from signature import BootstrapFewShot, Predict
from test_signature import TRIAGE


def test_failed_retries_are_logged_even_when_they_run_out() -> None:
    def always_times_out(x: str) -> str:
        raise TimeoutError("slow model")

    log = RunLog()
    with pytest.raises(TimeoutError):
        Lambda(always_times_out).with_retry(3).invoke("q", log)
    assert [e["attempt"] for e in log.events] == [1, 2, 3]
    assert all(e["output"].startswith("error:") for e in log.events)


def test_parse_takes_the_first_answer_and_reads_booleans_strictly() -> None:
    assert TRIAGE.parse("billing\nurgent: true\n\nticket: next\ncategory: access") == \
        {"category": "billing", "urgent": True}
    assert TRIAGE.parse("access\nurgent: True.")["urgent"] is True
    with pytest.raises(ValueError):
        TRIAGE.parse("access\nurgent: maybe")
    assert TRIAGE.parse("urgent: true\ncategory: billing") == {"category": "billing", "urgent": True}
    with pytest.raises(ValueError):
        TRIAGE.parse("access")                     # a required field is missing


def test_compile_keeps_the_demos_it_scored() -> None:
    demo = {"ticket": "laptop broken", "category": "hardware", "urgent": False}
    module = Predict(TRIAGE, lambda p: "hardware\nurgent: false" if "\ncategory: hardware" in p else "access\nurgent: false",
                     demos=[demo])
    metric = lambda ex, pred: float(ex["category"] == pred["category"])
    compiled = BootstrapFewShot(metric).compile(module, train=[], dev=[{"ticket": "laptop dead", "category": "hardware"}])
    assert compiled.demos == [demo]


class _Retriever:
    def retrieve(self, query: str, k: int) -> list[Passage]:
        return [Passage(id="hr-01", text="Parental leave is 16 weeks.", source="hr.md")]


def test_an_uncited_or_punctuated_insufficient_reply_abstains() -> None:
    for reply in ("INSUFFICIENT.", "Parental leave is 16 weeks."):
        llm = type("L", (), {"complete": lambda self, p, r=reply: r})()
        assert AnswerService(_Retriever(), llm).answer("parental leave?").abstained
    llm = type("L", (), {"complete": lambda self, p: "Leave is 16 weeks [hr-01, it-07]."})()
    assert AnswerService(_Retriever(), llm).answer("parental leave?").citations == ["hr-01"]


def test_recording_adds_to_an_existing_fixture_file(tmp_path: Path) -> None:
    path = tmp_path / "fixtures.json"
    echo = type("Echo", (), {"complete": lambda self, p: p.upper()})()
    RecordingLLM(echo, path).complete("p1")
    RecordingLLM(echo, path).complete("p2")
    replay = ReplayLLM(path)
    assert replay.complete("p1") == "P1" and replay.complete("p2") == "P2"
