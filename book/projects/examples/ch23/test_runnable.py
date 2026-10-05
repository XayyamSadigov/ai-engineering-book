# path: book/projects/examples/ch23/test_runnable.py
from __future__ import annotations

import pytest

from runnable import Lambda, Parallel, RunLog, Sequence


def test_pipe_builds_flat_sequence_and_invokes_in_order() -> None:
    chain = Lambda(str.strip, "strip") | str.lower | (lambda s: s.split())
    assert isinstance(chain, Sequence)
    assert len(chain.steps) == 3            # flattened, not nested
    assert chain.invoke("  Reset My Password ") == ["reset", "my", "password"]


def test_dict_coerces_to_parallel_fanout() -> None:
    chain = Lambda(str.strip) | {"upper": str.upper, "length": len}
    assert chain.invoke(" abc ") == {"upper": "ABC", "length": 3}
    assert isinstance(chain.steps[1], Parallel)


def test_run_log_records_every_step_with_io_and_latency() -> None:
    log = RunLog()
    chain = Lambda(str.strip, "strip") | Lambda(str.upper, "upper")
    chain.invoke(" hi ", log)
    assert [e["step"] for e in log.events] == ["strip", "upper"]
    assert log.events[0]["output"] == "hi" and log.events[1]["output"] == "HI"
    assert all(e["latency_ms"] >= 0 for e in log.events)


def test_with_retry_retries_only_listed_errors_and_logs_attempts() -> None:
    calls = {"n": 0}

    def flaky(x: str) -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("transient")
        return x.upper()

    step = Lambda(flaky, "flaky").with_retry(max_attempts=3, retry_on=(TimeoutError,))
    log = RunLog()
    assert step.invoke("ok", log) == "OK"
    assert calls["n"] == 3
    assert log.events[-1]["attempt"] == 3     # the retry wrapper made itself visible

    def wrong(x: str) -> str:
        raise ValueError("not retryable")

    with pytest.raises(ValueError):
        Lambda(wrong).with_retry(max_attempts=5, retry_on=(TimeoutError,)).invoke("x")


def test_batch_and_stream_defaults() -> None:
    chain = Lambda(str.upper)
    assert chain.batch(["a", "b"]) == ["A", "B"]
    assert list(chain.stream("a")) == ["A"]
