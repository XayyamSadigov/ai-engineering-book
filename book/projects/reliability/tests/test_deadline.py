# path: book/projects/reliability/tests/test_deadline.py
import asyncio

import pytest
from aie_core import CompletionRequest, Message

from reliability import Deadline, DeadlineExceeded, current_deadline
from reliability.deadline import HEADER


def test_child_never_outlives_parent(clock):
    parent = Deadline.after(8.0, clock=clock)
    child = parent.child(20.0)
    assert child.remaining() == pytest.approx(8.0)
    clock.advance(5.0)
    assert child.remaining() == pytest.approx(3.0)


def test_child_budget_and_reserve(clock):
    parent = Deadline.after(8.0, clock=clock)
    retrieve = parent.child(1.5)
    assert retrieve.remaining() == pytest.approx(1.5)
    model = parent.child(reserve_s=0.5)                  # keep time for post-validation
    assert model.remaining() == pytest.approx(7.5)
    half = parent.child(fraction=0.5)
    assert half.remaining() == pytest.approx(4.0)


def test_cancel_propagates_to_children(clock):
    parent = Deadline.after(8.0, clock=clock)
    grandchild = parent.child(2.0).child(1.0)
    parent.cancel("client disconnected")
    assert grandchild.expired
    with pytest.raises(DeadlineExceeded, match="client disconnected"):
        grandchild.check("rerank")


def test_check_needs_minimum_useful_time(clock):
    d = Deadline.after(1.0, clock=clock)
    d.check("retrieve", need_s=0.5)
    clock.advance(0.7)
    with pytest.raises(DeadlineExceeded) as info:
        d.check("model", need_s=0.5)
    assert info.value.stage == "model"
    assert info.value.retryable is False


def test_apply_stamps_remaining_budget_on_request(clock):
    d = Deadline.after(8.0, clock=clock)
    clock.advance(3.0)
    req = d.apply(CompletionRequest(messages=[Message.user("hi")]))
    assert req.timeout_s == pytest.approx(5.0)
    capped = d.apply(CompletionRequest(messages=[Message.user("hi")], timeout_s=2.0))
    assert capped.timeout_s == pytest.approx(2.0)


def test_header_round_trip_uses_relative_budget(clock):
    d = Deadline.after(2.5, clock=clock)
    header = d.to_header()
    assert header[HEADER] == "2500"
    downstream = Deadline.from_header(header[HEADER], default_s=30.0, max_s=10.0, clock=clock)
    assert downstream.remaining() == pytest.approx(2.5)
    capped = Deadline.from_header("999999", default_s=30.0, max_s=10.0, clock=clock)
    assert capped.remaining() == pytest.approx(10.0)


def test_scope_sets_current_deadline(clock):
    d = Deadline.after(1.0, clock=clock)
    assert current_deadline() is None
    with d.scope():
        assert current_deadline() is d
    assert current_deadline() is None


async def test_run_cancels_awaitable_on_expiry():
    d = Deadline.after(0.05)
    cancelled = asyncio.Event()

    async def slow() -> None:
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    with pytest.raises(DeadlineExceeded):
        await d.run(slow(), stage="tool")
    assert cancelled.is_set()
