# path: book/projects/reliability/tests/test_bulkhead.py
import asyncio
import threading

import pytest

from reliability import AsyncBulkhead, Bulkhead, BulkheadFullError, Bulkheads


def test_full_pool_rejects_without_borrowing():
    pools = Bulkheads({"interactive": 2, "eval": 1})
    pools["eval"].acquire()
    with pytest.raises(BulkheadFullError) as info:
        pools["eval"].acquire()
    assert info.value.reason == "bulkhead_full:eval"
    pools["interactive"].acquire()                  # eval saturation does not affect interactive
    pools["interactive"].acquire()
    assert pools["interactive"].stats.utilization == 1.0


def test_waiting_room_admits_when_slot_frees():
    b = Bulkhead("batch", 1, max_waiting=1, max_wait_s=2.0)
    b.acquire()
    got = threading.Event()

    def waiter() -> None:
        with b.slot():
            got.set()

    t = threading.Thread(target=waiter)
    t.start()
    b.release()
    t.join(timeout=2)
    assert got.is_set()


def test_waiting_room_is_bounded():
    b = Bulkhead("batch", 1, max_waiting=0, max_wait_s=1.0)
    b.acquire()
    with pytest.raises(BulkheadFullError):
        b.acquire()
    assert b.stats.rejected == 1


async def test_async_bulkhead_caps_concurrency():
    b = AsyncBulkhead("interactive", 3, max_waiting=100, max_wait_s=5.0)
    peak = 0
    running = 0

    async def task() -> None:
        nonlocal peak, running
        async with b.slot():
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.01)
            running -= 1

    await asyncio.gather(*(task() for _ in range(12)))
    assert peak == 3 and b.stats.admitted == 12


async def test_async_bulkhead_wait_timeout():
    b = AsyncBulkhead("interactive", 1, max_waiting=1, max_wait_s=0.05)
    await b.acquire()
    with pytest.raises(BulkheadFullError):
        await b.acquire()
