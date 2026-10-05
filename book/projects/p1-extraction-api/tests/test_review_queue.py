# path: book/projects/p1-extraction-api/tests/test_review_queue.py
import threading
import uuid

import pytest

from extraction_api.adapters import InMemoryReviewQueue, SQLiteReviewQueue
from extraction_api.application import AlreadyResolved, ReviewItem, ReviewNotFound, ReviewResolution
from extraction_api.domain import DocumentType


def _item(**kw) -> ReviewItem:
    base = dict(review_id=uuid.uuid4().hex, request_id="r1", doc_type=DocumentType.INVOICE,
                reasons=["TOTAL_MISMATCH"], data={"total": "59000.00"}, document_text="...",
                prompt_version="test")
    base.update(kw)
    return ReviewItem(**base)


@pytest.fixture(params=["memory", "sqlite"])
def q(request, tmp_path):
    if request.param == "memory":
        return InMemoryReviewQueue()
    return SQLiteReviewQueue(tmp_path / "review.db")


def test_enqueue_list_get(q):
    a, b = q.enqueue(_item()), q.enqueue(_item(reasons=["MISSING_PO"]))
    assert [i.review_id for i in q.list()] == [a.review_id, b.review_id]
    assert q.get(b.review_id).reasons == ["MISSING_PO"]
    assert q.counts() == {"pending": 2}
    with pytest.raises(ReviewNotFound):
        q.get("nope")


def test_resolve_once_only(q):
    item = q.enqueue(_item())
    done = q.resolve(item.review_id, ReviewResolution(decision="reject", reviewer="ap-clerk-7", note="duplicate"))
    assert done.status == "rejected" and done.resolution.reviewer == "ap-clerk-7"
    assert q.list() == [] and len(q.list("rejected")) == 1 and len(q.list(None)) == 1
    with pytest.raises(AlreadyResolved):
        q.resolve(item.review_id, ReviewResolution(decision="approve", reviewer="someone-else"))


def test_concurrent_resolution_has_exactly_one_winner(q):
    item = q.enqueue(_item())
    wins, losses = [], []

    def attempt(n: int) -> None:
        try:
            q.resolve(item.review_id, ReviewResolution(decision="approve", reviewer=f"r{n}"))
            wins.append(n)
        except AlreadyResolved:
            losses.append(n)

    threads = [threading.Thread(target=attempt, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(wins) == 1 and len(losses) == 7


def test_sqlite_survives_restart(tmp_path):
    path = tmp_path / "review.db"
    first = SQLiteReviewQueue(path)
    item = first.enqueue(_item(document_id="INV-007"))
    first.close()
    second = SQLiteReviewQueue(path)
    assert second.get(item.review_id).document_id == "INV-007"
