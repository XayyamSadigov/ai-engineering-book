# path: book/projects/toolkit/tests/test_approval.py
from __future__ import annotations

import pytest

from toolkit import ApprovalError, ApprovalManager, ApprovalStatus, args_hash


def test_request_is_deduplicated_and_bound_to_hash(ctx):
    m = ApprovalManager()
    a = m.request("send", {"to": "a@x", "body": "hi"}, ctx)
    b = m.request("send", {"body": "hi", "to": "a@x"}, ctx)
    assert a.id == b.id and a.args_hash == args_hash("send", {"to": "a@x", "body": "hi"})


def test_verify_requires_approved_and_exact_args(ctx):
    m = ApprovalManager()
    req = m.request("send", {"to": "a@x", "body": "hi"}, ctx)
    with pytest.raises(ApprovalError) as e:
        m.verify(req.id, "send", req.args_hash, ctx)
    assert e.value.code == "approval_pending"
    m.approve(req.id, "lead1")
    assert m.verify(req.id, "send", req.args_hash, ctx).status == ApprovalStatus.APPROVED
    with pytest.raises(ApprovalError) as e:
        m.verify(req.id, "send", args_hash("send", {"to": "evil@x", "body": "hi"}), ctx)
    assert e.value.code == "approval_mismatch"
    other = ctx.model_copy(update={"user_id": "someone-else"})
    with pytest.raises(ApprovalError):
        m.verify(req.id, "send", req.args_hash, other)


def test_expiry_and_single_use(ctx):
    now = [1000.0]
    m = ApprovalManager(ttl_s=10, clock=lambda: now[0])
    req = m.request("send", {"to": "a@x"}, ctx)
    m.approve(req.id, "lead")
    m.consume(req.id)
    with pytest.raises(ApprovalError) as e:
        m.verify(req.id, "send", req.args_hash, ctx)
    assert e.value.code == "approval_consumed"
    late = m.request("send", {"to": "b@x"}, ctx)
    now[0] += 11
    assert m.get(late.id).status == ApprovalStatus.EXPIRED
    with pytest.raises(ApprovalError):
        m.approve(late.id, "lead")


def test_four_eyes(ctx):
    m = ApprovalManager(allow_self_approval=False)
    req = m.request("send", {"to": "a@x"}, ctx)
    with pytest.raises(ApprovalError) as e:
        m.approve(req.id, ctx.user_id)
    assert e.value.code == "self_approval"
    assert m.reject(req.id, "lead", note="wrong customer").status == ApprovalStatus.REJECTED
    assert m.pending() == []
