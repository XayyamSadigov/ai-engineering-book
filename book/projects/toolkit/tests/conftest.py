# path: book/projects/toolkit/tests/conftest.py
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toolkit import (ApprovalManager, InMemoryAuditLog, InMemoryIdempotencyStore, PolicyEngine,  # noqa: E402
                     SideEffect, Tool, ToolContext, ToolError, ToolExecutor, ToolRegistry)


class EchoArgs(BaseModel):
    text: str = Field(min_length=1, max_length=50)


class SendArgs(BaseModel):
    to: str
    body: str = Field(max_length=500)


class Counter:
    def __init__(self) -> None:
        self.calls = 0
        self.fail_first = 0
        self.fail_with: ToolError | None = None

    def __call__(self, args, ex):
        self.calls += 1
        if self.calls <= self.fail_first and self.fail_with is not None:
            raise self.fail_with
        return {"echo": getattr(args, "text", None) or getattr(args, "to", None), "n": self.calls}


@pytest.fixture
def ctx() -> ToolContext:
    return ToolContext(user_id="u1", tenant="retail", groups=frozenset({"all", "support"}),
                       scopes=frozenset({"tickets:read", "mail:send"}), session_id="s1")


@pytest.fixture
def harness():
    echo = Counter()
    send = Counter()
    reg = ToolRegistry([
        Tool(name="echo", description="Echo text back.", args_model=EchoArgs, handler=echo,
             required_permission="tickets:read", tags=frozenset({"support"})),
        Tool(name="send", description="Send a message to an address.", args_model=SendArgs, handler=send,
             side_effect=SideEffect.EXTERNAL, required_permission="mail:send", idempotent=False,
             tags=frozenset({"support"})),
    ])
    audit = InMemoryAuditLog()
    approvals = ApprovalManager(ttl_s=60)
    policy = PolicyEngine()
    ex = ToolExecutor(reg, policy, approvals=approvals, idempotency=InMemoryIdempotencyStore(), audit=audit,
                      sleep=lambda s: None)
    yield {"reg": reg, "ex": ex, "audit": audit, "approvals": approvals, "policy": policy,
           "echo": echo, "send": send}
    ex.close()
