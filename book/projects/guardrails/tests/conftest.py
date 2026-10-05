# path: book/projects/guardrails/tests/conftest.py
from __future__ import annotations

import pytest

from guardrails import GuardContext, PIIVault


@pytest.fixture
def ctx() -> GuardContext:
    return GuardContext(tenant="retail", user_id="u-retail-1", groups=frozenset({"all", "support"}),
                        request_id="req-1")


@pytest.fixture
def vault_ctx(ctx: GuardContext) -> GuardContext:
    ctx.vault = PIIVault(tenant="retail", scope_id="req-1", key=b"k" * 32)
    return ctx
