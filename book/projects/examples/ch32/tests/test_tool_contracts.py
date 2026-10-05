# path: book/projects/examples/ch32/tests/test_tool_contracts.py
"""Contract tests for tools: what the model is told must match what the handler accepts,
and a schema change must be a deliberate, versioned event."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from northwind_triage.adapters.tools import (
    CONTRACTS,
    CREATE_TICKET,
    CreateTicketArgs,
    InMemoryTicketRepo,
    make_create_ticket_handler,
)
from northwind_triage.domain import Category, Priority

LOCK = Path(__file__).parent / "tool_schemas.lock"


@pytest.mark.parametrize("contract", CONTRACTS, ids=lambda c: c.name)
def test_advertised_schema_is_generated_from_validator(contract) -> None:
    assert contract.spec().parameters == contract.args_model.model_json_schema()


@pytest.mark.parametrize("contract", CONTRACTS, ids=lambda c: c.name)
def test_schema_is_model_friendly(contract) -> None:
    params = contract.spec().parameters
    assert params["type"] == "object"
    assert params.get("additionalProperties") is False, "reject invented arguments"
    assert set(params["required"]) >= {"idempotency_key"} if contract.side_effect == "write" else True
    assert 20 <= len(contract.description) <= 400, "descriptions are prompts; keep them specific"


@pytest.mark.parametrize("contract", CONTRACTS, ids=lambda c: c.name)
def test_documented_examples_hold(contract) -> None:
    for example in contract.valid_examples:
        contract.args_model.model_validate(example)
    for example in contract.invalid_examples:
        with pytest.raises(ValidationError):
            contract.args_model.model_validate(example)


@pytest.mark.parametrize("contract", CONTRACTS, ids=lambda c: c.name)
def test_schema_change_requires_version_bump(contract) -> None:
    """tool_schemas.lock pins name@version -> schema hash. Editing the schema without bumping
    the version fails here; bumping the version without updating the lock fails too."""
    locked = json.loads(LOCK.read_text())
    key = f"{contract.name}@{contract.version}"
    assert key in locked, f"{key} not in tool_schemas.lock: new version? add it deliberately"
    assert locked[key] == contract.schema_hash(), (
        f"{contract.name} schema changed but version is still {contract.version}")


def test_write_tools_are_idempotent() -> None:
    repo = InMemoryTicketRepo()
    handler = make_create_ticket_handler(repo)
    args = dict(CREATE_TICKET.valid_examples[0])
    first, second = handler(args), handler(args)
    assert first["ticket_id"] == second["ticket_id"]
    assert (first["created"], second["created"]) == (True, False)
    assert len(repo.tickets) == 1


def test_idempotency_keys_are_tenant_scoped() -> None:
    repo = InMemoryTicketRepo()
    handler = make_create_ticket_handler(repo)
    a = handler({**CREATE_TICKET.valid_examples[0], "tenant": "retail"})
    b = handler({**CREATE_TICKET.valid_examples[0], "tenant": "logistics"})
    assert a["ticket_id"] != b["ticket_id"]


valid_args = st.fixed_dictionaries({
    "tenant": st.sampled_from(["retail", "logistics", "shared"]),
    "subject": st.text(min_size=3, max_size=120).filter(lambda s: len(s) >= 3),
    "body": st.text(min_size=1, max_size=400),
    "category": st.sampled_from([c.value for c in Category]),
    "priority": st.sampled_from([p.value for p in Priority]),
    "idempotency_key": st.from_regex(r"^[A-Za-z0-9_-]{8,64}$", fullmatch=True),
})


@settings(max_examples=60)
@given(valid_args)
def test_handler_output_always_matches_result_schema(args) -> None:
    out = make_create_ticket_handler(InMemoryTicketRepo())(args)
    CREATE_TICKET.result_model.model_validate(out)
    CreateTicketArgs.model_validate(args)
