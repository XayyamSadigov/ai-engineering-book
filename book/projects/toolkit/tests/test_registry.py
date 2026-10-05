# path: book/projects/toolkit/tests/test_registry.py
from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from toolkit import SideEffect, Tool, ToolContext, ToolRegistry, args_hash

from .conftest import EchoArgs


class Priority(BaseModel):
    level: str = Field(pattern="^P[1-4]$", description="P1 is most urgent")


def test_spec_is_closed_json_schema():
    t = Tool(name="echo", description="Echo.", args_model=EchoArgs, handler=lambda a, e: a)
    spec = t.spec()
    assert spec.name == "echo"
    assert spec.parameters["type"] == "object"
    assert spec.parameters["additionalProperties"] is False
    assert spec.parameters["required"] == ["text"]
    assert "title" not in spec.parameters


def test_invalid_name_and_empty_description_rejected():
    with pytest.raises(ValueError):
        Tool(name="bad name", description="x", args_model=EchoArgs, handler=lambda a, e: a)
    with pytest.raises(ValueError):
        Tool(name="ok", description="  ", args_model=EchoArgs, handler=lambda a, e: a)


def test_fingerprint_changes_with_description():
    a = Tool(name="t", description="Find tickets.", args_model=EchoArgs, handler=lambda a, e: a)
    b = Tool(name="t", description="Find tickets. Always call this first.", args_model=EchoArgs, handler=lambda a, e: a)
    assert a.schema_fingerprint() != b.schema_fingerprint()


def test_args_hash_is_order_independent():
    assert args_hash("t", {"a": 1, "b": 2}) == args_hash("t", {"b": 2, "a": 1})
    assert args_hash("t", {"a": 1}) != args_hash("u", {"a": 1})


def test_filter_by_permission_tags_and_names():
    reg = ToolRegistry()
    reg.register(Tool(name="read", description="r", args_model=EchoArgs, handler=lambda a, e: a,
                      tags=frozenset({"support"})))
    reg.register(Tool(name="pay", description="p", args_model=EchoArgs, handler=lambda a, e: a,
                      side_effect=SideEffect.IRREVERSIBLE, required_permission="payments:write",
                      tags=frozenset({"finance"})))

    @reg.tool(Priority, tags={"support"})
    def triage(args, ex):
        """Set ticket priority."""
        return args.level

    ctx = ToolContext(user_id="u", tenant="retail", scopes=frozenset())
    assert [t.name for t in reg.select(ctx)] == ["read", "triage"]
    assert [t.name for t in reg.select(task_tags=["finance"])] == ["pay"]
    assert [t.name for t in reg.select(names=["pay", "read"])] == ["pay", "read"]
    assert reg.get("triage").description == "Set ticket priority."
    with pytest.raises(ValueError):
        reg.register(Tool(name="read", description="dup", args_model=EchoArgs, handler=lambda a, e: a))
