# toolkit

Governed tool calling for the AI Engineering book. Built in Chapter 16; imported by Chapters 19,
20, 22 and the capstone. The model proposes tool calls; `toolkit` decides, executes, and records.

```
toolkit/
  registry.py     SideEffect, Tool, ToolRegistry, args_hash, canonical_json
  policy.py       ToolContext, PolicyEngine (alias ToolPolicy), Decision, Verdict, recipient_allowlist, max_value
  approval.py     ApprovalManager, ApprovalRequest, ApprovalStatus, ApprovalError
  errors.py       ToolError, ErrorCategory (validation | permission | not_found | transient | fatal)
  idempotency.py  IdempotencyStore protocol, InMemoryIdempotencyStore, SQLiteIdempotencyStore
  audit.py        AuditEvent, InMemoryAuditLog, JsonlAuditLog, NullAuditLog
  executor.py     ToolExecutor, ToolResult, ExecutionContext, BoundTool, truncate_payload
  sandbox.py      SandboxRunner, SandboxLimits, SandboxResult, make_python_tool
  loop.py         ToolLoop, LoopResult
tests/            offline (FakeLLM from aie_core; real subprocesses for the sandbox)
```

## Install and test

```bash
uv pip install --python ../../../.venv/bin/python -e ../aie_core -e ".[dev]"   # or pip install -e
python -m pytest -q
```

No configuration of its own: everything is constructor arguments, so the application decides
where approvals, idempotency records, and audit events live.

## Usage

```python
from pydantic import BaseModel, Field
from aie_core import Message, make_llm_client
from toolkit import (Tool, ToolRegistry, SideEffect, ToolContext, PolicyEngine, ToolExecutor,
                     ApprovalManager, SQLiteIdempotencyStore, InMemoryAuditLog, ToolLoop,
                     recipient_allowlist)

class SendArgs(BaseModel):
    to: str = Field(pattern=r"^[^@\s]+@[^@\s]+$")
    body: str = Field(max_length=4000)

registry = ToolRegistry([Tool(name="send_reply", description="Send an email reply.", args_model=SendArgs,
                              handler=lambda a, ex: {"sent": True}, side_effect=SideEffect.EXTERNAL,
                              required_permission="replies:send", idempotent=False, requires_approval=True)])
policy = PolicyEngine()
policy.add_rule("send_reply", recipient_allowlist("to", ["northwind.example"]), rule_id="recipient_allowlist")
executor = ToolExecutor(registry, policy, approvals=ApprovalManager(),
                        idempotency=SQLiteIdempotencyStore("idem.db"), audit=InMemoryAuditLog())
ctx = ToolContext(user_id="ana", tenant="retail", scopes=frozenset({"replies:send"}), session_id="s1")
result = ToolLoop(make_llm_client(), executor).run([Message.user("Reply to Priya")], ctx)
```

Agent runtimes call `executor.bind(ctx)` to get `BoundTool` objects with `name`, `spec`,
`side_effect`, `requires_approval`, `idempotent`, and `execute(arguments, ctx)`.
