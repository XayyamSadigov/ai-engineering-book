# Chapter 18 examples: MCP and tool ecosystems

A minimal MCP-style server and client over stdio, written with the standard library and
pydantic, plus the host-side adapter that turns discovered tools into `aie_core.ToolSpec`
objects and authorizes every call independently of discovery. Everything runs offline:
the server reads `book/projects/shared-data`, the model is `aie_core`'s `FakeLLM`.

```
jsonrpc.py            JSON-RPC 2.0 messages, error codes, newline-delimited framing
schema_check.py       tiny JSON Schema subset validator (used by server and host)
northwind_server.py   server: initialize, ping, tools/*, resources/*, prompts/*; one tenant per process
mcp_client.py         StdioMcpClient: spawns the server, capability exchange, pagination, timeouts
host_adapter.py       Principal, ToolLock/ToolGrant (pinned fingerprints), McpToolAdapter, McpHost
demo.py               review -> pin -> connect to clean and poisoned servers, print decisions
test_ch18.py          16 tests; each server test spawns a real subprocess
```

## Run

```bash
# from the repository root, using the shared virtualenv (aie_core already installed)
.venv/bin/python -m pytest book/projects/examples/ch18 -q
cd book/projects/examples/ch18 && ../../../../.venv/bin/python demo.py

# standalone install
pip install -e ../../aie_core && pip install -e ".[dev]" && pytest -q

# talk to the server by hand (stateless mode skips the handshake)
python northwind_server.py --tenant retail --stateless
{"jsonrpc":"2.0","id":1,"method":"tools/list"}
```

## Configuration

| Setting | Where | Default | Meaning |
|---|---|---|---|
| `--tenant` | server CLI | required | `retail` or `logistics`; fixes the data scope of the process |
| `--stateless` | server CLI | off | answer requests without a prior `initialize` |
| `--poisoned` | server CLI | off | inject a malicious description and an `export_tickets` tool (tests only) |
| `--page-size` | server CLI | 50 | page size for `*/list` results (cursor pagination) |
| `LLM_PROVIDER`, `LLM_MODEL` | env | `fake` | only if you replace `FakeLLM` with `make_llm_client()` |

## What is deliberately missing

HTTP transport, authorization headers, server-initiated requests (sampling, elicitation),
subscriptions, progress notifications, and cancellation. Chapter 18 explains each; use an
official SDK when you need them, and keep `host_adapter.py`'s lockfile and authorization
in front of it either way.
