# Exercises — Chapter 18 — MCP and Tool Ecosystems

Solutions: `../solutions/ch18-solutions.md`


**Start here:** K4, K7, E3, P2, D3 (about 3 hours). The rest go deeper.

### Knowledge questions

**K1.** Name the three roles in an MCP deployment, state which of them can be third-party code, and explain why that matters for where authorization lives.

**K2.** Tools, resources, and prompts differ by who controls their use. State the controller of each and give one Northwind example of each.

**K3.** A `tools/call` for `get_ticket` with a well-formed but nonexistent ticket id returns a result rather than a JSON-RPC error. Explain the two error channels and how a host should treat each.

**K4.** Explain why discovery is not authorization, and why a host must re-authorize at call time even for tools it filtered at offer time.

**K5.** What does a move to stateless protocol requests change for deploying an HTTP server, and what does it not change about application state?

**K6.** Distinguish an Agent Skill from an MCP server, and describe one way they are used together.

**K7.** A host connects to a remote MCP server for the first time and receives HTTP 401. Describe, step by step, how it obtains a token for this user, and name the property of the token, and the check on the server, that stop the server from replaying it against another service.

### Engineering questions

**E1.** Northwind will connect forty internal servers and three vendor servers to four hosts across two tenants. Design the topology: where the gateway sits, how identity reaches each server, how tenancy is enforced, and which checks remain in each host.

**E2.** A team wants to wrap its in-process `lookup_employee` function, used only by Northwind Assist, as an MCP server "for consistency". Argue for or against, and name the signal that would change your answer.

**E3.** A remote server must read a user's mailbox to draft replies. Design the credential flow so that the server cannot act as a confused deputy and a token stolen from it cannot be replayed against other services.

**E4.** The ticket team wants to rename `search_tickets` to `find_tickets` and add an optional `since` parameter. Write the rollout plan covering protocol, server version, host lockfile, and evaluation.

### Practical exercises

**P1.** (about 2 hours) Add an HTTP transport: a FastAPI app with one `POST /mcp` endpoint that passes the body to `McpStyleServer.handle` in stateless mode and returns the response. Add a `HttpMcpClient` with the same interface as `StdioMcpClient` using `httpx`, and run the existing protocol tests against both transports.

**P2.** (about 90 min) Make the server declare `tools.listChanged: true` and emit `notifications/tools/list_changed` when a `--mutate-after N` flag causes it to change a description after N calls. Make the host re-run `discover` on that notification and add a test that the changed tool is quarantined mid-session.

**P3.** (about 2 hours) Run two servers that both expose a tool named `search`. Show that namespacing keeps them distinct, then write a `GatewayAdapter` that aggregates both servers behind one client interface with one combined, namespaced catalog and a single audit log.

**P4.** (about 3 hours) Add a `create_ticket` write tool to the server and route it through Chapter 16's `toolkit` in the host (`PolicyEngine`, `ApprovalManager`, and an `IdempotencyStore`, all driven by `ToolExecutor`): the model's call produces a pending approval bound to the exact arguments, and a retry with the same idempotency key creates one ticket, not two.

### Debugging exercises

**D1.** After a deploy of the ticket server, the share of answers that cite a ticket drops from about 60 percent to under 5 percent. No errors are logged; the model simply stops calling `search_tickets`. The server's changelog says only "improved tool descriptions". Diagnose and name the telemetry that confirms it.

**D2.** A newly connected HTTP ticket server on a 2025 revision, deployed with three replicas, fails about two thirds of `tools/call` requests with "server not initialized", while `initialize` itself always succeeds. A single-replica staging deployment never fails. Diagnose, and say whether the same deployment could fail this way on the 2026-07-28 revision.

**D3.** Two weeks after a calendar vendor's server was enabled for all users, an audit finds that some replies sent through the internal `send_reply` tool were copied to an external address. The internal server and its descriptions did not change, and every `send_reply` call was approved by a human who saw only the reply body. Reconstruct what happened and name the two control failures.
