# Exercises — Chapter 16 — Tool Calling

Solutions: `../solutions/ch16-solutions.md`


### Knowledge questions

**K1.** Why is tool calling described as a protocol rather than a capability of the model, and what does that imply about where authorization lives?

**K2.** Name the four side-effect classes used in this chapter, give a Northwind example of each, and explain why `external` is distinguished from `irreversible`.

**K3.** What is an idempotency key, who should choose it, and why does a timeout on a non-idempotent tool produce `outcome_unknown` instead of a retry?

**K4.** List the five error categories and, for each, who is expected to recover and whether the executor retries.

**K5.** What does it mean for an approval to be "bound to concrete arguments", and why does Project 4's `send_reply` take the full body instead of a draft id?

**K6.** List three things `SandboxRunner` bounds and two things it does not isolate.

### Engineering questions

**E1.** A product manager asks for a single `manage_account(action, account_id, payload)` tool "to keep the menu small". Argue for an alternative design, covering selection quality, validation, policy, and approval.

**E2.** Your assistant runs on six replicas behind a load balancer. Which `toolkit` components need shared state, what would break if they used the in-memory implementations, and what would you back each with?

**E3.** Design the policy for an `issue_refund(order_id, amount, reason)` tool at Northwind Retail: side-effect class, permission, argument rules, escalation thresholds, rate limits, idempotency key scope, and what the approver sees.

**E4.** The model is supposed to call `get_service_status` before `create_ticket`, but in 30% of conversations it skips the status check. Describe how you would diagnose and fix this without changing the model.

### Practical exercises

**P1.** Add a `close_ticket(ticket_id, resolution)` tool to Project 4 as a reversible write that only the ticket's creator or a lead may use. Include the args model, handler with tenant scoping, policy rule, and tests for allowed, denied, and duplicate calls.

**P2.** Implement `RedisIdempotencyStore` satisfying the `IdempotencyStore` protocol using `SET key value NX EX ttl` for `begin`, and run the existing duplicate-suppression test against it with a fake Redis or a local server marked `integration`.

**P3.** Add a `fetch_url(url)` read tool with an egress allowlist of hosts, a response-size cap, and a timeout, using `httpx` with a mock transport in tests. Show that a URL carrying data to a non-allowlisted host is denied before any request is made.

**P4.** Build a selection evaluation set of 30 Northwind requests labeled with the expected first tool (or none). Write a runner that replays them through `ToolLoop` with `max_rounds=1` and reports accuracy and a confusion matrix. Run it with `FakeLLM` handlers to test the runner, and with a real model behind `@pytest.mark.integration`.

### Debugging exercises

**D1.** A customer received the same reply email twice, eleven minutes apart. The audit log shows two `tool.approval_requested` events for `send_reply` with identical `args_hash` but different `session_id`s, two approvals by the same lead, and two `tool.executed` events with different `idempotency_key` values. No `tool.duplicate_suppressed` event exists. The idempotency store is configured and shared. What happened, and what would you change?

**D2.** After a deploy, conversations about outages average seven rounds instead of three. Audit events show repeated `tool.invalid` for `get_service_status` with `details.errors[0].field = "service"`, and the model's arguments include values like `"VPN"` and `"vpn-service"`. The `tool_fingerprint` for `get_service_status` changed in the deploy. Diagnose and fix.

**D3.** The on-call engineer sees twenty `outcome_unknown` failures for `create_ticket` in ten minutes, all with latency of about 5.0 seconds, and users say their tickets "might not exist". The ticket system's dashboard shows p99 write latency rising from 300 ms to 6 s in the same window. What is going on, what should the on-call do now, and what should change permanently?
