# Exercises — Chapter 38 — Durable and Long-Running Agents

Solutions: `../solutions/ch38-solutions.md`


### Knowledge questions

**K1.** Why must an idempotency key be derived from the event log rather than generated when the tool executes? What exactly goes wrong with a fresh key per attempt?

**K2.** Name the four crash points around a side-effecting tool call and the information the harness has after each one.

**K3.** What problem does a fence token solve that a lease alone does not?

**K4.** Why does this chapter treat an expired approval as a denial, and why does waiting time not count against an agent's run-time budget?

**K5.** Explain progressive disclosure for skills and why the description line matters more than the body for selection.

**K6.** Why must a voice agent's conversation history contain only what the caller heard, and what goes wrong if it contains the planned reply?

### Engineering questions

**E1.** Northwind's payment provider supports neither idempotency keys nor lookup by reference. Design the integration so that a refund tool can still be reconciled after a crash, and state which component owns the key.

**E2.** A coding agent needs `pip install` to complete some tasks. Design the permission boundary: what is allowed, what requires approval, where it runs, and what the Definition of Done must add.

**E3.** Your agents run on Kubernetes with rolling deploys every hour, and the slowest tool call takes up to four minutes. Choose a lease TTL and heartbeat strategy, and explain the recovery-time and duplicate-execution consequences of your choice.

**E4.** The support team wants the voice agent to begin creating a ticket as soon as the caller says "open a ticket" to save time. Propose a design that captures most of the latency benefit without acting on unstable transcripts.

### Practical exercises

**P1.** Add `DurableRunner.reopen(run_id, outcome)` for runs stopped on an unknown outcome: an operator records the true outcome in the ledger, and the run continues with that outcome as the observation. Write the tests.

**P2.** Extend `InterruptManager` with four-eyes approval for tools tagged `irreversible`: two distinct approvers from the chain, either of whom may reject. Include escalation and expiry behavior in the tests.

**P3.** Replace `digest_observation` for prose-heavy tools with an LLM summarizer through `aie_core`, keeping literal extraction, and add an identifier-recall evaluation comparing it with the extractive digest on a scripted 30-step run.

**P4.** Add a `run_linter` tool and a "no new lint errors" check to `coding_dod`, computed against the baseline, so pre-existing lint errors do not block the task but new ones do.

### Debugging exercises

**D1.** After a deploy, the ticketing team reports pairs of identical follow-up tickets created about forty seconds apart, always during rollouts. The logs show `Resumed(by="recovery")` on the affected runs and no `LeaseLost` errors. The tool is wrapped in `ReconcilingTool`. What is the most likely cause, and which telemetry confirms it?

**D2.** A long research run's final answer cites `INC-2231`, which does not exist. The event log shows the agent read `INC-2213` at step 6. Compaction statistics show the run was compacted from step 14 on with `keep_recent_steps=1`. Diagnose the failure and name two fixes.

**D3.** Approvals for `send_reply` are being granted overnight with nobody awake, and the audit log shows `decided_by = "clock"` on `resolved` interrupts. The `tick()` code was recently refactored. What broke, and what test would have caught it?
