# Exercises — Chapter 17 — AI Workflows and Orchestration

Solutions: `../solutions/ch17-solutions.md`


**Start here:** K1, K3, E2, P2, D2 (about 4 hours). The rest go deeper.

### Knowledge questions

**K1.** Define the five positions on the spectrum in one sentence each, using the question "who decides the next step" (and, for the two positions where code decides, whether there is a step sequence at all). Give one Northwind example per position that is not in this chapter.

**K2.** A chain has five model steps with per-step success probabilities 0.99, 0.98, 0.97, 0.99, and 0.95. What is the chain's success probability under independence? Name two reasons the real figure differs, one in each direction.

**K3.** Explain why the paused checkpoint is written before the approval node while every other checkpoint is written after its node. What goes wrong if you write all checkpoints before their nodes?

**K4.** Which of the five error classes does the engine represent as exceptions, and which as state? Why does the semantic class have to be state?

**K5.** A colleague proposes a router that returns the node name produced by the model in a `next_step` JSON field. Classify the resulting system on the spectrum and name the safeguard that is lost.

**K6.** Order the five positions from simplest to most complex and state the rule for moving from one to the next. Why is "the model is capable enough to figure out the steps" not evidence for moving right?

### Engineering questions

**E1.** Design the state record for a Northwind invoice-processing workflow (extract, validate totals, match to purchase order, approve over a threshold, post to the ledger). List fields, types, and which fields are append-only. State which node is irreversible and what idempotency key it uses.

**E2.** The triage graph is to run on a queue with at-least-once delivery. Describe where duplicate deliveries can start a second run of the same ticket and how you prevent it, both at submission and inside the `send` node.

**E3.** Propose a schema for a durable `Checkpointer` on PostgreSQL. Include the graph version, and describe the behavior when a run paused under version 3 is resumed by version 4 in which `approval` was renamed `review`.

**E4.** The team wants to parallelize `retrieve_policy` and a new `lookup_customer_history` node. Rewrite the relevant part of the graph using fan-out and fan-in and specify what the fan-in does when one branch raises `TransientError`.

**E5.** The triage workflow is to move onto a code-first durable workflow engine because refund approvals now wait up to three days. Say which of the seven nodes become activities and which logic stays in the workflow function, how `approval` and its `ResumeHandle` check are expressed, what the engine's history will contain and what that implies for retention, and what `send` still needs that the engine does not provide.

### Practical exercises

**P1.** (about 90 min) Add a `fallback_model` to `build_triage_graph`: when `draft` exhausts its retries with `TransientError`, run the draft once more against a second model callable before failing. Add a test with a primary that always raises and a secondary that succeeds, and assert the trace shows the attempts.

**P2.** (about 90 min) Implement `JsonlCheckpointer` that appends checkpoints to a file and reconstructs `latest` and `history` from it. Replace `InMemoryCheckpointer` in the crash test so the two "processes" share only the file.

**P3.** (about 2 hours) Add a `path` attribute to `RunResult` (the list of node names visited) and extend `compare.py` to print the path distribution over a set of twenty mixed tickets (shipping, refund, account, and one with a scripted `escalate` verdict). Write a golden-path test that asserts the expected path for each ticket.

**P4.** (about 3 hours) Build a miniature "deterministic workflow versus agent graph" comparison: implement a bounded three-step agent node (Chapter 19 style, with the fake model choosing between "retrieve more" and "draft") and run both designs over the same twenty tickets, reporting success rate, average model calls, and total simulated latency.

### Debugging exercises

**D1.** A trace shows a run with `draft` attempts equal to 3, `validate` attempts equal to 3, and then `failed` with `TransientError`. The provider status page shows a two-minute incident. The on-call engineer proposes raising `max_attempts` to 6. Diagnose the actual problem and name the change you would make instead, with the telemetry that would confirm it.

**D2.** Customers of the `logistics` tenant report that two identical replies arrived four minutes apart for the same ticket. The checkpoint log for that run contains two checkpoints for `send`, the first with status `failed` and no error text and the second with status `completed`. Reconstruct what happened and name the two defects.

**D3.** After a deploy, the fraction of runs ending in `escalate` rises from 4 percent to 19 percent while every per-node success metric is unchanged and the eval suite is green. Which change in the deploy is the likely cause, which metric found it, and what was missing from the eval suite?
