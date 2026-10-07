# Exercises — Chapter 19 — The Agent Loop

Solutions: `../solutions/ch19-solutions.md`


**Start here:** K2, K3, E2, P1, D2 (about 3.5 hours). The rest go deeper.

### Knowledge questions

**K1.** Name the five parts of the practical definition of an agent used in this chapter, and for two of them explain what kind of system you get if you remove that part.

**K2.** Why does `agentkit` derive `AgentState` from events instead of updating a state object directly? Give three capabilities that depend on this choice.

**K3.** Explain the difference between the repeated-action detector and the no-progress detector. Describe a run that only one of them would catch, in each direction.

**K4.** Which error classes does the runtime retry, under what conditions, and why is a transient failure on a non-idempotent tool not retried?

**K5.** What is the difference between harness replay and counterfactual replay? What does a replay miss mean, and why should a miss not be read as a failure of the new model?

**K6.** Why is the deadline budget measured in active time rather than wall-clock time since the run started?

### Engineering questions

**E1.** Northwind wants an agent that answers employee questions about their PTO balance by calling `lookup_employee` and `get_pto_balance`, and answering. Argue whether this should be an agent, a workflow, or a single call with tools, using the decision criteria from Chapters 1 and 17 and this chapter's Definition of Done.

**E2.** Design budgets (all five dimensions) and loop configuration for the incident-research agent, given these illustrative measurements on 200 successful evaluation runs: median 5 steps, p95 7 steps; median 24,000 tokens, p95 41,000; median 18 s, p95 35 s; median 6 tool calls, p95 9. Justify each number and say what the product should do on each budget stop.

**E3.** A Definition of Done for "draft a reply to a support ticket" must reject drafts that promise refunds the policy does not allow. Propose a set of verifiers, say which are deterministic and which need a judge, and describe how a rejection's feedback should be worded so the model can act on it.

**E4.** Two processes might call `resume` on the same paused run at the same time (a double-clicked approval button). Walk through what happens with `JsonlEventStore` today, and propose a design that makes the second resume a safe no-op.

### Practical exercises

**P1.** (about 2 hours) Add a `PlanTool` to `agentkit`: a read-only tool `update_plan(steps: list[str], done: list[int])` that stores the plan as an artifact. Add a verifier `plan_completed()` that passes only when every step is marked done or the answer explains why a step was abandoned. Write tests showing that an unchanged plan does not count as progress and that the verifier rejects an answer with open steps.

**P2.** (about 3 hours) Implement parallel execution of read-only tool calls proposed in the same step. Keep the event order deterministic by request id, charge each call against the tool-call budget before launching it, and make sure a fatal error in one call still stops the run. Add a test with two slow fake tools that proves the step takes roughly the time of the slower one.

**P3.** (about 2 hours) Add a `SqliteEventStore` implementing the `EventStore` protocol, with a unique constraint on `(run_id, seq)` so that concurrent appends fail. Run the existing store tests against it, and add a test where two runtimes try to resume the same paused run and exactly one succeeds.

**P4.** (about 3 hours) Build a trajectory evaluator: given a list of recorded runs and a spec per task (required tools, forbidden tools, maximum steps, required termination reason), produce a table of pass and fail per task and an aggregate success rate, step efficiency, denial rate, and cost per successful task. Run it on ten runs of the Northwind example with varied scripted planners.

**P5.** (about 2 hours) Make `agentkit` carry opaque provider reasoning items through the loop. Add an optional `provider_items` field to `ModelDecision` (provider-specific JSON plus the provider and model that produced it), make the transcript that `apply` rebuilds carry the items to the provider adapter next to the tool call they belong to (the neutral `Message` has no slot for them, so choose where they travel and justify it), and make harness replay serve them back while counterfactual replay with a different model strips them. Test it with a fake client that returns an item with each tool call and fails the next request if the item is missing or altered.

### Debugging exercises

**D1.** A support agent's dashboard shows the share of runs ending in `NO_PROGRESS` rising from 2 percent to 19 percent the day after a release. Sampled traces show `search_tickets` called with three or four different queries per run, each returning `[]`, followed by the stop. The release changed the system prompt and upgraded the ticket-search service. Describe how you would determine which change caused the regression, using only the event logs and replay, and what you expect to find in each case.

**D2.** A Northwind agent created two identical follow-up tickets for one incident. The event log for the run contains `ToolCallApproved` for `create_ticket` at seq 41, then a `Resumed` event with `by="recovery"` at seq 42, then `ToolResult` for the same request at seq 46. The ticketing system shows two tickets created 90 seconds apart. Explain what happened, which component is at fault, and what change prevents it.

**D3.** After switching to a new model, the incident agent's success rate drops, and many runs end in `VERIFICATION_FAILED`. The rejection notes all say `citations_grounded>=1: citations not found in any tool result: ['inc-2026-0217']`. The search tool's results look correct in the trace. Find the root cause and propose two fixes, one in the harness and one in the prompt or tool.
