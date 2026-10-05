# Chapter 19 — Solutions: The Agent Loop

Identifiers match the exercises in `book/chapters/19-the-agent-loop.md`. Code references point at `book/projects/agentkit/`.

## Knowledge questions

**K1.** Model, instructions, tools, state, and loop. Remove the loop and you have a single tool-calling request: the model can ask for a tool once, but it cannot react to what the tool returned, so it cannot gather information adaptively. Remove the tools and you have a chatbot: it may converse over many turns, but it takes no actions and observes nothing beyond what the user typed. Remove explicit state and the transcript becomes the only state. That fails on crashes and approval pauses, which need a durable record of what happened, and on long runs, where the transcript becomes too long and too ambiguous to answer "did we already send that?". Remove completion instructions and verification and the model alone decides when it is done, which is the decision it is least reliable at.

**K2.** State derived from events cannot disagree with the record of what happened. Every state change is an appended event processed by one pure function, `apply`. Capabilities that depend on this:

1. **Audit.** The log shows what the agent saw, decided, and did, with the policy verdicts and their reasons.
2. **Resume.** After a crash or an approval pause, a new process derives state from the log and continues. It never asks the model what it was doing.
3. **Replay.** Recorded `ToolResult`s are served by key to a rerun.
4. **Trajectory tests.** Tests assert on the event sequence, and they can check that `derive_state(events)` equals the live state.
5. **Transcript changes.** Compaction or a new message rendering is safe because the facts underneath are unchanged.

**K3.** The repeated-action detector counts executions per action key (tool name plus canonical arguments) and stops on an identical request beyond `max_identical_calls`. The no-progress detector counts consecutive steps that produced no new information, meaning no successful observation with unseen content and no artifact change.

- **Only repeated-action catches it.** The agent alternates between calling `get_service_status("trackline")` and a search that returns new hits each time. Every step makes progress because the searches return new content, but the status call repeats with identical arguments and trips the repeat detector.
- **Only no-progress catches it.** The agent searches "vpn error", "vpn issue", and "vpn problem", and each returns "no results". The keys all differ, so the repeat detector never fires, but no step after the first adds information.

**K4.** Only transient errors are retried inside the loop. The tool must be idempotent, and the retry count is capped by `LoopConfig.transient_retries`. Validation errors go back to the model for repair, which is not a retry with the same input. Permission errors are never retried. Fatal errors stop the run. Impossible errors are surfaced to the model. Model-call failures are retried by the gateway, not by the loop.

A non-idempotent tool is not retried on a transient failure because the failure is ambiguous. A timeout after submitting a payment or sending an email may mean the action happened and only the response was lost, so a blind retry can duplicate it. The tool layer must decide whether the action took effect, using its idempotency store or reconcile hook.

**K5.** Harness replay reuses the recorded model decisions (`RecordedLLM`) and the recorded tool results, and changes only the harness: the verifier, the policy, the truncation limit, or the runtime code. It regression-tests the shell.

Counterfactual replay uses a new model or prompt with the recorded tool results, holding the environment fixed. It shows where planning behavior changes.

A miss is a call the new planner made that the recording never saw, so there is no observation to serve. The replay tool returns a synthetic impossible-class failure instead. That means the replay cannot say what would have happened next. It does not mean the new model was wrong: the new call may be better, but it needs a live evaluation to judge.

**K6.** Approval pauses can last minutes or days, and that time is not the agent working. If the deadline counted wall-clock time since the start, every run that waited for a human would resume already past its deadline and stop immediately. The deadline exists to bound user-visible latency and runaway work, which is active time. The runtime records elapsed time in `BudgetUpdated` and restarts the clock from that value on resume.

## Engineering questions

**E1.** This is a single call with tools, or at most a two-step deterministic workflow. It should not be an agent.

- The action sequence is known: identify the employee, fetch the balance, answer. There is nothing for an agent to discover.
- The calls are read-only, but they touch personal data, so authorization must come from the authenticated principal. The employee id should come from the session, not from the model.
- Latency matters for an interactive question, and every extra model call adds to it.
- Done is trivially verifiable: the answer contains the balance returned by the tool.

The best design resolves the employee from the session, calls `get_pto_balance` directly in code, and uses one model call only to phrase the answer, or a template. If users ask follow-ups like "can I take next Friday off?", add a policy lookup step. That is still a workflow. An agent would add variable cost and failure modes such as loops and premature answers with no compensating benefit.

**E2.** One reasonable configuration, with illustrative reasoning:

| Setting | Value | Reasoning |
|---|---|---|
| `max_steps` | 10 | About 1.4 times the p95 of 7 steps. Room for one verifier rejection plus a retry. |
| `max_tokens` | 70,000 | About 1.7 times the p95 of 41,000. Leaves headroom for one long observation without permitting runaway growth. |
| `deadline_s` | 60 | About 1.7 times the p95 of 35 s. User-facing, so a stop beats waiting. |
| `max_tool_calls` | 14 | About 1.5 times the p95 of 9. Protects downstream APIs. |
| `max_cost_usd` | 2 times the p95 cost | Derived from the pricing table, so it tracks model changes. |
| `max_no_progress_steps` | 3 | No new information for three steps means the agent is stuck. |
| `max_identical_calls` | 2 | One deliberate re-check is fine. A third identical call is a loop. |
| `max_dod_rejections` | 2 | Allows recovery from a premature answer without letting the agent argue indefinitely. |
| `max_observation_chars` | 4,000 | Assumes the tools already shape their results. |

Product behavior on each stop:

- **Step, token, cost, or tool-call stop.** Return a partial report from the observations so far, offer "continue investigating", and implement that offer as `resume(run_id, budget=larger)` gated to an operator.
- **Deadline stop.** Same partial report, plus an option to continue asynchronously and notify the user.
- **No progress.** Show the queries tried and suggest a human escalation.
- **Repeated action.** Treat as a defect. Log it for prompt and tool-description review and escalate to a human.

Review the thresholds monthly against the success distribution, and track budget stops on tasks that humans later mark as solvable.

**E3.** Verifiers for a draft reply:

1. **Non-empty and within a length bound.** Deterministic.
2. **The policy document was retrieved.** `tool_was_called("search_docs")`, plus `citations_grounded(1)` with the returns-policy id. Deterministic.
3. **No refund promise unless the policy tool allowed it.** Deterministic, written as a `Check`: if the answer matches refund-promise patterns ("we will refund", "refund has been issued", an amount), then the state must contain an artifact `refund_eligibility` with `eligible: true` and an amount at least as large as any amount mentioned.
4. **Tone and policy consistency.** Needs a judge. A separate model with a rubric (Chapter 24) checks that the reply does not imply exceptions the policy does not grant.

Run the deterministic checks first. Run the judge only when they pass, to save cost.

Feedback must name the failed criterion and the fix, and it must not reveal how the check works in a way that teaches evasion. For example: "Unmet: refund_promise_requires_eligibility. The draft promises a refund, but no eligibility check was performed. Call check_refund_eligibility for this order, or remove the promise." Vague feedback such as "the answer is not good enough" leads to cosmetic rewrites that fail again.

**E4.** Today, both processes load the log and derive the same state: awaiting approval for request `1.0`. Each `JsonlEventStore` instance computes the next sequence number from the file length on its first append, and each caches it. Both append `Resumed` with the same seq. The JSONL file then contains two events with that seq, interleaved histories, and possibly two executions of the approved call. Without idempotency in the tool, that means two tickets or two emails. The in-process check cannot see the other process.

A safe design:

1. **Database store with a unique constraint on `(run_id, seq)`.** The second process's first insert fails with a constraint violation, it reloads, sees the run already resumed, and returns the current result. Optimistic concurrency.
2. **Conditional resume.** The approval endpoint passes the seq of the `Stopped(APPROVAL_REQUIRED)` event it displayed. `resume` refuses if the log has advanced past it, which also prevents applying a stale approval.
3. **Keep idempotency keys `run_id:request_id`.** Even if a duplicate slipped through, the tool would deduplicate.

With a file store, use an exclusive lock on the run file around load, decide, and append, plus a re-check after acquiring the lock.

## Practical exercises

**P1.** Expected implementation:

- A `FunctionTool` named `update_plan` with schema `{steps: array of string (minItems 1), done: array of integer}`, `side_effect=READ`, returning `ToolOutput(content="plan updated: k of n done", artifacts={"plan": {"steps": steps, "done": sorted(set(done))}})`.
- `plan_completed()` as a `Check` that reads `state.artifacts.get("plan")`. It passes when no plan exists, which is optional, or when every index is in `done`. Otherwise it passes only if the answer contains an "Abandoned:" section naming each open step.

Acceptance criteria:

1. A scripted run that calls `update_plan` twice with identical arguments has `StepCompleted.progress == False` on the second step. The artifact is unchanged and the content hash was already seen, so it counts toward `NO_PROGRESS`.
2. A final answer with an open step and no abandonment note produces a `dod_rejected` note naming `plan_completed`.
3. Marking all steps done and answering completes the run.
4. Existing tests still pass.

**P2.** Expected implementation:

- In `_process_calls`, first authorize every pending call sequentially, which keeps policy, approval, and budget decisions deterministic.
- Then partition the approved calls into read-only and others.
- Execute the read-only group with a `ThreadPoolExecutor`, collecting `(request_id, ToolOutput, attempts, latency)`.
- Emit the `ToolResult` events in request-id order after all of them finish, and execute non-read calls sequentially afterward.
- Charge the tool-call budget at authorization time, so a budget of N never launches N+1 calls. Track a pending count in the session, or record a "reserved" note.
- If any result has error class `FATAL`, emit all collected results and then stop.

Acceptance criteria:

1. A test with two tools that sleep 0.2 s each completes the step in well under 0.4 s. Assert on a monotonic clock with tolerance.
2. Event order is `ToolResult(1.0)` then `ToolResult(1.1)` regardless of completion order.
3. `max_tool_calls=1` with two proposed reads executes exactly one.
4. Replay of a parallel run reproduces the same decisions.

**P3.** Expected implementation:

- `SqliteEventStore(path)` creates a table `events(run_id TEXT, seq INTEGER, type TEXT, body TEXT, PRIMARY KEY (run_id, seq))`.
- `append` does `INSERT` inside a transaction and raises `ValueError` on `IntegrityError`.
- `load` selects ordered by seq and parses with `event_from_json`. `runs` selects distinct run ids.
- Parametrize the existing JSONL round-trip test over both stores.

Concurrency test:

1. Create a paused run and two `AgentRuntime` instances sharing the database file.
2. Call `resume(approve=True)` from two threads behind a barrier.
3. Assert that exactly one returns a completed `RunResult` and the other raises, or returns after reloading. Pick one behavior and document it.
4. Assert that the send tool executed once and that seqs are dense with no duplicates.

**P4.** Expected implementation:

- A `TaskSpec` pydantic model: `task_id`, `required_tools: set[str]`, `forbidden_tools: set[str]`, `ordering: list[tuple[str, str]]` (A before B), `max_steps`, `expected_reason`.
- An evaluator over `(spec, events)` that computes:
  - pass/fail per criterion, from `RunResult.trajectory()` and the `ToolCallRequested` and `Stopped` events;
  - steps, from `ModelDecision` count;
  - denial rate, as denials over requested calls;
  - cost, from `BudgetUpdated.usage.cost_usd` on the last event.
- Aggregate success rate, mean and p95 steps, cost per successful task (total cost over successes), and denial rate by class.
- Ten Northwind runs, made by varying the scripted planner: skip the metrics step, call `create_ticket` first, answer without searching, and so on.

Acceptance criteria:

1. The runs that answer without searching fail `required_tools`, and the evaluator reports them as `dod_rejected` or `VERIFICATION_FAILED`.
2. A planner that calls `create_ticket` before `search_docs` fails the ordering constraint.
3. The aggregate numbers match a hand calculation for the ten runs.

## Debugging exercises

**D1.** Separate the two changes with replay, holding everything else fixed.

1. Take a sample of pre-release trajectories that completed.
2. Run counterfactual replay with the new system prompt and the old recorded tool results. If the new prompt causes the regression, the replays diverge early: the new planner issues different, perhaps over-specific, queries, which show up as misses because the old recording never saw them. Runs that do not diverge stay completed.
3. Take post-release `NO_PROGRESS` trajectories and look at the `ToolResult` content for queries that also appear in pre-release logs. If the same query returned hits before the release and `[]` after, the search service upgrade is at fault. The planner behaved the same, but the environment returned nothing. Harness replay of pre-release trajectories stays identical in that case.

Telemetry that settles it:

- `ToolResult.content` and result hashes for identical action keys across the two periods.
- `ModelDecision` argument distributions, such as query length and filters, before and after.
- The search service's own logs for the new index. A common culprit is an unpopulated index or a tenant filter that defaults to empty.

**D2.** The tool executed twice. The first process received `ToolCallApproved` and called `create_ticket`, and the ticketing system created ticket one. The process died before writing `ToolResult`. On recovery, resume found the call approved but not done and executed it again, as designed, with the same idempotency key `run_id:request_id`. The ticketing tool ignored the key and created ticket two.

The fault is in the tool, not the runtime. At-least-once execution is the documented contract, and it is safe only for tools that deduplicate. The 90-second gap matches the crash-and-restart time.

Fixes:

1. Make `create_ticket` idempotent. Pass the key to the ticketing API as an idempotency key, or check for an existing ticket with that key before creating one, as the chapter's example does.
2. For APIs that cannot deduplicate, use Chapter 16's reconcile hook, which looks up whether the action took effect before re-executing.
3. Add a test that cuts the log after `ToolCallApproved`, resumes, and asserts one external effect.

Set `idempotent=False` on the tool so the loop never retries it on transient errors either.

**D3.** The incident's document id is `inc-2026-02-tracking-latency`, and the search results show that id in brackets. `INC-2026-0217` is the incident number in the report's body. The new model cites the incident number, `[inc-2026-0217]`, lowercased into bracket form, instead of the document id. `citations_grounded` checks that each cited id appears verbatim in an observation, so it correctly rejects the citation, and the model keeps making the same substitution until `VERIFICATION_FAILED`.

The rejection notes show the cited string, and the `ToolResult` content shows the correct id, which pinpoints the mismatch.

- **Harness fix.** Make the rejection feedback actionable: list the valid source ids seen in observations, for example "citations must use one of: [inc-2026-02-tracking-latency], ...". Optionally, have the verifier accept known aliases mapped from document metadata, but only aliases recorded in observations, never fuzzy matches.
- **Prompt or tool fix.** State the citation format in the system prompt with an example, and have `search_docs` format each hit as `source_id=inc-2026-02-tracking-latency` so the id is unambiguous next to the incident number.

Then add both kinds of answers to the regression set, and run counterfactual replay of the new model over the failing trajectories to confirm the fix before shipping.
