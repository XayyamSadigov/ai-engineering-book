# Exercises — Chapter 22 — Multi-Agent Systems

Solutions: `../solutions/ch22-solutions.md`


**Start here:** K2, K3, E4, P1, D2 (about 5 hours). The rest go deeper.

### Knowledge questions

**K1.** Name the five reasons that can justify multiple agents. For each, state the property you would measure to confirm it applies.

**K2.** Explain why a sequential single agent's input tokens grow roughly with the square of the number of tool steps while a supervisor-worker team's grow roughly linearly in the number of workers. What does the batched single agent change?

**K3.** What is the difference between the `skipped`, `budget_exhausted`, and `failed` statuses in a result envelope, and why does the parent need to distinguish them?

**K4.** Why is an append-only event log a better coordination record than a shared mutable findings document? Name two concrete failures of the shared document.

**K5.** What problem does a synthesis holdback solve, and what goes wrong without it?

**K6.** Why does Project 6 withhold the researchers' quotes from the verifier?

### Engineering questions

**E1.** A product manager wants a "debate" mode in which two researcher agents argue about each answer for three rounds before a judge decides. Estimate the cost multiplier relative to the current team, name the failure modes you expect, and propose the measurement that would decide whether to ship it.

**E2.** Northwind wants an agent that can read HR employee records (group `hr`) to answer managers' questions about their team's leave balances. Design the permission boundary: which agents exist, which principal and tools each has, what crosses between them, and how you would test that a prompt injection in a policy document cannot exfiltrate a record.

**E3.** The team's p95 latency on cross-cutting questions is 11 seconds against an 8-second target. Using the structure of Project 6, list the changes you would make in order of expected impact, and explain how you would verify each without degrading quality.

**E4.** Your benchmark shows the team ahead of `single+verify` by 0.4 rubric points at 1.3 times the tokens, on 8 questions. Is that enough to promote the team? What would you do before deciding?

**E5.** A partner logistics company offers its customs-rules research agent to Northwind over an agent interop protocol. Northwind wants Project 6 to delegate customs subquestions to it. Describe what changes for that one child: what the task message may contain, how the budget and deadline are enforced, how its claims are verified, how the trace is joined, and which statuses the supervisor must handle that it does not handle today.

### Practical exercises

**P1.** (about 3 hours) Implement pipelined verification: verify each researcher's claims as soon as that researcher finishes, in the same worker thread, instead of after the dispatch barrier. Keep the global budget invariant and the "both must agree" rule. Measure the change in wall time and tokens with the benchmark.

**P2.** (about 3 hours) Add a context-window constraint to the offline evaluation: make the scripted single agent's quality degrade when its transcript exceeds a configurable size (for example, by extracting claims only from the first and last passages in its context when the transcript exceeds the limit), then add questions that touch six or more policy areas. Report where, if anywhere, the team starts to pay off, and state clearly in the report that the degradation model is an assumption.

**P3.** (about 90 min) Add a cost-based global budget: give the team a `PricingTable` and `max_cost_usd`, reserve cost as well as tokens at admission, and add a test showing that a run stops admitting researchers when the cost pool is exhausted even if tokens remain.

**P4.** (about 3 hours) Replace the deterministic rubric's faithfulness criterion with an LLM judge built on Chapter 24's evalkit, run it on the offline answers with a scripted judge, and write the calibration procedure you would follow before trusting it on live answers.

### Debugging exercises

**D1.** After a planner prompt update, the cost per cross-cutting question rose 70% while the rubric score was unchanged. The team logs show `spawn_refused` events with reason `spawn_cap` on 40% of runs, and `duplicate_work` events with an average of nine duplicate claims per run, up from three. Diagnose what happened and say which telemetry confirms it.

**D2.** Users report that answers about working abroad say "up to 20 working days per year" but never mention that the request must be made ten working days in advance. Retrieval metrics are unchanged. In the researcher event logs, the eligibility passage appears in `ToolResult` events for the relevant runs. Identify the failure mode, explain how you distinguished it from a retrieval failure, and propose a fix.

**D3.** In a variant of the team with `max_rounds=1`, a run's status is "complete", but the answer to a four-area question covers only three areas. The team log shows four `task_dispatched` events and four `task_finished` events, all `succeeded`. One researcher's result envelope has an empty claims list and one gap, and its `verified` event lists no accepted claim from that task. What is wrong with the status logic, which telemetry proves it, and what should the run have reported?
