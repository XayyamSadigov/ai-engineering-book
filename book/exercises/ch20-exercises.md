# Exercises — Chapter 20 — Agent Architectures

Solutions: `../solutions/ch20-solutions.md`


**Start here:** K1, K3, E2, P2, D1 (about 5 hours). The rest go deeper.

### Knowledge questions

**K1.** For each of the nine patterns, name who owns the "what happens next" decision and give the number of sequential model calls on the critical path in the best case.

**K2.** Why does ReAct's input-token cost grow roughly quadratically with the number of steps while planner-executor's grows roughly linearly? State the assumption under which the advantage disappears.

**K3.** Explain the difference between reflection and evaluator-optimizer in terms of who owns the loop, what the evaluator can see, and which stop rules each can implement.

**K4.** In Project 5, why is the deterministic Definition of Done a hard gate while the rubric judge is advisory? Describe the failure you would introduce by reversing them.

**K5.** What does each fan-in policy (`all`, `quorum`, `any`) assume about the relationship between branches? Give a Northwind example for each.

**K6.** Why does the publish tool take an investigation id rather than the report text as an argument?

### Engineering questions

**E1.** The Northwind HR team wants an assistant that answers policy questions, files leave requests, and escalates harassment reports to a human. Choose an architecture, draw its structure, list each agent's tools and side-effect classes, and state where approval is required.

**E2.** Design two more deviation rules for Project 5 and specify their inputs, the reason string, and a test that shows each fires exactly when it should. One must use the deploy data.

**E3.** Project 5's seventeen sequential calls are too slow for a team that wants a draft within thirty seconds. Beyond the two changes named under Production considerations, propose changes that reduce critical-path calls without removing per-step isolation or the Definition of Done, and estimate the new call count on the critical path.

**E4.** A product manager proposes a three-level hierarchy (platform lead, tier leads, specialists) for incident research. Write the evaluation plan that would justify or reject it against Project 5, including the baseline, metrics, and decision rule.

### Practical exercises

**P1.** (about 3 hours) Add plan-driven parallelism to `IncidentResearchAgent`: add a `depends_on` field to its `PlanStep`, then execute steps whose dependencies are satisfied concurrently, with a configurable worker limit, keeping the trajectory recorded in plan order. Add a test asserting the same evidence ledger as the sequential version and fewer sequential rounds.

**P2.** (about 90 min) Implement memoization for read-only tools in `patterns/evaluator_optimizer.py` so that round two of `agent_generator` reuses round one's observations for identical calls. Show with `Meter` and tool counters that tool calls drop while answers are unchanged.

**P3.** (about 60 min) Add a four-eyes rule to `IncidentService.decide` (the approver must differ from the requester) and an audit field recording both. Cover it in the service, CLI, and API tests.

**P4.** (about 2 hours, needs a provider key) Record a cassette from a real provider for both sample alerts, commit it, and add a CI test that replays it. Then change the planner prompt and capture the miss report.

### Debugging exercises

**D1.** After a release, 30 percent of investigations end `failed` with "replan budget exhausted: step s3 (search_incidents 'pg-logi-prod seq scans after CHG-2026-0907') returned no evidence." Investigations of the same alerts passed the week before. What changed, and what do you check in the plans and step records to confirm it?

**D2.** A supervisor-based variant of the incident agent produces reports whose citations point only at `[incident_analyst result]` and never at metric or deploy ids, and the Definition of Done rejects every draft. Diagnose the cause from the ledger and event logs and propose the fix.

**D3.** In a variant, the publish run is driven by a real model with the default `LoopConfig` and no tool-call limit, and an `approver` callback approves any `post_report` call for an investigation a human has already approved, "so reviewers are not asked twice." The channel shows two identical reports for one investigation, posted eleven minutes apart. The record shows `published` with the second message id, and the publish run's event log shows a `Resumed` event by "recovery" followed by a second `post_report` request with a new request id and a second `ToolResult`. Reconstruct what happened and name the defects.
