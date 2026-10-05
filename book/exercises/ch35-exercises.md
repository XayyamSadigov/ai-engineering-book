# Exercises — Chapter 35 — System Design Method and Cases I

Solutions: `../solutions/ch35-solutions.md`


### Knowledge questions

**K1.** Why does the method require correctness to be defined in step 1 before architecture in step 2? Give one concrete consequence of skipping it for each of the four cases.

**K2.** State Little's Law and apply it to a system receiving 12 requests per second with an average end-to-end latency of 7 s. Explain what the result means for a model gateway with a concurrency limit of 64.

**K3.** In Case 1 the cache key includes tenant, sorted group set, index version, and prompt version. Explain what goes wrong if each of the four is omitted.

**K4.** Why is the document processing system in Case 3 a workflow rather than an agent, while the coding assistant in Case 4 is an agent? Use the decision criteria of Chapter 17.

**K5.** In the voice latency table, which stage is most often the true bottleneck in practice, and why does optimizing model time to first token alone fail to reach a one-second target?

**K6.** Explain why the review rate, not the token price, is the dominant cost term in Case 3, and compute the daily cost change if the review rate falls from 15 percent to 10 percent with the chapter's assumptions.

### Engineering questions

**E1.** Case 1 release two adds a `create_ticket` tool so employees can open IT tickets from the assistant. Walk through which of the ten steps change, what new rows appear in the security and failure tables, and what the evaluation set must gain.

**E2.** The support copilot's prefix cache hit rate drops from about 50 percent to 10 percent after a change. List three design changes that could cause this, the telemetry that distinguishes them, and the daily cost impact using the chapter's figures.

**E3.** A stakeholder proposes raising the Case 1 evidence budget from 8 chunks to 20 to "improve recall." Compute the token and cost impact per day, state what metric you would require before accepting, and describe the alternative you would propose first.

**E4.** The coding assistant must support a monorepo of 2 million lines. Redesign the retrieval step: which indexes, which tools, what working-set policy, and how you would evaluate that the agent still finds the right files.

### Practical exercises

Each is a design exercise. Deliver the completed worksheet (all ten rows), one Mermaid diagram with the trust boundary marked, the arithmetic for step 9 using `back_of_envelope.py`, and a failure table with at least six rows.

**P1.** Design an HR onboarding assistant for Northwind that answers new-hire questions, pre-fills forms from HR data, and schedules required training sessions. Acceptance criteria: side-effect classes identified for every tool; at least one tool requires confirmation bound to arguments; cost per new hire computed; degraded mode defined for the scheduling system being down.

**P2.** Design the voice channel of Case 2 for a 3,000-calls-per-hour peak with a 1.2 s p95 time-to-first-audio target. Acceptance criteria: latency table whose total meets the target with named overlaps; concurrency computed for audio, STT, TTS, and model streams; barge-in cancellation path described; a written rule stating which tool calls may use partial transcripts (expected: none that write).

**P3.** Design a contract-renewal alerting pipeline on top of Case 3: extract renewal dates and notice periods, then notify owners 60 days before notice deadlines. Acceptance criteria: field-level evaluation plan with critical-field weighting; idempotent notification design; an explicit decision, with numbers, on whether fine-tuning is justified for the two largest contract families.

**P4.** Extend Case 4 with a "fix the failing CI build" mode triggered by a CI failure webhook. Acceptance criteria: Definition of Done written as deterministic checks; the forbidden-path list; budget per task in steps, tokens, time, and cost; the evaluation suite's source of historical tasks and the clarification-or-stop cases it must include.

### Debugging exercises

**D1.** In Case 1, the groundedness score on the gold set is unchanged but the production "insufficient evidence" rate tripled over a week. Traces show recall@50 is normal and recall@8 fell from 0.91 to 0.62. Nothing was deployed. Diagnose, naming the stage and the two most likely root causes, and state the telemetry that confirms each.

**D2.** In Case 2 voice, a customer reports a ticket created for an order they explicitly said not to cancel. The call state shows a `create_ticket` event with a confirmation turn present. Describe how to use the per-stage timestamps of STT partials, finals, and the orchestrator's decision to establish whether the tool was called from a partial transcript, and name the two code paths to inspect.

**D3.** In Case 4, cost per agent task doubled in a week while steps per task and task success stayed flat. Using the chapter's step-9 arithmetic, identify which single telemetry field almost certainly changed, explain the mechanism, and name the context-builder change that most often causes it.
