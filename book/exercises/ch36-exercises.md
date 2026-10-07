# Exercises — Chapter 36 — System Design Cases II

Solutions: `../solutions/ch36-solutions.md`


**Start here:** K1, K3, E3, P2, D2 (about 3 hours). The rest go deeper. The design cases themselves are the main practice: if you have not yet done the Try it first boxes, do at least one before these exercises.

### Knowledge questions

**K1.** For each of the five cases, name the architecture class and the one step of the ten where most of the design effort went. Explain in a sentence why that step dominated.

**K2.** The research agent runs synthesis outside the loop and the verifier after synthesis. What goes wrong if synthesis runs inside the loop? What goes wrong if the verifier is removed and the synthesizer is simply told to "cite carefully"?

**K3.** Explain why the analytics assistant's guard and the database role are both necessary. Give one failure the guard catches that the role would not, and one the role catches that the guard would not.

**K4.** Why does the serving platform admit requests by KV-token budget rather than by request count? Using the chapter's figures, how many 32,000-token requests equal one hundred typical interactive requests in KV memory?

**K5.** In the evaluation platform, why is a change of judge version recorded as a judge change rather than treated as part of the system under test? What would a dashboard show if this were not done?

### Engineering questions

**E1.** The research agent's budget is 12 searches, 40 documents, 6 iterations. A product manager asks for "deeper" reports. Propose how you would decide whether to raise each budget, which metrics you would watch, and what you expect to happen to cost and to citation support rate.

**E2.** The analytics assistant's semantic-layer share is 65 percent. Design the instrumentation and the review process that would raise it to 80 percent over a quarter without hand-writing every new metric. State who owns each metric definition.

**E3.** The vendor-onboarding workflow needs a new step: a credit check against an external bureau with a per-call fee. Where does it go in the state machine, what idempotency and retry semantics does it need, what does the audit record contain, and what happens if the bureau is down for a day?

**E4.** The serving platform's interactive pool is sized at five large-model replicas. A team wants to move its 12,000-token system prompt to the platform. Estimate the effect on KV capacity and TTFT with the chapter's figures and propose two alternatives to accepting it as is.

### Practical exercises

**P1.** (about 2 hours) Extend `sql_guard.py` with a `max_joins` check and an `EXPLAIN`-based plan check that takes a callable `explain(sql) -> dict` and rejects plans whose estimated rows exceed a cap. Acceptance: tests for both engines; a query with a missing join condition is rejected; an existing passing query still passes; the plan check is skipped with a recorded violation code when `explain` is not provided.

**P2.** (about 60 min) Build the result-equivalence checker for the analytics assistant: given two result sets as lists of dicts, decide equivalence with column-name normalization, multiset row comparison, and relative numeric tolerance. Acceptance: tests for reordered rows, reordered columns, `revenue` versus `REVENUE`, floating-point differences at 1e-7, and a genuine mismatch; a function that explains the first difference found.

**P3.** (about 3 hours) Implement the vendor-onboarding state machine on the Chapter 17 workflow engine with a fake ERP that fails on the first call 30 percent of the time and a fake approver. Acceptance: no run creates two vendors for one workflow (checked by the fake ERP's store); every transition appears in an append-only audit log with actor and step; an approval timeout moves the case to the exception state; a resumed exception continues from the failing step.

**P4.** (about 2 hours) Design the data model of the evaluation platform as pydantic models and write a migration-free in-memory store. Acceptance: a run can be compared with a baseline run producing per-case deltas and per-slice aggregates; a judge version change is visible in the comparison output; a suite is immutable once a run references it (an attempt to modify raises).

### Debugging exercises

**D1.** The research agent's dashboard shows coverage flat at 0.78 for two months while cost per report has risen from $0.60 to $1.10 and the budget-exhaustion rate has gone from 8 to 31 percent. Nothing in the agent's code changed. List three hypotheses, the trace fields that distinguish them, and the order in which you would check them.

**D2.** An analytics-assistant user reports that "net revenue by region last quarter" returned numbers that are 4 percent lower than the finance report. The SQL shown looks right, the guard approved it, self-consistency agreed 3 of 3. The nightly gold set passed. Where do you look first, and what single trace attribute would most likely explain it?

**D3.** After a serving-platform upgrade, interactive TTFT p95 doubled between 09:00 and 11:00 while accelerator utilization stayed at 55 percent and the request rate was normal. Batch jobs had finished at 06:00. Which two metrics would you pull first, and what is the most likely cause given the chapter's design?
