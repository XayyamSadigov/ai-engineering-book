# Exercises — Chapter 7 — Model Selection and Routing

Solutions: `../solutions/ch07-solutions.md`


**Start here:** K2, K4, E1, P2, D1 (about 4 hours). The rest go deeper.

### Knowledge questions

**K1.** Name the ten selection axes and classify each as a hard constraint or a trade-off axis. Which one can be either, and what decides it?

**K2.** Define false accept and false escalation in a two-stage cascade. Which one costs quality, which one costs money, and why does router accuracy fail to capture the difference?

**K3.** Why does the selection procedure apply the quality floor to the lower bound of a confidence interval rather than to the point estimate? What does it cost to do so?

**K4.** What is expected calibration error, and why does an overconfident top bin limit what any escalation threshold can achieve?

**K5.** When is a small model with retrieval a better choice than a reasoning model? Give the conditions, not an example.

**K6.** What does pinning a model version protect against, and what new obligation does it create?

### Engineering questions

**E1.** Northwind's incident researcher uses tools and sometimes needs 150k tokens of logs. Its primary model has a 200k window and tool calling. Design its fallback chain from the illustrative catalog and state what the router should do when the primary is down and the request is 250k tokens.

**E2.** Product says a misrouted P1 ticket costs a human 20 minutes, while a misrouted P4 ticket costs nothing measurable. How would you change the cascade evaluation and the router to reflect this, and what new data would you need?

**E3.** A team proposes a model-as-router that reads every request and picks among four routes. Write the arguments for and against, the test you would require before accepting it, and the safeguards it needs in production.

**E4.** Your cascade's mean latency is 520 ms and its p95 is 2,050 ms. The SLO is p95 under 1,500 ms. List three design changes that could meet the SLO and the cost or quality price of each.

**E5.** Northwind classifies about 3 million tickets a month, each about 600 input and 40 output tokens, mostly during business hours. A team proposes moving `nw-general`'s share of that traffic to a self-hosted open-weight model. List the numbers you need to compare the two options, explain how traffic shape changes the answer, and name the evaluation you would run before any cost comparison matters.

### Practical exercises

**P1.** (about 2 hours) Add a third cascade signal to the router: agreement. Call the small model twice (the second time with a different temperature, or a second small model) and escalate when the labels differ. Extend `cascade_eval` to simulate it from collected outcomes and compare its utility with the self-reported confidence signal at two error prices.

**P2.** (about 2 hours) Make the cascade evaluation slice-aware: compute utility per priority (P1 to P4) with a different silent-error cost per priority, choose a threshold per slice, and extend the router so a route's `min_confidence` can depend on request metadata. Show that per-slice thresholds beat a single threshold on the ticket set.

**P3.** (about 60 min) Add a `deprecation_date` field to `ModelProfile` and a startup check that warns when any routed alias's pin expires within a configurable window and fails when it has passed. Add tests for both.

**P4.** (about 90 min) Extend the selection harness to evaluate effort levels as separate candidates: given a client and a list of effort levels, produce one row per level with its own quality, latency, and cost, and include them in the Pareto front.

### Debugging exercises

**D1.** After a routine prompt update to the ticket classifier, the monthly model bill for Northwind Assist rises 40% while accuracy on the evaluation set is unchanged. Traces show the `small_first` route's share of traffic is unchanged, `router.escalated` is true on 61% of its requests (it was 15%), and the attempts on escalated requests show outcome `low_confidence` with confidences clustered between 0.70 and 0.79. Diagnose.

**D2.** During a two-hour outage of the cloud provider, incident-research answers stopped citing log lines and several claimed that "no errors were found in the period." Traces from the window show `router.model` as `nw-reasoning`, but the completion's `model` field reports `longctx-2025-12`; `finish_reason` is `stop`, there are zero tool calls, and the decision's `substitutions` list is empty. The same requests before the outage were served by `nw-reasoning` with three to six tool calls each. What is wrong with the system, and which code path allowed it?

**D3.** A new embedding-based route classifier passes its offline evaluation with 92% route accuracy. Two weeks after launch, the share of requests on the `reasoning` route has fallen from 18% to 6%, the share decided at stage `default` has risen from 5% to 19%, and user ratings for analytical questions have dropped. No code or catalog change was deployed in that period. What happened, what telemetry confirms it, and what would have caught it earlier?

**D4.** On a Monday morning, cost per request on Northwind's `general` route rises about fivefold and its p95 latency moves from under 2 s to about 6.5 s. Quality ratings are unchanged and nothing was deployed. Calls to cloud models go through Northwind's internal inference proxy. Every trace on the route shows two attempts: the first on `nw-general` with outcome `error` and detail `ProviderUnavailableError`, after a latency of a few milliseconds, and the second `ok` on `nw-reasoning`. Other customers of the same provider report no incident. The catalog entry for `nw-general` was last changed eight months ago. Diagnose the root cause, explain why the router behaved as it did, and name the two changes that would have turned this into a loud, early failure.
