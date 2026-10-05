# Exercises — Chapter 25 — Evaluating AI Systems in Practice

Solutions: `../solutions/ch25-solutions.md`


### Knowledge questions

**K1.** Why must `traj_success` require `traj_safe`, and what would a dashboard of task-completion rates show for an agent whose runtime silently skipped approvals?

**K2.** Explain replay fidelity. Which planner changes can replay evaluate well, and which require a live sandbox run?

**K3.** In extraction, why is a wrong value counted as both a false positive and a false negative, and why does the evaluator score evidence only for fields whose value was correct?

**K4.** Macro-F1, per-class recall, and calibration answer different questions about a classifier. State the question each answers and the Northwind decision that depends on it.

**K5.** Why does checking a canary every hour with a 5% significance test produce more than 5% false rollbacks, and what does the planned-looks rule change?

**K6.** Describe two distinct biases synthetic question generation introduces and how each shows up in an evaluation report.

### Engineering questions

**E1.** Design the trajectory specification for a new agent task: "find the on-call engineer for the warehouse scanner service and page them if the service is degraded; never page for a healthy service." Give allowed tools, reference steps, budget, state predicates (including forbidden ones), and fixed arguments. Explain how you would evaluate the "healthy service" branch with replay.

**E2.** The extraction team wants to add contracts with 40 fields to the suite. Propose a weighting scheme, critical fields, evidence requirements, and slices, and explain how you would keep the fast suite under five minutes.

**E3.** Your team runs evaluations in GitLab CI on merge requests from forks as well as internal branches, and the nightly suite uses a real provider. Design the pipeline so that secrets are never exposed, the fast suite still runs for forks, and baselines can only be updated by the release process.

**E4.** Product wants the canary to promote as soon as possible when the new version is clearly better. Explain the risk of adding early promotion to the planned-looks rule, and propose a design that allows it without inflating the error you care about.

### Practical exercises

**P1.** Add a live-sandbox mode to the agent suite: run the candidate planner in a real `agentkit.AgentRuntime` over scripted, deterministic `FunctionTool`s (including a degraded service and a tool that raises a transient error once), selected per case when replay fidelity on that case falls below the gate threshold. Export the resulting event log with `trajectory_from_events`, and report which mode each case used in the run metadata.

**P2.** Extend `evidence_status` to verify the quote's label as well as its value (for example, a total's quote must contain "total" and must not contain "subtotal"), and add test cases from the untaxed statements that the current check passes incorrectly.

**P3.** Add a `summarization` suite to `suites.py` over the incident reports in `shared-data/docs/` (files starting with `inc-`), with key facts written by hand, a stand-in summarizer with baseline and regressed versions, and a gate section that requires coverage, faithfulness, qualifiers, and compression together.

**P4.** Implement a nightly job that runs the classification suite three times with a nondeterministic stand-in (seeded noise on confidence and occasional label flips), reports pass^3 and the flaky cases, and fails the gate when the flaky rate exceeds a configured limit. Add the rule to `gates.toml` through a new aggregate.

### Debugging exercises

**D1.** After a planner prompt change, the agent suite shows `traj_success` up from 0.75 to 1.0 and `traj_efficiency` up as well. `replay_fidelity` dropped from 1.0 to 0.55, and the gate passed because the fidelity rule was commented out "temporarily" last month. In the trajectories, many tool results have status `unrecorded` and the planner's final answers say the service could not be reached and a ticket was not needed. Diagnose what happened and what the gate should have done.

**D2.** The classification gate fails on `aggregate ece` with an observed value of 0.31 right after a model upgrade, while accuracy and macro-F1 improved. The reliability bins show almost every case in the 0.9 to 1.0 bin with an observed accuracy of 0.68 in that bin. Explain what changed, whether the release should be blocked, and what the team should fix.

**D3.** Online, the new prompt version shows a correction rate half that of the old version, yet labeled accuracy on delayed labels is unchanged. The join statistics show orphan events jumped from 2% to 41% the day the new version rolled out. Diagnose the cause and the telemetry that would confirm it.
