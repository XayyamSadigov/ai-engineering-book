# Exercises — Chapter 39 — Capstone: Northwind Assist

Solutions: `../solutions/ch39-solutions.md`


### Knowledge questions

K1. Why does `prepare` run before the response stream starts, and what would a client observe if admission rejections were sent as `error` events inside a `200` stream?

K2. The capstone's guardrail tool rules do not require approval for `send_reply`, although sending must always be approved. Explain where the approval is enforced and why a second approval gate in the guardrail would be harmful rather than redundant.

K3. In the ACL-disabled run, recall@5 barely changes while `no_permission_leak` falls to 0.788. Why can the pipeline's final ACL check not catch this bug, and which check does?

K4. What is the difference between the retrieval cache and the answer cache in what they store, and why does that difference make one of them safer to enable by default?

K5. Explain why the model-facing tool schema accepts PII tokens in email fields while toolkit validates real addresses. What attack does re-hydrating only inside the tool layer prevent?

K6. What does `attack_detected` measure, why is it reported and not gated, and what would happen to the system over time if it were gated?

### Engineering questions

E1. A product manager asks for per-department document permissions (finance, legal) in addition to tenants and groups. List every component in the capstone that must change, in the order you would change them, and the test that proves each change.

E2. The provider adds a feature that caches long prompt prefixes at a lower price. Which parts of the request path should change to earn the discount, and how would you show the saving in the cost report without double counting?

E3. Design the canary verdict for a change that swaps the general model for a cheaper one. Which per-arm metrics would you compare, over which traffic, and what is the rollback rule?

E4. The service desk wants the assistant to send routine "your ticket was resolved" replies without approval. Argue for or against, and if for, specify the policy rule, its constraints, and the evaluation evidence you would require first.

### Practical exercises

P1. **Durable approvals.** Replace toolkit's in-memory `ApprovalManager` with Chapter 38's `InterruptManager` (SQLite) for `send_reply`, with a 15-minute expiry and escalation to a second lead after 10 minutes, and persist the requester's context (user, tenant, scopes, groups) with each approval. Acceptance: a pending approval survives an API restart and can be approved afterwards, with policy re-checked against the persisted requester context (the restored approval executes under the persisted scopes and groups, never under defaults); an expired approval cannot be executed; escalation is visible in `GET /v1/approvals`; all existing tests pass.

P2. **Real-provider integration suite.** Add `@pytest.mark.integration` tests that run the grounded-answer, agent and extraction paths against one real provider configured with `LLM_PROVIDER` and `NA_MODEL_MAP`, plus an eval run whose report is compared with the offline reference. Acceptance: skipped by default; with credentials, all pass; the report shows per-stage latency and cost from real spans; no test asserts exact model text.

P3. **Intent classifier with a baseline.** Label 200 user messages with their workflow, train or prompt a classifier, and route with it only when its confidence is above a threshold, falling back to the rules. Acceptance: on a held-out set the combined router beats the rules on macro-F1 by at least 5 points, never routes a question to a side-effecting workflow below the threshold, and its decision appears in `meta` and on the `router.decide` span.

P4. **Calibrated online judge.** Sample 1 percent of production answers into an evaluation job that runs ragkit's faithfulness judge with a separate model, writes scores as `eval.score` spans joined by `response.id`, and reports agreement with 50 human labels. Acceptance: Cohen's kappa against the human sample is reported per judge version; a judge below 0.6 cannot be used to gate; the dashboard shows faithfulness by version fingerprint.

P5. **Compose end to end.** Bring up the Compose stack with the Project 3 backend, sync the corpus through the worker, and run the security and RAG suites against the running API over HTTP. Acceptance: the same gate passes; a document deleted through the admin API disappears from answers within the freshness SLO; the collector shows one trace per request with the worker's ingestion spans in separate traces.

### Debugging exercises

D1. After a deploy, the cost dashboard shows retail spend dropping by 60 percent while request volume is flat and no cache settings changed. Thumbs-down feedback rose. The `request` spans show `degrade.level = 1` on most requests, and `/v1/admin/status` shows both breakers closed. What is the likely cause, which attribute or setting confirms it, and what would you change?

D2. A lead reports that approving a reply returns `403 approval_mismatch` even though nobody edited the text. The audit log shows `tool.approval_requested` with one `args_hash` and `tool.denied` on execution with a different one. The requester's message contained an email address. Where do the two hashes come from, and what changed between proposal and execution?

D3. In a load test, p95 time to first `delta` measured at the client is 7.8 seconds, while the `llm.complete` spans show a provider time to first token of 0.9 seconds and the `request` spans finish in 8.1 seconds. No errors are logged. Name two causes consistent with these numbers and the trace or header that distinguishes them.
