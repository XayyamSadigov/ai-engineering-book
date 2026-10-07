# Exercises — Chapter 28 — AI Application Architecture

Solutions: `../solutions/ch28-solutions.md`


**Start here:** K1, K3, E2, P1, D2 (about 4 hours). The rest go deeper.

### Knowledge questions

**K1.** Name the four trust boundaries in an AI application's data path, the component that enforces each, and one attack that crosses each if the component is missing.

**K2.** The sequence diagram reserves 1.5 seconds for retrieval and 5.3 seconds for the model stream inside an 8-second total. Explain why `Budget.stage_timeout` returns the planned figure capped by remaining time rather than simply the remaining time, and what would go wrong if it returned only the planned figure or only the remaining time.

**K3.** List the seven artifacts that must be versioned to reproduce an answer, and for each say which table references it. Which of them can change without any deploy on your side?

**K4.** Under what conditions can a Server-Sent Events client actually resume after reconnecting with `Last-Event-ID`? What must the server have done, and what is the pragmatic alternative for a chat answer?

**K5.** Why does `JobService.get` return `404` for a job owned by another tenant instead of `403`? What information would `403` leak?

**K6.** State the decision rule for request/response versus SSE versus WebSocket versus polling in two sentences, and give one interaction in Northwind Assist for each.

### Engineering questions

**E1.** Northwind adds a third tenant with strict data-residency rules: its documents and prompts may not leave a specific region. Which components of the reference architecture change, which tables gain a column or a row, and where does the per-tenant provider decision live?

**E2.** The embedding model is being replaced. Design the migration using `index_versions` and `chunks`: which states does the new index pass through, what runs in the workers, which evaluation gates activation, and how is retrieval routed during the overlap? What is the storage cost of the overlap?

**E3.** A product manager asks for "live typing indicators and the ability to interrupt the assistant mid-answer". Decide whether this requires WebSocket. Specify the transport per feature and the server-side changes to cancellation.

**E4.** The worker pool is one deployment with one queue. Ingestion bursts delay agent jobs by minutes. Redesign the async tier: how many queues or pools, what each scales on, and what metric tells you the redesign worked.

### Practical exercises

**P1.** (about 3 hours) Replace `InMemoryJobQueue` and `InMemoryJobRepo` in the skeleton with SQLite-backed adapters (standard library only) that implement leasing with `lease_until`, so that a job whose lease has expired is returned to `queued`. Make the existing tests pass unchanged and add one that simulates a dead worker.

**P2.** (about 90 min) Add a `ContextBuilderPort` to the skeleton and an in-memory adapter that enforces a token budget (use a word count as the token estimate) with the order: system prompt, history, evidence. Emit a `context` SSE event with the number of evidence items that fit and the number dropped. Test that dropping happens from the lowest-scored evidence.

**P3.** (about 60 min) Implement `GET /v1/conversations/{id}/messages/{message_id}/lineage` that returns the stored lineage plus a synthesized "reproduce" payload: prompt version, model, index version, and chunk ids. Write a test that the payload for two tenants' messages with the same conversation id never crosses.

**P4.** (about 45 min) Write an architecture fitness test: parse `api_skeleton.py` with the `ast` module and fail if any function defined under the routers section references a name from the adapters section directly (not via `build_app`). Document the rule in a comment at the top of the test.

### Debugging exercises

**D1.** After a deploy, p95 time-to-first-token measured in the browser jumps from 1.4 seconds to 7.9 seconds, while the server-side `model.ttft_ms` attribute is unchanged at about 1.1 seconds and total completion is unchanged. The deploy included a new ingress controller version. Diagnose, and name the single header or setting you would check first.

**D2.** A support engineer reports that a `logistics` user saw a leave-policy answer stating "25 days per year, carry over up to 5 days", which is the `retail` policy. Traces show the retrieval span for that request returned chunk `ret-pol-001` with `request.tenant_id=logistics`, and the span has `cache.hit=true`. The retrieval query itself, when replayed, returns only `log-pol-001`. Name the root cause and the two lines of code to inspect.

**D3.** Nightly, the `ingest_document` queue depth climbs to about 4,000 and never returns to zero, and the same 12 job ids appear in worker logs every few seconds with `state=running`. Workers are healthy and other job types complete. Explain the mechanism, name the two fields on the jobs table that should have prevented it, and say what you would change.
