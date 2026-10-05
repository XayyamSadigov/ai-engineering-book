# Exercises — Chapter 21 — Memory Systems

Solutions: `../solutions/ch21-solutions.md`


### Knowledge questions

**K1.** Name the six memory types in this chapter, and for each give its typical writer and a reasonable default lifetime.

**K2.** Why does `memorykit` reject `retrieved_content` at the write policy instead of relying on the instruction-pattern heuristic?

**K3.** What is the difference between confidence and salience? Give a Northwind memory with high salience and low confidence.

**K4.** Why must tombstone fingerprints be keyed with a secret rather than a plain SHA-256 of the value?

**K5.** Explain why the relevance gate is applied before blending rather than folded into the weighted score.

**K6.** Why does a superseding record reference the old one as `supersedes:<id>` rather than `mem:<id>`?

### Engineering questions

**E1.** Northwind's HR system lists Ana's office as Berlin; Ana tells the assistant she moved to Lisbon last week. Design the end-to-end behavior: what is stored, what is shown to the model, what the user sees, and who is notified.

**E2.** The logistics business unit wants the incident agent to share episodic memory across both tenants because "outages are the same infrastructure". Evaluate the request and propose a design that captures the benefit without breaking tenant isolation.

**E3.** Your memory store uses a separate managed vector database synced from PostgreSQL by a change-data-capture job. Describe how deletion, expiry, and export must work across both stores, and what you would monitor.

**E4.** Product wants the assistant to "remember everything automatically, no confirmation prompts". Write the argument you would make, and the compromise you would propose, in terms of measurable outcomes.

### Practical exercises

**P1.** Add a `PostgresStore` implementing `MemoryStore` with psycopg and a pgvector column, plus row-level security on `tenant`. Make the existing store contract tests run against it when `DATABASE_URL` is set, and skip otherwise.

**P2.** Extend `WritePolicy` with per-key validity: a fact extracted as "on parental leave until 2027-03-01" should get `expires_at` from its value, capped by the kind TTL. Add tests.

**P3.** Build a "what I remember about you" endpoint in FastAPI with list, edit, and delete operations on profile and semantic memories for the authenticated user. Deletion must cascade and must return the tombstones. Test that one user cannot edit another's memory.

**P4.** Build a memory-on versus memory-off evaluation over five synthetic multi-session Northwind scenarios, including one stale-fact scenario and one override scenario. Report task success and repeated-information counts for both arms.

### Debugging exercises

**D1.** After an embedding model upgrade, users report that the assistant "forgot everything". The profile facts render correctly, but semantic recall returns nothing. The recall latency dropped by half. What happened, which telemetry confirms it, and what is the fix?

**D2.** A user deleted their personal phone number on Monday. On Thursday, the assistant greets them with a summary that mentions "the mobile number ending in 678". The profile store has no phone record and a tombstone exists. Trace the possible paths by which the number returned, and say which logs distinguish them.

**D3.** The incident agent now restarts the payment adapter as its first action on every POS ticket, including network outages where the restart cannot help. Success rate on network tickets fell. The episodic store has 40 success episodes for adapter restarts and none for network outages. Diagnose the mechanism and propose two fixes, one in memory and one in the agent.
