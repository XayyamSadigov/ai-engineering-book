# Exercises — Chapter 30 — Performance and Cost Engineering

Solutions: `../solutions/ch30-solutions.md`


**Start here:** K3, K4, E1, P3, D1 (about 4 hours). The rest go deeper.

### Knowledge questions

**K1.** Explain why the end-to-end p95 latency of a request is usually lower than the sum of its stage p95s, and why the sum is still useful in budgeting.

**K2.** A team says provider prompt caching will cut their cost because "the model will remember the previous conversation". What is wrong with this description, and what actually determines a cache hit?

**K3.** List the components a retrieval-cache key needs for a multi-tenant assistant with document ACLs, and state what goes wrong if each one is missing.

**K4.** Why is cost per successful task a better unit than cost per model call when comparing a single strong-model call with a multi-step agent on a cheaper model?

**K5.** Name the three things called batching in an AI system and the situation in which each one is appropriate.

**K6.** Why does a spend guard need reservations rather than a check of remaining budget before each call?

### Engineering questions

**E1.** Northwind wants to add a guardrail model call after generation that checks every answer for policy violations. It takes 600 ms at p95. Using the chapter's budget (8 s total, 2 s TTFT, streaming answers), where can it go, and what design choices does each placement force?

**E2.** The logistics tenant has a 9 percent semantic-cache hit rate, and retail has 2 percent. Retail's product owner asks to lower the similarity threshold for retail only. What data would you collect before deciding, and what would make you say no?

**E3.** A finance stakeholder asks for chargeback that includes the shared vector database, the platform team's on-call cost and an idle failover GPU. Propose an allocation rule, explain what behavior it encourages in tenants, and name one rule you would avoid.

**E4.** You run three replicas of the API behind a load balancer and the in-memory `SpendGuard`. Describe the failure this causes and design the shared-ledger replacement, including what happens when the ledger store is unavailable.

**E5.** Northwind is considering one system prompt per tenant, each about 3,000 tokens, instead of one shared prompt. The provider charges an illustrative 1.25 times the input price to write a prefix into its cache, 0.1 times to read it, and entries live 5 minutes after their last use. The retail tenant sends about 40 requests an hour and a small tenant about 3 an hour. Estimate the expected prefix cost per request for each tenant with and without the split, and recommend a layout.

### Practical exercises

**P1.** (about 90 min) Extend `LatencyTracker` to report, per stage, the share of end-to-end violations in which that stage itself exceeded its budget. Add a test with synthetic spans where retrieval causes most violations.

**P2.** (about 2 hours) Implement a `ToolResultCache` for read-only tools with per-tool TTLs, a tool-version component, and a refusal to cache any tool not in a read-only registry. Declare its `key_components` and make it pass `lint_cache_key("tool", ...)`.

**P3.** (about 2 hours) Build a threshold-tuning script for `SemanticCache`: given labeled question pairs (same answer or not), compute hit rate and false-hit rate for thresholds from 0.70 to 0.99 using `FakeEmbeddings(vocabulary=...)`, and pick the lowest threshold whose false-hit rate is at or below a target.

**P4.** (about 2 hours) Add an hourly spend anomaly detector to `cost.py`: from trace JSONL, compute spend per tenant per hour and flag hours more than three times the trailing 7-day median for the same hour of the week. Test it with a synthetic spike.

### Debugging exercises

**D1.** After a prompt release, cost per answer rose about 14 percent while traffic, model and average input tokens were unchanged. Traces show `cached_input_tokens` per call fell from about 800 to near zero. The diff of the release shows the system prompt now starts with "You are Northwind Assist. Today is {date} {time}." and tool definitions are emitted from a Python `set`. Diagnose the cause and the fix, and name the telemetry that confirms the fix.

**D2.** A warehouse supervisor in the logistics tenant reports seeing an answer that quotes salary bands, which only HR should see. The retrieval cache hit rate is 35 percent. The retrieval cache key is built from `normalize_text(query)`, `tenant` and `index_version`. ACL filtering happens in the SQL query on a miss. Explain how the leak happened, which spans show it, and what change and test prevent recurrence.

**D3.** Retail's daily limit is 50 USD in `enforce` mode, yet yesterday's committed spend was 210 USD. The guard logged no blocked alert until 14:05, and traces show 4,000 requests admitted between 13:58 and 14:05 from a batch summarization job. The service runs four replicas. Identify the contributing causes and the fixes.
