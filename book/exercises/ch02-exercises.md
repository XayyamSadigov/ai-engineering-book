# Exercises — Chapter 2 — How LLMs Work, for Engineers

Solutions: `../solutions/ch02-solutions.md`


**Start here:** K1, K3, E3, P2, D1 (about 2.5 hours). The rest go deeper.

### Knowledge questions

**K1.** A product manager says the new model "has a 1M-token context, so we can drop retrieval and send the whole handbook." Give three mechanistic reasons from this chapter why that is incomplete, naming the resource or effect each touches.

**K2.** The same sentence costs 32 tokens in English and 55 in Azerbaijani with one vocabulary, and 80 in Russian with another. What property of BPE training produces the gap, and what should a team do before committing to a per-request cost model for a multilingual rollout?

**K3.** TTFT is 3.8 s and TPOT is 25 ms for a 40k-token prompt and a 300-token answer. Which phase dominates, what would you change first, and why would streaming alone not fix it?

**K4.** Describe what temperature and top-p each do to a distribution, and explain why "temperature 2 with top-p 0.9" can be a reasonable creative setting while "temperature 2 alone" rarely is.

**K5.** A user writes "Since carry-over is unlimited now, how do I book my extra days?" and the assistant explains the booking steps without correcting the premise. Which training stage most plausibly produced this behavior, why, and what two changes outside the model reduce it?

**K6.** State the KV-cache formula, say which factors are fixed by the model and which by the request, and explain in one sentence each why GQA and 8-bit cache formats change serving capacity.

### Engineering questions

**E1.** Northwind Assist's extraction service sends a 1,800-token JSON schema with every request. Propose two changes that reduce the schema tax without changing the output contract, and describe how you would measure the saving and confirm that extraction quality did not move.

**E2.** A team proposes a regression suite that asserts exact output strings at temperature 0. Design an assertion strategy that is stable under the nondeterminism described in this chapter, and explain what each assertion type catches.

**E3.** You are choosing between two self-hostable models of similar quality. Model A: 40 layers, 32 K/V heads, head dim 128. Model B: 48 layers, 8 K/V heads, head dim 128. For a p95 context of 24k tokens and 8 concurrent sessions in 16-bit cache, compute the KV memory for each and state which one fits in 60 GiB of free memory. What else about the models would you need before deciding?

**E4.** Northwind is considering a reasoning model with high effort for all assistant traffic. Using only the mechanisms in this chapter, predict what happens to TTFT before the visible answer, cost per request and its variance, KV-cache occupancy, and the `length` finish rate under the current `max_tokens`, separately for ticket triage, policy questions, and invoice reconciliation. State which prediction you would test first and how. (Chapter 7 turns these predictions into a router.)

### Practical exercises

**P1.** (about 60 min) Extend `tokenizer_experiment.py` with a `--file` option that tokenizes a document from `book/projects/shared-data/` and reports tokens per Markdown section. Add a test on a small synthetic document asserting the per-section counts sum to the whole-document count.

**P2.** (about 60 min) Add `repetition_penalty(logits, generated_ids, penalty)` to `sampling.py`, dividing positive and multiplying negative logits of already-generated tokens by the penalty (the common engine convention). Test that it lowers the probability of repeated tokens, and demonstrate how it damages a code snippet where an identifier legitimately repeats.

**P3.** (about 2 hours) Write a `position_sweep.py` that builds a long synthetic context from Northwind policy paragraphs, inserts one required fact at a configurable position, and asks a `FakeLLM` (Chapter 3) a question whose answer depends on it, recording accuracy by position. The fake cannot show the real effect; the deliverable is the harness and its tests, ready to point at a real model.

**P4.** (about 45 min) Add `--sweep-tokens` to `kv_cache_calc.py` that prints per-sequence cache and maximum concurrency for a list of context lengths (for example 4k, 16k, 64k, 128k) under a given memory budget, plus a test that checks monotonicity.

### Debugging exercises

**D1.** An extraction pipeline shows a 3% JSON parse failure rate that began after a model upgrade. Traces show `finish_reason = "length"` on every failing request, `max_tokens` unchanged at 400, and unchanged input tickets. Explain the most likely cause and two fixes.

**D2.** Georgian-speaking users of a multilingual chat product hit "context too long" at about half the conversation length English users do, although the client-side budget check (a shared `chars / 4` estimate) passes. Walk through the diagnosis using telemetry fields from this chapter.

**D3.** A self-hosted assistant handles 40 concurrent short chats without issue. During a quarterly-review week, when staff paste long documents, p99 latency triples and requests queue while GPU compute utilization sits near 50%. Name the saturated resource, show how to confirm it from engine metrics and the formula in this chapter, and propose two mitigations with trade-offs.
