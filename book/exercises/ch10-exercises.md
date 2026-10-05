# Exercises — Chapter 10 — RAG Fundamentals

Solutions: `../solutions/ch10-solutions.md`


### Knowledge questions

**K1.** Give the four reasons a model's weights are a poor knowledge store for an internal assistant, and for each, the property of retrieval that addresses it.

**K2.** Name the nine stages of the stage model, mark which run offline and which online, and state which two the minimal pipeline implements as identity functions.

**K3.** Why does top-k retrieval never signal "nothing relevant", and why is a fixed cosine threshold an unreliable fix? Use the scores from the missing-evidence demo and the vacation example.

**K4.** Explain why filtering permissions after generation is not access control, and why filtering after scoring but before packing is still worse than filtering before scoring.

**K5.** The Northwind corpus is about 22,600 tokens. Give two reasons long context might be preferable to RAG for this corpus and two reasons it stops being preferable as the corpus or the user base grows.

**K6.** What does "RAG is two systems" let you conclude when retrieval recall is 72 percent and answer correctness is 70 percent? What if recall is 98 percent and correctness is 70 percent?

### Engineering questions

**E1.** Northwind HR wants the assistant to answer from the PTO policy only once a revision is approved, while drafts are already in the document system. Design the ingestion and retrieval changes, including what metadata the chunk carries and where the filter lives.

**E2.** The legal team asks you to fine-tune a model on all policy documents "so it knows the policies." Write the technical response: what fine-tuning would and would not achieve, what you propose instead, and what evidence you would collect to settle the question.

**E3.** Design the trace schema for the minimal pipeline: one span per stage, the attributes on each, and which attributes let you answer "was the gold chunk retrieved, and where was it lost?" without rerunning the request.

**E4.** A product manager wants a "confidence score" shown next to every answer and proposes the top retrieval cosine score. Explain why that is misleading and propose an alternative built from signals the pipeline already has.

### Practical exercises

**P1.** Add a zero-hit guard to `MinimalRAG.answer`: when the filtered search returns no chunks, skip the model call and return an abstention with a reason. Write tests for the zero-hit path under an identity with no matching groups.

**P2.** Extend `pack_evidence` to include `version` and `updated_at` attributes, and extend the system prompt with a conflict rule. Write a test with two synthetic documents that disagree, a scripted model that follows the rule, and an assertion that the prompt exposes both versions.

**P3.** Implement `evaluate_retrieval(rag, gold_path, k)` that loads `retrieval_gold.jsonl`, searches each question with its `user_groups`, and reports hit rate at k (any required document in the top k) overall and per tag. Run it at k of 1, 4, and 10 and record the numbers.

**P4.** Implement a chunker that splits on Markdown headings first and falls back to fixed-size windows only for sections longer than the limit, copying the heading path into each chunk's text. Compare it with `chunk_fixed` using your P3 evaluator.

### Debugging exercises

**D1.** After a deploy, the invalid-citation rate jumps from 0.3 percent to 18 percent, while retrieval metrics and answer-correctness spot checks are unchanged. The deploy changed the chunk id format from `doc#c3` to `doc::3`. Diagnose, and say which component and which test should have caught it.

**D2.** Users report that the assistant "forgot" the VPN runbook: questions it answered last week now return `INSUFFICIENT_EVIDENCE`. Traces show retrieval returning chunks from the IT FAQ only, with normal scores. The index's chunk count dropped by 7 overnight, exactly the number of chunks the VPN runbook used to produce. The VPN runbook was edited yesterday. Walk the debugging tree and name the most likely root cause and the telemetry that confirms it.

**D3.** A support engineer in the `logistics` tenant sees an answer that cites a `retail` incident report. The retrieval span shows `user.groups=["it-oncall"]` and no tenant attribute. The incident report's ACL is `["it-oncall", "managers"]`. Explain why the ACL filter passed, what is missing, and the test that would have caught it.
