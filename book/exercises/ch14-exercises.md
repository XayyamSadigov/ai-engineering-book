# Exercises — Chapter 14 — RAG Evaluation

Solutions: `../solutions/ch14-solutions.md`


### Knowledge questions

**K1.** A question requires two documents. The retriever returns ten chunks: chunks of the first required document at ranks 3 and 4, an acceptable document at rank 1, and nothing from the second required document. Compute hit@5, recall@5, precision@5 (chunk level), reciprocal rank, and nDCG@5 with required = 2 and acceptable = 1.

**K2.** Why are forbidden-document cases excluded from recall averages, and what two checks replace recall for them?

**K3.** Explain the difference between faithfulness and correctness for a RAG answer, and give a Northwind example that scores high on one and low on the other.

**K4.** List four biases of LLM-generated synthetic questions and state, for each, whether a mechanical filter can reduce it.

**K5.** Why does stage isolation report the earliest stage that lost evidence rather than the last, and why does a permission leak override every other label?

**K6.** What does context relevance measure that recall and evidence sufficiency do not?

### Engineering questions

**E1.** Northwind is adding 2,000 scanned PDF invoices and contracts to the corpus. Design the additions to the gold set (labels, tags, permission contexts, abstention cases) and say which granularity of label you would use and why.

**E2.** A team proposes a single "RAG score" = 0.4 × recall@5 + 0.4 × faithfulness + 0.2 × citation precision for the release dashboard. Write the response: what the score hides, and what you would put on the dashboard and in the gate instead.

**E3.** Design the evaluation for adding a cross-encoder reranker: which metrics you compute on which lists, the k values, the slices, the latency budget, and the gate rules that would block it.

**E4.** Your faithfulness judge's calibration shows a false pass rate of 8 percent overall and 30 percent on `conflicting-versions` cases. Decide how the gate should use the judge, and what you would change in the judge or the dataset.

### Practical exercises

**P1.** Add span-level labels to `RagExpectation` (`required_spans: list[str]`) and a `span_recall_at_k` metric that counts a required span as found when it occurs (whitespace-insensitive) in any of the top k chunks. Label five gold cases and show where span recall and document recall disagree.

**P2.** Add a `first_stage_recall` evaluator that reads each candidate list from the trace and reports recall@50 per list and for their union, so a report can show coverage before fusion and reranking.

**P3.** Implement a judge cache for `FaithfulnessJudge` keyed by judge version, a hash of the answer, and the packed chunk ids, with a JSONL backend. Show with a test that re-scoring a stored run makes no judge calls.

**P4.** Extend `synthesize_questions` with an embedding-based dedupe using `aie_core` embeddings and a multi-chunk mode that shows the model two chunks from different documents and asks for a question requiring both. Report how the difficulty distribution changes.

### Debugging exercises

**D1.** After a retriever refactor, the stage table shows 0 `dropped-by-fusion` and 0 `dropped-by-rerank` cases for a week, while `not-retrieved` doubled. Recall@50 on first-stage lists is unchanged. Diagnose the likely cause and name the trace fields you would inspect.

**D2.** Faithfulness rose from 0.86 to 0.97 after a prompt change, rubric coverage stayed flat, and the abstention table shows false abstains rising from 4 to 15. Explain what happened and how the report should have made it obvious.

**D3.** A candidate configuration passes the gate. In production, a retail store manager receives an answer citing a logistics incident report. The evaluation run shows `no_permission_leak` at 1.0 on all cases. List the ways the evaluation could have missed the leak, most likely first, and the change to the evaluation that would catch each.
