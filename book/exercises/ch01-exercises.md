# Exercises — Chapter 1 — What AI Engineering Is

Solutions: `../solutions/ch01-solutions.md`


### Knowledge questions

**K1.** State in two sentences how AI engineering differs from ML engineering, naming the primary lever and the primary artifact of each.

**K2.** For each of the five layers, name one failure that belongs to it and one commonly attempted fix that belongs to a different layer.

**K3.** Why does the decision ladder place the agent loop last, when agents are the most capable architecture? Answer in terms of the state space that must be evaluated and secured.

**K4.** A colleague says, "The model returned valid JSON for the `send_reply` tool, so we executed it." Name the mental model this violates and list four checks that should have run between the model's output and the execution.

**K5.** Rank the following as evidence for the claim "model X is better at extraction than model Y": a vendor blog post with a benchmark table; a replicated academic result on a public extraction benchmark; a 300-case evaluation set built from your own invoices; a colleague's report that X "seemed better" on a few documents. Explain the ranking.

**K6.** Why does the lineage record store both the prompt registry version and a content hash of the rendered prompt?

### Engineering questions

**E1.** Northwind wants a feature that classifies incoming support tickets into one of twelve categories and routes them to a queue. Walk the ladder for this requirement. Which rung do you stop at, and what measurement would justify climbing one more?

**E2.** A different team proposes an agent for the policy assistant: "it will decide which knowledge sources to search and when to ask clarifying questions." Write the justification you would require before accepting an agent loop here, and describe the deterministic workflow you would propose as the baseline to beat.

**E3.** Design the minimal set of lineage fields for an extraction API (Project 1) that has no retrieval and no tools. Which fields from `RequestLineage` drop out, which remain, and which new field becomes essential for an extraction task?

**E4.** The policy assistant must respect document permissions. Explain why the permission filter must run before ranking rather than after, and describe one way a post-ranking filter can leak information even when the filtered document never appears in the final answer.

### Practical exercises

**P1.** Extend `RequestLineage` with per-stage latencies (retrieval, rerank, model, gates) and a method that returns the stage that consumed the largest share. Add a test with a record whose total latency is dominated by retrieval and assert the method names it.

**P2.** Write a function that takes a list of `RequestLineage` records and produces a completeness report: for each lineage question, the fraction of records that cannot answer it. Test it with a mix of complete and incomplete records.

**P3.** Add a consistency check for tools: a `ToolEvent` with outcome `pending_approval` must not coexist with a `PolicyGate` decision of `allow` for an approval gate on the same request. Write the failing case first, then make it pass.

**P4.** Draw (in Mermaid) the five-layer data flow for a ticket-classification endpoint with no retrieval and no tools. Mark which layers are present, which are degenerate, and where the trust boundary is.

### Debugging exercises

**D1.** A trace shows: `prompt.version = "7"`, `evidence = []`, `gates = [tenant_filter: allow, evidence_gate: allow]`, `output_hash` non-empty, `eval_outcome = FAIL` with the note "answer not grounded." The evidence gate allowed an answer with no evidence. Identify the fault and the layer it belongs to, and name the trace field that reveals it.

**D2.** Over a week, the policy assistant's offline pass rate falls from 0.94 to 0.86. Prompt version, index version, and gate configuration are unchanged in every trace. `model.version` is null in all of them. What is the most likely cause, what in the lineage design allowed it to go undetected, and what two changes would you make?

**D3.** A `logistics` employee receives an answer citing `hr/retail-bonus-plan.md`, a document tagged `groups: ["retail"]`. The trace for the request shows `principal_groups = ("all", "logistics")` and the evidence list includes the chunk with `acl_groups = ("retail",)`. The retrieval service's permission filter has unit tests and they pass. List three hypotheses that are consistent with these facts and the single additional trace field that would distinguish them.
