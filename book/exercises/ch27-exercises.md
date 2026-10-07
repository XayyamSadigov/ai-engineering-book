# Exercises — Chapter 27 — Guardrails and Safe Tooling

Solutions: `../solutions/ch27-solutions.md`


**Start here:** K1, K5, E1, P2, D1 (about 5 hours). The rest go deeper.

### Knowledge questions

**K1.** Name the four pipeline stages and, for each, one guardrail that is a boundary and one that is a sensor, or explain why the stage has no natural sensor.

**K2.** Why does the injection heuristic default to flagging rather than blocking, and why does it fail open? Under what conditions, if any, would you configure it to block?

**K3.** Explain how the nonce in `wrap_untrusted` prevents a document from closing the untrusted-data block. What does the wrapper still not prevent?

**K4.** Why do card and IBAN detectors use checksums, and what class of error does each checksum remove? Give one Northwind string that a phone regex without validation would wrongly report.

**K5.** What is the difference between masking and tokenization, and what three conditions must hold for `PIIVault.rehydrate` to return a value?

**K6.** Define false-positive rate, bypass rate, and effect bypass rate as used in this chapter. Which one would you put in a CI gate first, and why?

### Engineering questions

**E1.** Northwind wants to add a public-facing returns chatbot for retail customers, reusing the guardrail pipeline. Which checks change their fail mode, thresholds, or action, and which new checks are needed? Justify each change by the asset it protects.

**E2.** The output stage currently replaces off-allowlist links with a visible "[link removed]" marker. Product asks for zero markers ("it looks broken"). Argue for or against, and propose a design that satisfies security and product without hiding removals from audit.

**E3.** Design the streaming variant of the output stage for the RAG assistant: buffering rules, which checks run per chunk and which on the full answer, what happens when a later check blocks after earlier chunks were shown, and the latency cost.

**E4.** The LLM classifier adds latency to every request. Propose an architecture that keeps its recall benefit on suspicious traffic while removing it from the critical path for most requests, and describe how you would verify that the change did not raise the effect bypass rate.

**E5.** Security proposes replacing the injection heuristic and `KeywordModerator` with an open guard model served in-house, and product proposes also dropping the tool-stage recipient allowlist "since the guard model catches exfiltration attempts." Decide which part of each proposal to accept, where the guard model plugs into the pipeline, and what evidence you would require before switching.

### Practical exercises

**P1.** (about 2 hours) Add a `national_id` detector to `pii.py` for an identifier format of your choice with a real checksum, including at least three false-positive tests drawn from Northwind-style reference numbers, and show the measurement of its false-positive rate on the shared tickets.

**P2.** (about 3 hours) Implement a streaming output guard, `StreamingOutputGuard`, that wraps an `aie_core` stream, buffers to safe boundaries, runs the output stage on each buffered segment, and never yields text a check has not seen. Test it with a markdown image split across three chunks.

**P3.** (about 2 hours) Write an adapter that implements the `authorize(call, ctx)` delegate on top of Chapter 16's `toolkit.policy.PolicyEngine`, mapping its allow, deny, and needs-approval decisions to `ToolDecision`, and add a red-team test where the engine's group deny stops a call that the guardrail rules alone would allow.

**P4.** (about 3 hours) Extend the measurement script with a per-check false-positive gate (`--max-fp check=rate`) and a labeled set of at least 30 additional benign user questions from your own domain. Report how the heuristic's FP and bypass rates change when you tune one signal weight, and keep the change only if both intervals support it.

### Debugging exercises

**D1.** After a deploy, the support agent stopped sending any replies, even approved ones. Traces show:

```text
guardrail.tool  tool=send_reply  action=block  blocked_by=tool_policy
guardrail.check check=tool_policy  action=block  reason="approval required for these exact arguments"
                error=false  findings=[]
ui.approval     approval_token=9c1e...  status=approved
guardrail.tool  tool=send_reply  action=block  blocked_by=tool_policy  (retry 2 s later)
```

The approval UI shows the same recipient and body. Diagnose the most likely root cause, name the field you would compare, and state the fix.

**D2.** The RAG assistant's input block rate jumped from 0.2 percent to 9 percent overnight with no code change. All blocks have `blocked_by=injection_classifier` and `error=false`, with confidence scores clustered near 0.91. What changed, how do you confirm it from telemetry, and what immediate and durable remediations do you apply?

**D3.** A security review finds full customer email addresses in the trace store, in spans named `llm.complete`, even though every guardrail span shows only hashes and sizes. The guardrail pipeline is configured with `RedactingTracer`. Where is the leak, and what test would have caught it?
