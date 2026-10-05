# Exercises — Chapter 13 — Grounded Generation and Citations

Solutions: `../solutions/ch13-solutions.md`


### Knowledge questions

**K1.** Why does the packer assign short evidence ids such as `E1` instead of passing chunk ids to the model? Give three reasons.

**K2.** The lexical support check treats numbers strictly but words leniently. Explain why, and name one class of unsupported claim it will miss.

**K3.** Explain why text inside a flagged instruction-like span is excluded from support even though it is genuinely part of the document.

**K4.** What is the difference between a `conflict_unreported` warning and a `stale_source_preferred` error, and why is one an error and the other a warning?

**K5.** Why must score floors for pre-generation abstention be defined per retrieval stage?

**K6.** Self-consistency keeps claims that most samples agree on. Describe the situation in which it confidently keeps a wrong claim.

### Engineering questions

**E1.** Northwind Legal wants every answer about contracts to include verbatim supporting text. Design the change across the generator configuration, the schema usage, the validator, and the UI, and estimate the effect on output tokens and latency qualitatively.

**E2.** The product team wants streamed answers for the support copilot and also wants an LLM groundedness judge on every answer. Propose a design that satisfies both as far as possible, and state what the user sees when the judge disagrees after text has been shown.

**E3.** Design how supersession should be represented at ingestion so the packer never needs the heuristic for HR policies. Specify the metadata fields, who maintains them, and how Chapter 15's indexing pipeline would propagate a new policy version.

**E4.** An answer about the expense policy cites five blocks from three documents, and the validator drops two of six claims. Walk through how the abstention policy decides, and argue whether `max_dropped_ratio` of 0.5 is right for an HR assistant versus an incident-response assistant.

### Practical exercises

**P1.** Add a regenerate-once path to `GroundedQA`: when the validator reports `stale_source_preferred`, or drops every claim while evidence is non-empty, call the generator again with the issues appended as feedback, validate again, and keep the better answer. Write tests with scripted models for both triggers, and assert the second call's request contains the feedback.

**P2.** Implement a `NegationGuard` judge hook that flags a claim when it contains a negation (not, never, no longer, except) that the cited evidence does not contain near the same content words, or vice versa. Test it with "Employees may not carry over unused PTO days" citing the PTO block.

**P3.** Add an `effective_date` field to the shared fixture by giving the PTO chunk metadata `effective_date: 2026-01-01` and the FAQ none. Write tests showing that conflict notes use the effective date when present, and design a case where `updated_at` and `effective_date` disagree on which document is newer.

**P4.** Build a small FastAPI endpoint `POST /answer` that runs `GroundedQA` and returns the envelope, plus `POST /answer/stream` that returns server-sent events from `GroundedStreamer`. Include a test that consumes the stream with an HTTP test client and asserts that no text event contains an unknown evidence id.

### Debugging exercises

**D1.** After a model upgrade, the abstention rate on answerable gold questions rises from 6 to 21 percent. Unknown-citation errors are flat, `unsupported_claim` errors have tripled, and spot checks show the new model's answers read well and look correct. Diagnose the likely cause and the telemetry and data that would confirm it.

**D2.** A user reports that the assistant answered "10 days" for PTO carryover but the citation card showed the HR FAQ, version 1.4. The trace shows the packed evidence had the FAQ as E1 and the policy as E2, the model's claim cited E2, and the validator reported no issues. Find the bug and say which test would have caught it.

**D3.** Security reports that an answer about Brightline invoicing included the sentence "Contact partners@brightline-supply.example for directory updates." The block containing the injection paragraph was in the evidence, the validator ran, and the claim was cited to that block. The trace shows no `support_only_flagged` error and no flagged-source event. Diagnose what failed and how you would make the defense less dependent on that component.
