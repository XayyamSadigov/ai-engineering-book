# Exercises — Chapter 5 — Context Engineering

Solutions: `../solutions/ch05-solutions.md`


**Start here:** K2, K5, E2, P1, D1 (about 3 hours). The rest go deeper.

### Knowledge questions

**K1.** List the four costs of an input token named in this chapter, and give one production metric that observes each.

**K2.** Why does `ContextBuilder` raise on a pinned item that fails the permission check, instead of dropping it like any other item?

**K3.** Explain why the untrusted-data notice is rendered on every request, even requests that contain no untrusted items.

**K4.** Name three kinds of state that must never exist only in a model-written summary, and say where each should live instead.

**K5.** What does a section floor protect against that a section cap cannot? Give a Northwind example.

**K6.** The builder's prefix hash stays constant across requests, but the provider reports almost no cached input tokens. Give two plausible causes.

**K7.** Your provider caches only up to explicit markers, and you may set up to three per request. For a multi-turn RAG conversation rendered in this chapter's layout, where do you place them, and what does each one let the next request reuse?

### Engineering questions

**E1.** Northwind's incident-research agent (Project 5) makes up to 30 tool calls per task. Its largest tool result is a log search that returns up to 4,000 tokens. Design the compaction policy for tool output: what is kept as facts, what is summarized, what is kept verbatim, and when compaction fires. Estimate input tokens per step before and after, with your assumptions labeled.

**E2.** The product team wants the assistant to greet users by name and mention today's date. Where do these values go in the rendered prompt, and why? What test would stop a later change from moving them into the stable prefix?

**E3.** A tenant administrator asks that the assistant "forget" a phone number a user typed three weeks ago. List every place the number may exist in this chapter's design, and describe the deletion procedure for each, including the summary.

**E4.** You must support a "read this 80-page contract and answer questions" feature. Its documents exceed the evidence cap many times over. Compare three designs: raising the cap, retrieval over the contract's sections, and a map-reduce summary. Name the failure mode each one risks and the evaluation that would choose between them.

### Practical exercises

**P1.** (about 90 min) Add a `tool_result` compactor: a function that takes a JSON tool result and a list of field paths the next step reads, and returns a trimmed `ContextItem` with a `metadata["full_ref"]` pointer to the original. Test that the required fields survive, that the token count drops, and that the original can be rehydrated from the reference.

**P2.** (about 60 min) Extend `ContextBuilder` with a `restate` option. When the rendered prompt exceeds a configurable token count, it appends a one-line constraint reminder from the system contract just before the request. Keep the stable prefix hash unchanged, and test that it is.

**P3.** (about 2 hours) Implement a leave-one-out attribution script. Given an evaluation set and a scripted `FakeLLM` handler that answers from specific source ids, it builds each request, removes each included evidence item in turn, and reports items whose removal never changes the answer.

**P4.** (about 90 min) Add a length sweep to `context/experiments/`: vary the number of distractors at a fixed needle position, report accuracy and mean prompt tokens per length, and test it with a simulated reader whose accuracy declines with length.

### Debugging exercises

**D1.** After a retrieval upgrade that returns longer chunks, follow-up questions such as "and for contractors?" are answered as if they were new questions. Single-turn evaluation scores improved. The manifests show `budget` as the most common drop reason, and `used_by_section.history` averages 90 tokens, down from 1,100. What is happening, and what is the smallest fix?

**D2.** A support conversation's draft reply quotes ticket INC-4812. The conversation was about INC-4821. The `state:facts` item in that request's manifest contains `ticket: INC-4821`. The summary item contains "ticket INC-4812 escalated". Compaction reports for that session show `accepted: true` on every run. How did the wrong id get past the guard, and what change would have stopped it?

**D3.** Cost per request rose 40 percent on Tuesday and TTFT p95 rose 35 percent. Traffic, prompt length, and answer length are unchanged. The provider's `cached_input_tokens` fell from about 70 percent of input to under 5 percent. The `context.prefix_hash` span attribute has a different value on every request since Tuesday's deploy. What do you look for in the deploy, and how do you prevent a repeat?

**D4.** Users report that the assistant sometimes forgets the answer it just gave. They ask "what was the second step again?", and the model replies that it has not listed any steps. It happens mostly on slow turns. The traces for one affected session show a request that wrote the assistant's answer as turn 7 and logged `state saved: version 12 -> 13`. A background compaction job for the same session logged `state loaded: version 12` before that request finished and `state saved: version 12 -> 13` two seconds after it. The session store is a key-value store written with a plain `put`. What happened, and what is the fix?
