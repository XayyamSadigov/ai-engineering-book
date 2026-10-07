# Exercises — Chapter 23 — Frameworks

Solutions: `../solutions/ch23-solutions.md`


**Start here:** K1, K5, E1, P4, D2 (about 4 hours). The rest go deeper.

### Knowledge questions

**K1.** For each of the following, name the chapter and primitive it maps to and one thing the
framework version hides: a LangChain retriever, a LangGraph checkpointer, a LlamaIndex response
synthesizer in refine mode, a DSPy optimizer, an agent SDK handoff.

**K2.** Explain why `with_retry` on a composed runnable can cost more than a retry inside the model
client, and where in this book retries are supposed to live.

**K3.** A LangGraph node performs a side effect and then calls `interrupt`. Describe what happens on
resume and two ways to make it safe.

**K4.** "Prompt optimization against a metric" in DSPy changes what, and does not change what? List
the four evaluation requirements the chapter attaches to it.

**K5.** State the four things a framework typically adds and the three it typically hides. For each
hidden item, name the telemetry that would reveal it.

**K6.** A vendor offers a hosted agent runtime: you upload tools and instructions, and it runs the
loop, stores sessions, and executes code in its sandbox. Which rows of the primitive ledger does it
absorb, which selection criterion is it most likely to score 0 on, and which three responsibilities
stay with you regardless?

### Engineering questions

**E1.** Northwind's HR knowledge assistant (Project 3) must never show a passage to a user outside
its ACL, must cite every claim, and must support Chapter 14's stage-isolated evaluation. Using the
scoring table, score three candidates: plain `ragkit` primitives (Chapters 10 to 15), a retrieval
framework with a query engine, and a hosted agent runtime with built-in file search. State your
weights, justify the two criteria you weighted highest, and name the one finding that would change
your conclusion.

**E2.** A team uses a framework's query engine for the HR knowledge base. Retrieval quality is good
but answers sometimes include facts not in the evidence. Propose a diagnosis path using this chapter's
tools (extract hidden text, fixtures, stage isolation) and say which chapter's contract the fix comes
from.

**E3.** Design the `Workflow` port for Northwind Assist so that both Chapter 17's `Graph` and a
compiled LangGraph can implement it. Specify the methods, the state type that crosses the boundary,
and how a `ResumeHandle` is represented in both.

**E4.** Your observability vendor's SDK auto-instruments the framework and exports prompts verbatim.
Specify the adapter that maps its spans to Chapter 31's attribute schema with redaction and sampling,
and say which attributes must never be dropped.

### Practical exercises

**P1.** (about 90 min) Extend `runnable.py` with a stream-aware `Sequence.stream` that propagates chunks through
steps that declare themselves stream-safe and joins before steps that do not. Add tests showing a
model-like step streaming three chunks through an upper-casing step and being joined before a JSON
parsing step.

**P2.** (about 60 min) Extend `signature.py` with a `ChainOfThought` module that adds a `reasoning` output field
before the first declared output, and show with a fake model and the existing optimizer whether it
improves the dev score on the triage signature. Keep the signature object unchanged.

**P3.** (about 2 hours) Write an adapter that makes Chapter 17's `Graph` implement the `Workflow` port from E3, and a
second adapter over a hand-written stand-in for a compiled state graph (do not install the framework).
Write one contract test suite that both adapters pass, including pause and resume.

**P4.** (about 60 min) Build a `RecordingRetriever` / `ReplayRetriever` pair in the style of `ports.py`, record a
fixture over the `FrameworkRetrieverLike` stand-in, then change the stand-in's default `k` and show
the replay test detecting the change.

### Debugging exercises

**D1.** After a dependency upgrade, Northwind's ticket-triage eval accuracy drops from 0.91 to 0.84
(illustrative numbers) with no change in the repository. The prompt registry version on model-call
spans is unchanged, but the prompt hash attribute differs from last week's traces. No retries are
visible. Diagnose, name the telemetry that confirms it, and say what should have caught it before
deploy.

**D2.** During a provider incident, the cost dashboard shows nine model calls per failed request. The
code configures a `ModelGateway` with `max_attempts=3`; the orchestration framework's node has its
default retry policy of three attempts. Explain the nine, identify
which layer should keep retries, and state the test that would have shown this.

**D3.** An approval workflow built on a graph framework occasionally creates two tickets for one
incident. The audit log shows both `create_ticket` calls carry the same run ID; one precedes a pause
checkpoint and one follows the resume. Name the mechanism, the fix, and the test.
