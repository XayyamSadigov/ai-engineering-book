# Chapter 23 — Frameworks: Solutions

## Knowledge questions

**K1.**

| Framework concept | Primitive and chapter | One thing the framework version hides |
|---|---|---|
| LangChain retriever | Chapter 12 `Retriever` protocol over `HybridRetriever` / `Reranker` | Whether scores are returned, the default `k`, and whether metadata filters run before or after the vector search |
| LangGraph checkpointer | Chapter 17 `Checkpointer` (`save`, `latest`, `history`) plus `resume_from_checkpoint` | Whether the side effect inside the node running at crash time completed; the checkpoint has state but no idempotency record |
| LlamaIndex refine-mode synthesizer | Chapter 13 `EvidencePacker` + `GroundedGenerator` | The grounding prompt text (how the model is told to use evidence and whether it may answer from prior knowledge) and the one-call-per-chunk cost multiplier |
| DSPy optimizer | Chapter 4 prompt contract searched against Chapter 24 metrics | The compiled prompt text itself, which must be dumped and versioned to satisfy the registry rule |
| Agent SDK handoff | Chapter 22 supervisor-worker / router expressed as a tool call | Budget and trace boundaries between the agents; the SDK sees one conversation |

**K2.** A composed runnable's `with_retry` wraps the entire sequence, so one failed attempt re-runs
every step inside it: the prompt render, the model call, and the parser. If the parser failed on a
valid completion, the retry pays for a second model call that was not needed; if the model call
failed, the gateway may already have retried it, so attempts multiply across layers. Retries in this
book live at exactly one layer, Chapter 3's `ModelGateway` (`RetryPolicy`) for model calls and
Chapter 17's per-node `RetryPolicy` for workflow steps, each with a budget and each recording attempts
in the trace (Chapter 29 explains the one-layer rule).

**K3.** On resume, LangGraph re-executes the node that called `interrupt` from its first line, because
the interrupt is implemented by raising out of the node and the node is re-entered with the human's
value available. Any side effect before the call runs a second time. Two fixes: move the side effect
to a node after the interrupt (the pause node does nothing but pause, as `pause_before=True` does in
Chapter 17), or make the side effect idempotent with Chapter 16's `IdempotencyStore` so the second
execution is a no-op keyed on run ID plus step.

**K4.** It changes the prompt contents: the few-shot demonstrations and, in instruction-search
optimizers, the instruction wording. It does not change model weights (it is not fine-tuning) and it
does not change the signature, which is the stable spec. The four requirements: a metric that is a
faithful proxy for the real objective; enough labeled examples (dozens for demonstration bootstrapping,
hundreds for instruction search) so the confidence interval is narrower than the improvement; a frozen
holdout never seen by the optimizer, to avoid selection leakage; and a re-run whenever the model
changes, because the compiled prompt is tuned to one model.

**K5.** Adds: persistence (checkpointers, sessions), streaming (token and step events through the
pipeline), tracing (spans for every framework object, with a UI), ecosystem (integrations). Hides:
defaults, revealed by a configuration dump compared against the values your settings declare, and by
the chunk size, `k`, or `max_tokens` visible on spans; retries, revealed by model calls per request
exceeding one during a provider incident while the code shows a single policy, or by duplicated
model-call spans with no attempt link; prompt text, revealed by the prompt hash on model-call spans
differing from the registry version's hash, or by prompt text in traces that does not appear in the
registry.

## Engineering questions

**E1.** A defensible scoring, weights in parentheses (1 to 3):

| Criterion (weight) | Plain primitives (Ch 16, 17, 19) | Framework with hosted checkpointer |
|---|---|---|
| State transparency (3) | 2: one pydantic state you define | 1: TypedDict with reducers; readable but merge semantics live in annotations |
| Persistence (3) | 1: `Checkpointer` protocol exists, Postgres backend is yours to write and operate | 2: production backend exercised by many users |
| Streaming (1) | 1: `StepRecord` (Chapter 17) and the `agentkit` event types such as `ModelDecision` and `ToolResult` (Chapter 19) exist; UI plumbing is yours | 2: multiplexed modes out of the box |
| Tracing (2) | 2: your `Tracer` and Chapter 31 schema | 1: vendor attribute names unless mapped |
| Retry semantics (2) | 2: one layer, budgeted, attempts in trace | 1: configurable but layered with the model wrapper |
| Testability (2) | 2: fakes for every port | 1: fake model exists; checkpointer tests need the backend or an in-memory saver |
| Deployment (2) | 2: plain library | 1 to 2 depending on whether the hosted saver is required |
| Ecosystem (1) | 1 | 2 |
| Upgrade churn (2) | 2 | 1 |
| Lock-in (2) | 2 | 1 unless adapters are already in place |

Weighted totals land close, with the framework ahead only if persistence is weighted 3 and the team
would not otherwise build the Postgres saver. The two highest weights are persistence (a pod restart
mid-run is the stated requirement, and resuming without duplicating `create_ticket` is the hard part)
and state transparency (approval requires a human to read the state at the pause, and an audit
requires diffing state across checkpoints). A good answer notes that the approval pause itself is
Chapter 17's `pause_before` either way, that idempotency for `create_ticket` is Chapter 16's
`IdempotencyStore` either way, and that the framework should enter behind the `Workflow` port from E3.

**E2.** Diagnosis path: (1) Pull the exact prompt the synthesizer sent from the trace and diff it
against the registry; the synthesizer's default template is almost certainly not in the registry and
almost certainly does not forbid answering from prior knowledge. (2) Record fixtures for ten failing
cases with `RecordingLLM` so the fix can be verified offline. (3) Run Chapter 14's stage isolation:
retrieval metrics on the same cases to confirm retrieval is good (the evidence contains the answer)
and the failure is in generation. (4) Check the synthesizer mode: refine mode asks the model to
"improve" a prior answer, which invites additions not in the current chunk. The fix comes from Chapter
13's grounded answer contract: cite-or-abstain prompt text (data is not instructions, answer only from
evidence, reply with an abstention token otherwise) passed explicitly to the synthesizer, versioned in
the registry, plus `CitationValidator` on the output. Confirm with the fixtures and with Chapter 24's
faithfulness metric on the eval set.

**E3.** A port shaped so both engines can implement it:

```python
class WorkflowStatus(str, Enum): COMPLETED = "completed"; PAUSED = "paused"; FAILED = "failed"

class ResumeHandle(BaseModel):
    run_id: str
    node: str            # the node waiting for a decision
    binding: str | None = None   # opaque token tying the decision to the state the human saw

class WorkflowResult(BaseModel, Generic[S]):
    run_id: str
    status: WorkflowStatus
    state: S             # the domain's pydantic state, never a framework dict
    handle: ResumeHandle | None = None
    error: str | None = None

class Workflow(Protocol[S]):
    def start(self, state: S, run_id: str | None = None) -> WorkflowResult[S]: ...
    def resume(self, handle: ResumeHandle, decision: Any) -> WorkflowResult[S]: ...
    def recover(self, run_id: str) -> WorkflowResult[S]: ...
    def history(self, run_id: str) -> list[tuple[str, S]]: ...
```

The state type crossing the boundary is the domain's pydantic model. For Chapter 17's `Graph`, the
adapter is thin: `start` calls `run`, `resume` calls `resume`, `recover` calls
`resume_from_checkpoint`, `history` calls `replay`, and `ResumeHandle` maps one to one, with `binding` carrying Chapter 17's paused `seq` and `state_hash` so that a resume against a changed state is refused by either engine. For a framework that has no such check, the adapter computes the hash itself at pause time and compares it before resuming. For a compiled
state graph, the adapter converts the pydantic state to the graph's `TypedDict` on the way in and back
on the way out (`model_validate` on the returned dict), maps `run_id` to the framework's `thread_id`
in the invocation config, detects a pause by the presence of a pending interrupt in the returned
state snapshot and builds a `ResumeHandle(run_id=thread_id, node=<interrupted node>)`, implements
`resume` by invoking with a resume command carrying `decision`, and implements `history` by iterating
the framework's state history and converting each snapshot. Acceptance: one contract test suite runs
against both adapters (see P3).

**E4.** The adapter implements Chapter 31's `Tracer` protocol and forwards to the vendor SDK. Inside
`span()`: rename attributes to the book's schema (`prompt.version`, `model.name`, `usage.input_tokens`,
`usage.output_tokens`, `cost.usd`, `cache.hit`, `evidence.ids`, `policy.result`, `eval.score`); apply
redaction before export (a configured list of regexes for emails, employee IDs, ticket bodies; prompt
and completion text stored as a hash plus the first N redacted characters unless the request carries
an explicit debug flag); apply sampling by request class (100 percent of errors and policy denials,
a configurable fraction of successes, always keep the span tree whole for a sampled request). Never
drop: request ID and trace ID, tenant, prompt version, model and provider, token counts, cost, finish
reason, error class, tool name and policy decision for every tool call, evidence IDs for every
retrieval. These are the join keys and the fields Chapter 31's debugging playbook and Chapter 30's cost
accounting depend on; text can be sampled or hashed, identifiers and numbers cannot.

## Practical exercises

**P1.** Expected implementation: add a `stream_safe: bool` attribute to `Runnable` (default `False`;
`Lambda` accepts it as a constructor flag). `Sequence.stream` iterates the steps: it calls `stream` on
the first step to get a chunk iterator; for each subsequent step, if `stream_safe` it maps the step
over the chunk iterator lazily, otherwise it joins the chunks (string concatenation for `str`, list
for others) and calls `invoke`, after which the iterator is a single-element iterator. Acceptance
criteria: a fake model step whose `stream` yields `["{\"a\"", ": 1", "}"]` passed through a
`stream_safe` upper-casing step yields three upper-cased chunks; the same stream followed by a JSON
parsing step yields exactly one chunk equal to `{"A": 1}`; the `RunLog` records the parser once and
the upper-casing step three times (or once with a chunk count, as long as the test documents the
choice); `invoke` behavior is unchanged and existing tests pass.

**P2.** Expected implementation: `ChainOfThought(Predict)` whose constructor derives a new
`Signature` with `Field("reasoning", "think step by step before answering")` inserted at the front of
`outputs`, keeps a reference to the original signature for the caller, and strips `reasoning` from
the returned dict unless asked to keep it. Acceptance: the original `TRIAGE` object is unchanged
(assert identity and equality); a fake model that only answers "hardware" when the prompt ends with
`reasoning:` shows the dev score moving with the module and not with the signature; `BootstrapFewShot`
compiles a `ChainOfThought` module without modification, which demonstrates that optimizer and module
are orthogonal. A good solution also notes that the fake model cannot show a real reasoning benefit
and that the honest measurement is Chapter 24's held-out evaluation with a real model.

**P3.** Expected implementation: `GraphWorkflowAdapter(Graph)` and `StandInGraphAdapter(StandInGraph)`
both implementing the `Workflow` port from E3. The stand-in is a minimal class with `invoke(state,
config)`, `get_state(config)`, and a `pending_interrupt` field that mimics a compiled state graph's
behavior without the dependency. The contract test suite is parametrized over a fixture that yields
each adapter and asserts: a three-node run completes with the expected final state; a run with a pause
node returns `PAUSED` with a handle naming the node; `resume(handle, "approve")` completes and the
decision is visible in state; `resume` with a stale handle raises; `recover(run_id)` after a simulated
crash between nodes continues from the last checkpoint and a counting side-effect node executed
exactly once; `history` returns one entry per completed node. Acceptance: the same test file passes
for both adapters with no adapter-specific branches.

**P4.** Expected implementation: `RecordingRetriever(inner: Retriever, path)` writes
`{hash(query, k): [passage dicts]}`; `ReplayRetriever(path)` reads it and raises `LookupError` on a
miss with a message naming query and `k`. Record a fixture over `FrameworkRetrieverAdapter(
FrameworkRetrieverLike(DOCS))` with `k=4`. Then change the adapter or stand-in so the effective `k`
differs (for example the stand-in applies its own `[:2]` cap, mimicking a framework default), and
show the replay test fails because the key includes `k` and the stored passages no longer match the
live call. Acceptance: round trip equality on the recorded query; a `LookupError` or an assertion
failure, by design, when the default changes; the test's failure message tells the reader what
changed.

## Debugging exercises

**D1.** Root cause: the dependency upgrade changed prompt text inside the framework (a prebuilt
template, a structured-output instruction, or demonstration formatting) while the registry version,
which only covers the text your code owns, stayed the same. Telemetry: the `prompt.hash` attribute on
model-call spans differs between the two weeks while `prompt.version` is identical; diffing one
pre-upgrade and one post-upgrade prompt from the trace shows the changed framework-owned section. What
should have caught it: the recorded-fixture replay suite run on dependency updates (replay fails with
`LookupError` on the changed prompt), and Chapter 25's CI eval gate on the triage dataset, which would
have blocked the deploy at 0.84 against a 0.91 baseline.

**D2.** Nine is three layers of three attempts in the two outer layers multiplying partially: the
framework model wrapper retries three times per call, the `ModelGateway` it wraps (or is wrapped by)
retries three times per attempt, and the node policy retries the node once more on the resulting
error. Nine is consistent with wrapper 3 times gateway 3 with the node retry disabled or exhausted by
timeout, or other combinations; the exact product matters less than the fact that three independent
policies exist. Only the `ModelGateway` should keep retries: it classifies errors (`retryable`,
`retry_after_s`), honors a budget, records attempts in the trace, and is the single place cost is
accounted. Set the framework wrapper's retries to zero and the node's `max_attempts` to one for
model-call nodes (node-level retries are for tool steps with their own error classes). The test: a
fake client that raises `ProviderUnavailableError` on every call, wired through the full stack,
asserting the number of `complete` calls equals the gateway policy's `max_attempts`.

**D3.** Mechanism: interrupt double-execution. The `create_ticket` call sits in the same node that
pauses for approval (or before an `interrupt()` call); on resume the framework re-executes the node
from the top, so the call runs again with the same run ID. Fix: move `create_ticket` to the node after
the approval (the pause node only pauses), and in addition give `create_ticket` an idempotency key of
run ID plus step name through Chapter 16's `IdempotencyStore` so a replayed node is a no-op. Test: a
counting fake `create_ticket` tool in a graph with a pause before it; run, assert count is zero at
`PAUSED`, resume with approval, assert count is one; separately, resume twice with the same handle and
assert the second attempt raises or is a no-op. The telemetry that reveals it in production: two tool
audit entries with the same run ID and tool name, one timestamped before a `paused` checkpoint and one
after the resume.
