# Chapter 23 — Frameworks

LLM frameworks package the primitives you built in Chapters 3 to 19 behind their own vocabulary, and
they make choices on your behalf. This chapter teaches you to see each framework concept as a
primitive you already own, so you can adopt a framework for what it adds without inheriting what it
hides.

**You will be able to:**
- Translate any framework concept into a primitive from this book: a prompt template is Chapter 4's
  registry entry, a runnable chain is function composition with a log, a state graph is Chapter
  17's `Graph`, a query engine is Chapter 10's pipeline, a DSPy optimizer is a metric-driven search
  over prompt contents.
- Find the defaults, retries, and prompt text a framework hides, and make each one explicit.
- Score a framework, a provider agent SDK, and plain primitives against ten selection criteria with
  your own weights.
- Keep domain logic framework-independent with ports and adapters.
- Test a framework-backed component from recorded fixtures and migrate off a framework without
  rewriting the domain.

**Prerequisites:** Chapters 3 and 4 (gateway, prompt registry), 10 to 13 (the RAG pipeline), 16, 17,
and 19 (tool executor, workflow engine, agent loop). | **Code:** `book/projects/examples/ch23/`
(run: `cd book/projects/examples/ch23 && pytest -q`) | **Builds:** a composable pipeline, a typed
prompt specification with a metric-driven optimizer, and a ports-and-adapters layout with
record-and-replay LLM fixtures.

## Why this matters

Frameworks are where most teams start and where many get stuck. The start is reasonable: a chat
model wrapper, a prompt template, a vector store integration, and an agent loop in an afternoon. The
getting stuck has a recognizable shape. Six months in, nobody can say which prompt text production
actually sends, because the framework assembled it from three templates and a default system message.
A retry policy nobody configured has been tripling the cost of failed calls. An upgrade changed how
tool results are formatted and quality dropped with no change in the repository. An engineer wants to
add a human-approval step and finds the agent abstraction has no seam for it. None of these are
failures of the framework. They are failures of understanding what the framework was doing on the
team's behalf.

The principle is blunt: agent frameworks are useful only after you understand the underlying
primitives, because framework APIs change and the primitives do not. This book took
that literally. Between Chapter 3 and Chapter 19 you wrote, with tests, every component a framework
offers: client and gateway, prompt registry, structured output, chunkers, retrieval, grounded
generation, tool registry, workflow engine, MCP, and agent runtime. The purpose of this chapter is not
to teach you the frameworks from scratch. It is to show that you already know them, and to give you the
discipline to use one without losing sight of what it hides.

> **Mental model:** Reliability is engineered around the model, not expected from it. A framework is
> someone else's engineering around the model. Adopt it where their choices match yours, and make
> every hidden choice visible before it reaches production.

## Mental model

Treat every framework concept with a fixed three-question discipline:

1. **Which primitive is this?** Name the thing you built, with its chapter, and picture the
   plain-Python version. If you cannot, you are not ready to use the framework's version.
2. **How does the framework express it?** Learn the vocabulary: runnable, node, query engine,
   signature, handoff. Vocabulary is cheap and is where most of the perceived complexity lives.
3. **What does it add, and what does it hide?** Added value is almost always one of four things:
   persistence, streaming, tracing, or ecosystem. Hidden cost is almost always one of three: defaults
   you did not choose, retries you did not configure, and prompt text you did not write.

The three questions turn framework evaluation into an engineering review and give you a migration
plan for free: a component whose primitive you can name is a component you can replace.

A second idea organizes the architecture advice. Your domain owns its interfaces: Northwind Assist's
domain defines `Retriever`, `LLMClient`, and `Tool` as Protocols with exactly the methods it needs.
Framework objects never appear in the domain; they live in adapters, one class per framework object,
that translate the framework's vocabulary into yours. Chapter 32 develops ports and adapters for the
whole application; here we apply it to the one boundary between you and the framework.

## Core concepts

### The primitive ledger

Every row is something you have implemented and tested. Keep the table open while reading the
framework sections; each is a walk through part of it.

| Primitive | Chapter | Name in this book | Framework words for it |
|---|---|---|---|
| Provider-neutral model call with retries, fallback, cache | 3 | `LLMClient`, `ModelGateway` | chat model, `with_retry`, `with_fallbacks` |
| Versioned prompt with rendering and tests | 4 | `PromptRegistry` | prompt template, signature |
| Context assembly under a token budget | 5 | `ContextBuilder` | memory, session |
| Schema-validated output with repair loop | 6 | `complete_structured` | output parser, `with_structured_output` |
| Splitting documents into retrievable units | 11 | `Chunker` | text splitter, node parser |
| Dense, lexical, hybrid retrieval and reranking | 12 | `HybridRetriever`, `Reranker` | retriever, index, postprocessor |
| Packing evidence and generating a cited answer | 13 | `EvidencePacker`, `GroundedGenerator` | response synthesizer, query engine |
| Tool schema, policy, idempotency, approval | 16 | `ToolRegistry`, `PolicyEngine`, `ToolExecutor` | `@tool`, `bind_tools`, function tool |
| Typed state, nodes, routers, checkpoints, pause | 17 | `Graph`, `Checkpointer`, `ResumeHandle` | `StateGraph`, checkpointer, interrupt |
| Tool discovery and invocation over a protocol | 18 | MCP client and server | MCP SDK |
| Loop with events, budgets, termination, replay | 19 | `AgentRuntime` | agent, runner, session |
| Metrics, datasets, judges, statistics | 24 | `evalkit` | evaluator, metric, optimizer objective |
| Spans with prompt, tokens, cost, evidence IDs | 31 | `Tracer`, `Span`, `AITracer` | tracing SDK, callbacks |

The framework sections below describe each library's API and defaults at the time of writing.
Names, defaults, and module layout change between releases, so treat each specific claim as
something to confirm against current documentation and source code, which is the habit this chapter
teaches.

### The landscape as of 2026

Product names churn faster than categories. Learn the categories: each one absorbs a predictable
set of ledger rows, adds a predictable kind of value, and hides a predictable kind of choice. When a
new library appears, place it in a row first, then read its documentation with that row's "hides"
column as your checklist.

| Category | Examples (as of 2026) | Ledger rows it absorbs | What it adds | What it hides |
|---|---|---|---|---|
| Provider agent SDKs | for example OpenAI Agents SDK, Claude Agent SDK, Google Agent Development Kit | Ch 19 loop, Ch 22 handoffs, Ch 5 and 21 sessions, Ch 31 tracing | a short path to a working tool loop, provider-hosted tools, tracing into the provider's console | policy and approval layer, budgets, compaction policy, coupling to one provider's features |
| Graph orchestration libraries | for example LangGraph | Ch 17 graph, checkpoints, pause; parts of Ch 19 | persistence backends, interrupts, multiplexed streaming | reducer semantics, checkpoint granularity, node re-execution on resume |
| Composition and integration frameworks | for example LangChain, LlamaIndex | Ch 3 to 13: clients, templates, parsers, chunkers, retrievers, synthesizers | integrations with many stores, loaders, and providers | defaults (`k`, chunk size, retries) and shipped prompt text |
| Typed agent frameworks | for example Pydantic AI | Ch 6 structured output, Ch 16 tool schemas, Ch 19 loop | output and tool schemas derived from type hints, test models | how many times invalid output is re-asked, and the text of the re-ask |
| Multi-agent and role frameworks | for example CrewAI, Microsoft Agent Framework | Ch 22 topologies, Ch 17 workflows | role, task, and conversation abstractions | delegation prompts, per-agent budgets, trace boundaries between agents |
| Prompt programming and optimization | for example DSPy | Ch 4 prompt contracts, Ch 24 metrics | metric-driven search over prompt contents | the compiled prompt text |
| Hosted agent runtimes | provider-managed agent services and cloud agent platforms | Ch 19 loop, Ch 16 sandboxed execution, Ch 21 memory, Ch 38 durable state | hosting, scaling, sandboxes, stored sessions, no servers to run | where state and data live, the execution log, network egress |
| Durable execution engines | for example Temporal-style workflow engines | Ch 38 log, lease, timer, signal | crash-proof long-running runs | determinism rules your code must follow (see Chapter 38) |

Three patterns in the table matter more than any row. First, the further down the stack a product
reaches (from a library you import to a runtime you call), the more it adds and the more it hides:
a hosted runtime removes operations work and also removes your event log from your database.
Second, provider-hosted tools (web search, code execution, remote MCP connectors run by the
provider) execute outside your `ToolExecutor`; Chapter 16 covers what that means for policy, audit,
and data egress. Third, every category leaves the same three rows to you: the tool policy, the
budget, and the evaluation. No framework in any row decides what your agent is allowed to do or
whether its answers are good.

**Freshness note.** As of 2026 this market consolidates and renames often: libraries merge, SDKs
gain and drop features between minor releases, and hosted runtimes appear under new names. The
examples column will age; the "absorbs" and "hides" columns will not. Before adopting anything, read
the current documentation and source with the three questions from the mental model.

### LangChain

LangChain is a broad application framework: model wrappers, prompt templates, output parsers,
retrievers, tools, and a composition layer, LCEL (LangChain Expression Language), that pipes them
together with `|`. Its value is ecosystem breadth. Its typical cost is what this chapter calls
architecture by abstraction: stacking components without understanding the data flow between them.

**Chat models.** The primitive is Chapter 3's `LLMClient` behind a `ModelGateway`:

```python
# reminder: Chapter 3 primitive
req = CompletionRequest(messages=[Message.system(sys_text), Message.user(user_text)],
                        temperature=0.0, max_tokens=400)
completion = gateway.complete(req)   # retries, fallback chain, cost accounting happen here
completion.text; completion.usage.input_tokens
```

The framework expresses it as a chat-model object:

```python
# API shape at the time of writing; check current docs
model = init_chat_model("provider:model-name", temperature=0)
ai_message = model.invoke([("system", sys_text), ("human", user_text)])
ai_message.content; ai_message.usage_metadata
```

Added: many providers behind one interface, async and batch variants, normalized usage metadata.
Hidden: default retry count and backoff, default timeout, how a content-filter error is mapped, and
in some integrations a default `max_tokens`. Each is a number you set deliberately in Chapter 3; read
the wrapper's source code for those four values and set them explicitly.

**Prompt templates.** The primitive is Chapter 4's registry entry: a named, versioned template with
declared variables, rendered with escaping, covered by golden tests, its version on every trace.
`ChatPromptTemplate.from_messages([...])` with `{variable}` placeholders expresses the rendering half.
Added: composition of message lists and a placeholder for conversation history. Hidden: nothing about
the text, but there is no notion of a version, so prompt text lives wherever the Python object is
constructed, scattered across modules and inside prebuilt chains. Chapter 4's rule stands: every
prompt that reaches a model has a name and a version in your registry, whoever renders it.

**Output parsers and structured output.** The primitive is Chapter 6's `complete_structured`: send a
JSON Schema via `response_schema`, validate with pydantic, re-ask with the validation error, raise
`MalformedResponseError` when attempts are exhausted. `model.with_structured_output(Schema)`
expresses the happy path. Added: a provider-specific strategy (native structured output, tool-call
coercion, or JSON mode) chosen for you. Hidden: which strategy, whether any repair happens on failure
(typically it raises), and what happens to refusal and truncation. Output parsers that regex a format
out of free text are what Chapter 6 told you not to do when the provider can constrain output.

**Retrievers.** The primitive is Chapter 12's `Retriever` protocol, `retrieve(RetrievalQuery) ->
RetrievalResult`, whose result carries scored hits per stage and a trace, so fusion and reranking
are visible. A framework retriever is any object
with `invoke(query) -> list[Document]`, a `Document` having `page_content` and `metadata`;
`vectorstore.as_retriever(search_kwargs={"k": 4})` makes one from a store. Added: a large catalog of
integrations behind one call. Hidden: the fusion algorithm for hybrid search, the default `k`, whether
metadata filters apply before or after the vector search (which changes recall), and whether scores
are returned at all. Chapter 14's stage isolation needs those scores and that boundary; an integration
that hides them has cost you the most important debugging capability in RAG.

**Runnables and LCEL.** The composition layer is the part most worth understanding because it is the
smallest. A runnable has `invoke`, `batch`, `stream`, async twins, the `|` operator for sequences,
and a dict literal for fan-out. The primitive is function composition with a uniform calling
convention:

```python
# reminder: this is all `prompt | model | parser` means
def chain(x):
    return parser(model(prompt(x)))
```

```python
# API shape at the time of writing; check current docs
chain = prompt | model | StrOutputParser()
chain.invoke({"question": q}); chain.batch([...]); chain.stream({"question": q})
chain = chain.with_retry(stop_after_attempt=3).with_fallbacks([other_chain])
```

Added: every step gets a span with inputs and outputs, `stream` propagates token
deltas through the pipeline, `batch` parallelizes, async comes free. Hidden: `with_retry` retries the
whole wrapped runnable, so a chain re-renders the prompt and re-pays for the model call even when a
cheap parser failed; the retried exception classes default to everything; and attempts are invisible
in cost accounting unless the tracer records them. The `runnable.py` example builds the same
operators in about a hundred lines so you can see exactly where attempts are logged.

**Tools.** The primitive is Chapter 16: a `ToolSpec`, a registry, argument validation outside the
model, a policy that classifies side effects and requires approval for irreversible ones,
idempotency keys, timeouts, result truncation. The `@tool` decorator derives the schema from a
signature and docstring; `model.bind_tools(tools)` attaches the specs to the request. Added: schema
generation and a convenient loop in prebuilt agents. Hidden: everything after the model proposes a
call. The decorator has no concept of side-effect class, approval, idempotency, or permission, and a
prebuilt agent that executes directly from the proposal has removed the "code authorizes" half of
the book's sixth mental model ("the model proposes, code authorizes," Chapter 1). Use the schema
generation; keep execution in your `ToolExecutor`, which owns validation, policy, approval, and
idempotency for every tool in the `ToolRegistry`.

### LangGraph

LangGraph models a workflow as a graph over typed state: nodes read and write state, edges choose
the next node, a checkpointer persists state after every step, interrupts pause for a human. That is
the docstring of Chapter 17's `workflow_engine.py`, which does the same in about 250 lines; Chapter
19's `AgentRuntime` adds the event log, budgets, and termination conditions that LangGraph largely
leaves to you.

**StateGraph and nodes.** The primitive is `Graph(state_type)` with `add_node(name, fn)` over a
pydantic state:

```python
# reminder: Chapter 17 primitive
class TriageState(BaseModel):
    ticket: str; category: str | None = None; draft: str | None = None; approved: bool = False

g = Graph(TriageState, checkpointer=InMemoryCheckpointer())
g.add_node("classify", classify).add_node("draft", draft)
g.add_node("approve", lambda s: s, pause_before=True, on_decision=apply_decision)
g.add_edge("classify", "draft").add_edge("draft", "approve").add_edge("approve", END)
```

LangGraph uses a `TypedDict` state whose fields may carry reducers (functions that merge a node's
partial update into the existing value), and nodes return partial updates:

```python
# API shape at the time of writing; check current docs
class State(TypedDict):
    ticket: str
    messages: Annotated[list, add_messages]   # reducer: append, dedupe by id
    category: str | None

builder = StateGraph(State)
builder.add_node("classify", classify)        # returns {"category": ...}
builder.add_edge(START, "classify")
```

Added: reducers make parallel branches that write the same key well-defined, and partial updates make
nodes shorter. Hidden: merge semantics live in the annotation, so a reader of the node cannot tell
whether a returned list replaces or appends. Document each reducer where you declare the state, and
define the domain state schema yourself: the library should orchestrate your
design, not become it.

**Conditional edges.** The primitive is `add_router(src, fn)` with `fn: (S) -> str`. LangGraph's
`add_conditional_edges("node", router, {"label": "target"})` is the same with an explicit map the
framework uses to draw the graph. Both are pure functions of state, which is what makes them
unit-testable without a model.

**Checkpointers.** The primitive is Chapter 17's `Checkpointer` protocol (`save`, `latest`,
`history`), called after every node, and the two resume paths built on it: `resume(handle,
decision)` after a human pause and `resume_from_checkpoint(run_id)` after a crash. LangGraph
attaches the checkpointer at `compile(checkpointer=...)` and keys it by a `thread_id` in the
invocation config. Added: production backends (for example PostgreSQL and Redis savers), state
history, and forking a run from an earlier checkpoint. Hidden: the granularity (per node, or per
super-step: one round of parallel nodes), the serialization of your state, and the fact that a
checkpoint does not know whether the side effect inside the node running at crash time completed.
Chapter 19's rule applies: the event log records tool request, approval, result, and idempotency
key, so on restart the harness determines whether an action completed instead of asking the model. A
checkpointer gives you the state; Chapter 16's `IdempotencyStore` gives you the answer.

**Interrupts and human-in-the-loop.** The primitive is `pause_before=True` on a node, a
`ResumeHandle` returned to the caller, and `on_decision` to fold the human's answer into state.
LangGraph expresses it declaratively (`interrupt_before=["approve"]` at compile) or by calling
`interrupt(payload)` inside a node; the caller resumes with a command carrying the human's value.
Added: the pause is durable because it is a checkpoint. Hidden: on resume, the node that called
`interrupt` re-executes from its beginning, so code before the call runs twice. Side effects before
an interrupt must be idempotent or moved to an earlier node: Chapter 17's double-execution hazard,
made easy to forget. Chapter 17's `step_key()` shows the key to use, one that stays the same when
the same visit to a node is re-executed.

**Streaming of state.** The primitive is Chapter 17's per-node `StepRecord` trace and Chapter 19's
per-step event stream, surfaced as server-sent events. `graph.stream(input, config, stream_mode=...)`
yields full state after each node, per-node updates, or token deltas. Added: one call that multiplexes
node-level and token-level events. Hidden: little, but choose the mode early; full-state snapshots are
expensive for large states, and token deltas alone cannot show which node is running.

### LlamaIndex

LlamaIndex is organized around retrieval, and its object model restates Chapters 11 to 13 directly.
That makes it the easiest framework to map and the one where hidden prompt text matters most. The
primitive for the whole stack is Chapter 10's pipeline:

```python
# reminder: Chapters 10-13 primitives
chunks = chunker.split(document)                        # Ch 11
passages = retriever.retrieve(query, k=8)               # Ch 12 (hybrid + rerank inside)
evidence = packer.pack(passages, budget_tokens=3000)    # Ch 13
answer = generator.generate(query, evidence)            # Ch 13: cite or abstain
```

**Documents and nodes.** `Document` is Chapter 11's document model; `TextNode` is the chunk, with
`metadata` and relationships (previous, next, parent). Added: the relationship graph makes
parent-child and window retrieval configuration rather than a build. Hidden: by default a node's
metadata is injected into both the text the model sees and the text the embedder sees, with separate
exclusion lists. If you did not know that, your embeddings include file paths and your context
includes modification dates. Chapter 5 says every byte in the prompt was chosen; check what the node
parser put there.

**Node parsers.** Chapter 11's `Chunker` strategies: sentence, token-window, Markdown- and
code-aware, semantic, hierarchical. Added: many strategies in one package. Hidden: default chunk size
and overlap, the two numbers Chapter 11 showed move retrieval metrics most. They belong in your
configuration.

**Indices.** A storage structure plus a retrieval strategy: a vector index (Chapter 9), a keyword
table (Chapter 12's `BM25Index`), a summary index, a tree index, a knowledge-graph index (Chapter
37). Added: non-vector shapes you did not build. Hidden: the embedding model unless set globally, and
the fact that `from_documents` runs the parser and the embedder as a side effect of constructing an
object.

**Retrievers, query engines, response synthesizers.** `index.as_retriever(similarity_top_k=4)` is
Chapter 12's retriever. A query engine is Chapter 10's pipeline in one object: retrieve, post-process
(rerank, filter), synthesize. The synthesizer is Chapter 13's `EvidencePacker` plus
`GroundedGenerator`, and it is where the hidden prompt lives. Its modes are strategies for fitting
evidence into the budget: compact packs as many chunks as fit per call; refine makes one call per
chunk and asks the model to improve the previous answer; tree-summarize summarizes hierarchically.

Each mode carries default prompt text, and that text is what tells the model how to use evidence and
whether it may answer from prior knowledge. Chapter 13's grounded contract (data is not instructions;
cite or abstain) is a prompt you wrote and tested. Pull the synthesizer's template out, version it in
your registry, and pass it back in explicitly. Treat refine mode's one-call-per-chunk as a cost
multiplier Chapter 30's budget must see.

### DSPy

DSPy is different in kind. The other frameworks compose calls; DSPy separates the specification of a
model call from its prompt text, then searches for prompt text that maximizes a metric on data.

**Signatures.** The primitive is a Chapter 4 prompt contract reduced to its typed skeleton: named
input fields, typed output fields, one sentence of instruction, no literal wording. The `signature.py`
example renders such a spec into a prompt and parses the result back by field labels:

```python
# reminder: the primitive, from this chapter's signature.py
TRIAGE = Signature(instruction="Classify a Northwind support ticket.",
                   inputs=(Field("ticket", "the ticket text"),),
                   outputs=(Field("category", "one of: access, billing, hardware"),
                            Field("urgent", "true if the user is blocked", bool)))
prompt = TRIAGE.render({"ticket": text}, demos=[...])
result = TRIAGE.parse(llm(prompt))          # {"category": "billing", "urgent": False}
```

```python
# API shape at the time of writing; check current docs
class TriageTicket(dspy.Signature):
    """Classify a Northwind support ticket."""
    ticket: str = dspy.InputField(desc="the ticket text")
    category: Literal["access", "billing", "hardware"] = dspy.OutputField()
```

**Modules.** A module turns a signature into a strategy. `Predict` renders and parses once;
`ChainOfThought` adds a reasoning field before the answer; a custom `Module` composes several in
`forward`, which is Chapter 4's decomposition with the prompt text factored out. Added: the strategy
is swappable without rewriting prompts. Hidden: the prompt text itself, by design. DSPy owns field
formatting, output-format instructions, and demonstration layout. Ask the module for its compiled
prompt and store that in the registry if you want Chapter 4's guarantee that production prompt text is
versioned and reviewed.

**Optimizers.** An optimizer takes a module, a training set of input-output examples, and a metric
`(example, prediction) -> score`, and returns a module whose prompts score higher. The simplest
bootstraps few-shot demonstrations: run the module on the training set, keep examples the metric
accepted, try subsets as demonstrations, measure on a held-out set. More elaborate ones also propose
instruction wordings with a model and search over the combination. "Prompt optimization against a
metric" means exactly this: a search over prompt contents driven by a metric and data. It is not
fine-tuning; the weights do not change, the prompt does.

Four evaluation requirements follow from Chapter 24. A metric that is a faithful proxy for what you
care about, because the optimizer maximizes whatever you hand it. Enough labeled examples, dozens for
demonstration bootstrapping and hundreds for instruction search, because a metric on twelve examples
has a confidence interval wider than the improvement you seek. A frozen holdout the optimizer never
sees, because selecting demonstrations on the set you report on is Chapter 24's leakage. And a re-run
when the model changes, because the compiled prompt is tuned to one model.

Treat a compiled module as a build artifact: produced by a reproducible job from a dataset version
and a model version, stored under a version, promoted through Chapter 25's CI gate like any prompt
change.

### Provider agent SDKs and MCP SDKs

Provider agent SDKs vary in vocabulary but share three concepts.

**The tool loop.** An agent object holds instructions and tools; a runner sends the conversation to
the model, executes proposed tool calls, appends results, repeats until the model stops or a step
limit is hit. The primitive is Chapter 19's loop:

```python
# pseudocode: Chapter 19's AgentRuntime.run, simplified (names from agentkit)
append(GoalSet(goal))
while state.status is RUNNING:                       # stops on a TerminationReason
    if (reason := budget.exceeded(usage)): stop(reason)   # MAX_STEPS, MAX_COST, DEADLINE, ...
    completion = llm.complete(CompletionRequest(messages=state.messages, tools=visible_specs))
    append(ModelDecision(...)); append(BudgetUpdated(...))
    if not completion.tool_calls:                    # a final answer is a claim
        verdict = dod.verify(completion.text, state) # Definition of Done decides
        append(FinalAnswer(...) if verdict.passed else Note(kind="dod_rejected"))
        continue
    for call in completion.tool_calls:
        append(ToolCallRequested(call))
        decision = policy.check(tool, call.arguments, principal)  # agentkit ToolPolicy protocol
        if not decision.allowed: append(ToolCallDenied(...)); continue
        if decision.requires_approval and approver is None: stop(APPROVAL_REQUIRED)
        append(ToolCallApproved(...)); append(ToolResult(tool.execute(...)))
    append(StepCompleted(...))
```

Every `append` writes to the event store before state changes, which is what makes `replay` and
`resume` possible. With `executor_tools`, `tool.execute` is Chapter 16's `ToolExecutor`, so
permission, approval binding, and idempotency stay in `toolkit`.

The SDK version has none of the policy layer, budget, or event log; it typically offers a hook around
tool execution. Use the hook to route every call through your `ToolExecutor` so approval, idempotency,
and side-effect classification survive. Do not let the SDK call your functions directly.

**Handoffs.** One agent transfers the conversation to another, expressed as a tool whose execution
switches the active agent. This is Chapter 22's supervisor-worker or router pattern. Added: a tidy
split of instructions and tool sets by role. Hidden: budget and trace boundaries between agents.
Chapter 22's rule that budgets exist at parent and child and that a trace ID propagates across the
handoff is yours to enforce, because the SDK sees one conversation.

**Sessions.** A session stores conversation history so the next turn does not resend it: Chapter 5's
context input plus Chapter 21's conversation memory, with the SDK deciding how much history is kept
and whether it is compacted. Check that policy against your budget.

The MCP SDKs are thinner and map one to one onto Chapter 18: a server decorates functions as tools,
resources, or prompts; a client connects over stdio or HTTP, lists tools, calls them. Added:
correctness on protocol details (capability negotiation, framing, schema conventions) your minimal
implementation covered only partly. Hidden: the trust question. A tool description arriving over MCP
is untrusted text (Chapter 26), and the SDK presents it to the model as received. Filtering and
allow-listing stay in your code.

### Observability frameworks

Tracing SDKs for LLM applications (the ones bundled with the orchestration frameworks above, and
vendor-neutral ones built on OpenTelemetry's GenAI semantic conventions) record a span per model
call, tool call, retrieval, and chain step, with prompt, completion, tokens, and cost as attributes,
and ship them to a UI. The primitive is Chapter 31's trace model and `aie_core.observability`: a
`Tracer` yielding `Span` objects with the attribute schema Chapter 31 fixes (prompt version, model,
tokens, cost, cache hits, evidence IDs, policy results, eval scores).

Added: automatic instrumentation of the framework's objects, a UI, dataset collection from traces,
often an evaluation harness joining online traces with offline scores. Hidden: redaction (by default
usually none, so prompts with customer data leave your network), sampling (usually none, so cost
scales with traffic), and attribute names that will not match Chapter 31's schema unless you map
them. Emit spans through your own `Tracer` with your attribute names, and make the vendor SDK one
`Tracer` implementation behind it. Then the UI is a convenience and not a dependency.

## How it works

A framework adds value in four ways and hides cost in three. Knowing the pattern lets you evaluate a
framework you have never seen.

**Persistence.** Checkpointers, sessions, and thread stores give durable state across processes. You
built the interface in Chapter 17; the framework supplies backends exercised by many users. This is
usually the strongest argument for adopting an orchestration framework.

**Streaming.** Propagating token deltas and step events through a composed pipeline is plumbing you
would otherwise thread through every layer.

**Tracing.** Automatic spans for every framework object, with a UI. Valuable in development; in
production, map onto your schema so dashboards do not depend on one vendor's names.

**Ecosystem.** Integrations with stores, loaders, providers, and tools: the value that attracts
teams and decays fastest, because the integration you need is often the one that lags.

**Defaults.** Every number you set deliberately (chunk size, `k`, timeout, retry count,
`max_tokens`, temperature, history length) has a default somewhere in the framework, right for demos
and wrong for your workload. List them, set them, test them.

**Retries.** Frameworks retry at several layers: provider wrapper, runnable, graph node, sometimes
the tool. Three attempts at each of three layers is twenty-seven on a persistent failure, with the
cost in your bill and nowhere in your code. Chapter 29 explains why retries belong at exactly one
budgeted layer; find the framework's layers and disable all but one.

**Prompt text.** Prebuilt agents, query engines, synthesizers, structured-output coercion, and
optimizers all ship prompt text that is part of your system's behavior and invisible to code review
unless extracted. Frameworks are the main way prompts escape Chapter 4's registry.

## Architecture

The first diagram is the chapter's thesis: framework vocabulary on the left, primitives you own on
the right.

```mermaid
flowchart LR
  subgraph LC["LangChain"]
    lc1["chat model"] --> p3
    lc2["prompt template"] --> p4
    lc3["with_structured_output"] --> p6
    lc4["retriever"] --> p12
    lc5["runnable / LCEL"] --> p3b
    lc6["@tool / bind_tools"] --> p16
  end
  subgraph LG["LangGraph"]
    lg1["StateGraph + nodes"] --> p17
    lg2["conditional edges"] --> p17
    lg3["checkpointer"] --> p17c
    lg4["interrupt"] --> p17c
    lg5["stream_mode"] --> p19
  end
  subgraph LI["LlamaIndex"]
    li1["node parser"] --> p11
    li2["index / retriever"] --> p12
    li3["query engine"] --> p10
    li4["response synthesizer"] --> p13
  end
  subgraph DS["DSPy"]
    ds1["signature"] --> p4
    ds2["module"] --> p4
    ds3["optimizer"] --> p24
  end
  subgraph SDK["Agent and MCP SDKs"]
    sk1["runner / tool loop"] --> p19
    sk2["handoff"] --> p22
    sk3["MCP client/server"] --> p18
  end
  subgraph OWN["Primitives you built"]
    p3["Ch 3 LLMClient + ModelGateway"]
    p3b["Ch 3 composition + Ch 31 spans"]
    p4["Ch 4 PromptRegistry"]
    p6["Ch 6 complete_structured"]
    p10["Ch 10 RAG pipeline"]
    p11["Ch 11 Chunker"]
    p12["Ch 12 Retriever + Reranker"]
    p13["Ch 13 EvidencePacker + GroundedGenerator"]
    p16["Ch 16 ToolExecutor + PolicyEngine"]
    p17["Ch 17 Graph + add_router"]
    p17c["Ch 17 Checkpointer + pause_before"]
    p18["Ch 18 MCP"]
    p19["Ch 19 AgentRuntime events"]
    p22["Ch 22 supervisor / router"]
    p24["Ch 24 evalkit metrics"]
  end
```

The second shows where framework objects may live. Arrows point inward: a framework upgrade changes
the top row and possibly the adapters; the domain and its tests do not move.

```mermaid
flowchart TB
  subgraph Framework["Framework objects (third party)"]
    F1["vectorstore.as_retriever()"]
    F2["chat model"]
    F3["@tool function"]
    F4["compiled StateGraph"]
  end
  subgraph Adapters["adapters/ (yours, one class per framework object)"]
    A1["FrameworkRetrieverAdapter"]
    A2["FrameworkLLMAdapter"]
    A3["FrameworkToolAdapter"]
    A4["GraphWorkflowAdapter"]
  end
  subgraph Domain["domain/ (pure Python, no framework imports)"]
    D1["Retriever Protocol"]
    D2["LLMClient Protocol"]
    D3["Tool Protocol"]
    D4["Workflow Protocol"]
    D5["AnswerService"]
  end
  subgraph Tests["tests/ (fakes and recorded fixtures)"]
    T1["ScriptedLLM / ReplayLLM"]
    T2["InMemoryRetriever"]
  end
  F1 --> A1 --> D1
  F2 --> A2 --> D2
  F3 --> A3 --> D3
  F4 --> A4 --> D4
  D1 --> D5
  D2 --> D5
  D3 --> D5
  T1 --> D2
  T2 --> D1
```

## Implementation

Three plain-Python modules under `book/projects/examples/ch23/`, each with offline tests. None
imports a framework; they show the primitive side of the mapping. The excerpts below carry the idea
of each module; the full files, including the pieces marked "on disk", are in the repository.

### A runnable pipeline (`runnable.py`)

The base class gives every step the same calling convention; `Sequence` and `Retrying` are the two
operators worth reading. `RunLog` (on disk) is a list of dicts, one per step execution, with input,
output, latency, and attempt number. `Lambda` (on disk) wraps a plain function and records itself in
the log; `Parallel` (on disk) runs every branch of a dict on the same input.

```python
# path: book/projects/examples/ch23/runnable.py (excerpt; full file on disk)
class Runnable(Generic[In, Out]):
    name: str = "runnable"

    def invoke(self, x: In, log: RunLog | None = None) -> Out:
        raise NotImplementedError

    def batch(self, xs: list[In], log: RunLog | None = None) -> list[Out]:
        return [self.invoke(x, log) for x in xs]

    def stream(self, x: In, log: RunLog | None = None) -> Iterator[Out]:
        yield self.invoke(x, log)   # default: one chunk; real steps override

    def __or__(self, other: "Runnable[Out, Any] | Callable[[Out], Any]") -> "Sequence":
        return Sequence([self, coerce(other)])

    def with_retry(self, max_attempts: int = 3, retry_on: tuple[type[BaseException], ...] = (Exception,),
                   base_delay_s: float = 0.0) -> "Retrying":
        return Retrying(self, max_attempts, retry_on, base_delay_s)

# ...

class Sequence(Runnable[Any, Any]):
    name = "sequence"

    def __init__(self, steps: list[Runnable]) -> None:
        self.steps = steps

    def invoke(self, x: Any, log: RunLog | None = None) -> Any:
        for step in self.steps:
            x = step.invoke(x, log)
        return x

    def __or__(self, other: Runnable | Callable) -> "Sequence":
        return Sequence([*self.steps, coerce(other)])   # flatten instead of nesting

# ...

class Retrying(Runnable[In, Out]):
    # ...
    def invoke(self, x: In, log: RunLog | None = None) -> Out:
        attempt = 0
        while True:
            attempt += 1
            t0 = time.perf_counter()
            try:
                out = self.inner.invoke(x, log)
                if log is not None and attempt > 1:
                    log.record(self.name, x, out, (time.perf_counter() - t0) * 1000, attempt)
                return out
            except self.retry_on as exc:
                if log is not None:   # every failed attempt is visible, including the last one
                    log.record(self.name, x, f"error: {exc!r}", (time.perf_counter() - t0) * 1000, attempt)
                if attempt >= self.max_attempts:
                    raise
                time.sleep(self.base_delay_s * (2 ** (attempt - 1)))


def coerce(obj: Runnable | Callable | Mapping) -> Runnable:
    if isinstance(obj, Runnable):
        return obj
    if isinstance(obj, Mapping):
        return Parallel(obj)
    if callable(obj):
        return Lambda(obj)
    raise TypeError(f"cannot coerce {type(obj).__name__} to Runnable")
```

### A typed prompt specification with an optimizer (`signature.py`)

A `Signature` is the stable spec: an instruction and typed input and output fields (`Field` is a
frozen dataclass with `name`, `desc`, and `kind`). `render` turns it into prompt text; `parse` reads
the completion back by field label. `_coerce` (on disk) converts each value to its declared type and
reads booleans strictly, so an unrecognized value is an error rather than a silent `False`.

```python
# path: book/projects/examples/ch23/signature.py (excerpt; full file on disk)
@dataclass(frozen=True)
class Signature:
    """What goes in, what comes out, and one sentence of instruction.
    Prompt wording lives in the Module, not here: the spec is stable, the
    rendering is swappable. This is the property Chapter 4's registry wants."""

    instruction: str
    inputs: tuple[Field, ...]
    outputs: tuple[Field, ...]

    def render(self, values: dict[str, Any], demos: Sequence[dict[str, Any]] = ()) -> str:
        lines = [self.instruction, "", "Fields:"]
        for f in self.inputs + self.outputs:
            lines.append(f"- {f.name}: {f.desc}")
        for ex in demos:
            lines.append("")
            lines += [f"{f.name}: {ex[f.name]}" for f in self.inputs + self.outputs]
        lines.append("")
        lines += [f"{f.name}: {values[f.name]}" for f in self.inputs]
        lines.append(f"{self.outputs[0].name}:")
        return "\n".join(lines)
    # ... parse(text) -> dict: one typed value per output field, by label
```

The module and the optimizer are where DSPy's idea lives. Demonstrations are state of the module;
the optimizer searches over them with a metric and never touches the signature.

```python
# path: book/projects/examples/ch23/signature.py (excerpt; full file on disk)
@dataclass
class Predict:
    """The simplest Module: render, call, parse. Demos are *state* of the
    module that an optimizer may set; the signature never changes."""

    signature: Signature
    llm: LLMFn
    demos: list[dict[str, Any]] = field(default_factory=list)
    calls: int = 0

    def __call__(self, **inputs: Any) -> dict[str, Any]:
        self.calls += 1
        completion = self.llm(self.signature.render(inputs, self.demos))
        result = self.signature.parse(completion)
        return result

# ...

    def compile(self, module: Predict, train: Sequence[dict[str, Any]],
                dev: Sequence[dict[str, Any]]) -> Predict:
        input_names = [f.name for f in module.signature.inputs]
        candidates: list[dict[str, Any]] = []
        for ex in train:
            pred = module(**{k: ex[k] for k in input_names})
            if self.metric(ex, pred) >= self.threshold:
                candidates.append({**{k: ex[k] for k in input_names}, **pred})
        best_demos, best_score = list(module.demos), self.evaluate(module, dev)   # keep what was scored
        for n in range(1, min(self.max_demos, len(candidates)) + 1):
            trial = Predict(module.signature, module.llm, demos=candidates[:n])
            score = self.evaluate(trial, dev)
            if score > best_score:
                best_demos, best_score = candidates[:n], score
        return Predict(module.signature, module.llm, demos=best_demos)
```

`compile` is a method of `BootstrapFewShot`, which holds the `metric`, `max_demos`, and `threshold`;
its `evaluate` (on disk) is the mean metric over a dataset.

### Ports, adapters, and recorded fixtures (`ports.py`)

The domain owns three Protocols and one service. `Passage` and `Answer` (on disk) are small pydantic
models. `AnswerService.answer` (on disk) retrieves, builds a cite-or-abstain prompt, and abstains
when the model's reply cites no retrieved passage.

```python
# path: book/projects/examples/ch23/ports.py (excerpt; full file on disk)
# --- ports (owned by the domain) -----------------------------------------
@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[Passage]: ...


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, prompt: str) -> str: ...


@runtime_checkable
class Tool(Protocol):
    name: str
    def run(self, arguments: dict[str, Any]) -> str: ...

# ...

# --- adapter: framework -> port ------------------------------------------
class FrameworkRetrieverAdapter:
    def __init__(self, inner: FrameworkRetrieverLike) -> None:
        self._inner = inner

    def retrieve(self, query: str, k: int) -> list[Passage]:
        docs = self._inner.get_relevant_documents(query)[:k]
        return [Passage(id=d.metadata["id"], text=d.page_content,
                        source=d.metadata.get("source", "unknown"),
                        score=float(d.metadata.get("score", 0.0))) for d in docs]

# ...

@dataclass
class ReplayLLM:
    """Serve recorded completions. A prompt that was never recorded is a
    *test failure*, not a silent miss: it means the prompt text changed."""
    path: Path

    def complete(self, prompt: str) -> str:
        table = json.loads(self.path.read_text())
        try:
            return table[_key(prompt)]
        except KeyError as exc:
            raise LookupError(f"no recorded completion for prompt hash {_key(prompt)}; "
                              "the prompt text changed, re-record or update the fixture") from exc
```

`FrameworkRetrieverLike` (on disk) is a stand-in for a third-party retriever with its own vocabulary
(`get_relevant_documents`, `page_content`, `metadata`). `RecordingLLM` (on disk) is the other half
of the fixture pair: it wraps a live client, stores the SHA-256 prefix of each prompt with its
completion, and writes the JSON file atomically through a temporary file. The test that matters
most records one prompt and then sends a different one:

```python
# path: book/projects/examples/ch23/test_ports.py (excerpt; full file on disk)
def test_replay_fails_loudly_when_prompt_text_drifts(tmp_path: Path) -> None:
    fixture = tmp_path / "answers.json"
    retriever = FrameworkRetrieverAdapter(FrameworkRetrieverLike(DOCS))
    RecordingLLM(ScriptedLLM("ok [it-07]"), fixture).complete("old prompt")
    svc = AnswerService(retriever, ReplayLLM(fixture))
    with pytest.raises(LookupError, match="prompt text changed"):
        svc.answer("who handles vpn outages")
```

```bash
cd book/projects/examples/ch23 && pytest -q
```

## Code walkthrough

**Composition is flattening plus a calling convention.** `Sequence.__or__` appends to the step list
instead of nesting, which is the whole trick behind a readable trace: a flat list of named steps,
each recorded once in `RunLog`. `coerce` is why a bare function or a dict literal can appear to the
right of `|`. `stream` defaults to one chunk: in a real pipeline only the model step streams, and
every later step must be stream-aware or force a join, which is why "streaming through a JSON
parser" is a limitation many frameworks document.

**Retries announce themselves.** `Retrying.invoke` logs every failed attempt with its error, and the
attempt that finally succeeds, so a persistent failure leaves three entries rather than none. The
tests assert a `TimeoutError` is retried, a `ValueError` is not, and the final log entry carries
`attempt == 3`. A framework retry visible only as duplicated model-call spans cannot
answer "how much did retries cost this week".

**A signature is stable; demonstrations are module state.** `Signature` is frozen; `Predict.demos`
is mutable. The optimizer never touches the signature; it builds new `Predict` instances and keeps the
one that scores best on `dev`. The fake model misclassifies a "laptop" ticket until a demonstration
appears in the prompt, so dev accuracy moving from one half to one is the optimizer's effect made
observable. A second test checks that when nothing improves, the compiled module keeps the demos it
started with (none, in that test) rather than adding tokens for no gain.

**The adapter is the only place framework vocabulary appears.** `FrameworkRetrieverAdapter` reads
`page_content` and `metadata` from a stand-in framework document and emits a domain `Passage`; the
test asserts the prompt the model received contains `[hr-01]` and not `page_content`. The
`runtime_checkable` Protocols make `isinstance` the cheapest contract test an adapter can have, though
it checks only that the method names exist, not their signatures or return types; the contract
tests do the rest.

**Recorded fixtures key on prompt text.** `RecordingLLM` wraps a live client once and writes prompt
hash to completion; `ReplayLLM` raises `LookupError` on a missing prompt. The test shown above
matters most: changing the prompt after recording makes replay fail with a message that says why. A silent fallback
would hide exactly the drift you are testing for. In Northwind Assist, fixtures are re-recorded by an
`@pytest.mark.integration` test that runs only when a provider key is present.

## Framework selection criteria

Score each criterion from 0 to 2 for the framework as you would use it, not as its README describes
it, and weight by what matters to the system you are building. The last column says how to measure
rather than guess.

| Criterion | 0 | 1 | 2 | How to verify |
|---|---|---|---|---|
| State transparency | opaque or spread across objects | inspectable with effort | one typed object you define, readable at every step | dump state between two nodes; can you diff it? |
| Persistence and checkpointing | in-memory only | pluggable, your backend community-maintained | first-class backend you already run, with history | kill the process mid-run; resume; count duplicated side effects |
| Streaming | final result only | token deltas only | token deltas and step events, multiplexed | wire to a UI and show which node is running |
| Tracing | prints | vendor UI only | emits through your tracer with your attribute names | find prompt version and evidence IDs on one span |
| Retry semantics | hidden, stacked | configurable per layer | one layer, budgeted, attempts visible | force a persistent failure; count model calls billed |
| Testability with fakes | needs network | fake model object exists | every port fakeable, recorded fixtures supported | run the suite with the network off |
| Deployment model | requires vendor runtime | in-process with optional service | plain library, no required service | list the processes a deploy adds |
| Ecosystem fit | your store or provider missing | present but lagging | present and maintained | check the integration's last change and open issues |
| Upgrade churn | breaking changes per minor release | deprecations with migration notes | stable core, versioned extensions | read two release notes; count breaking items |
| Lock-in | domain types are framework types | adapters possible but thin | ports and adapters already clean | count framework imports outside `adapters/` |

The weights are yours: a regulated workflow weights transparency and persistence heavily; an internal
prototype weights ecosystem fit and little else. And "plain Python with your own primitives" is a
legitimate row to score alongside the frameworks. For a small deterministic workflow it usually wins,
and frameworks pay off when you need durable state, branching, checkpointing, tool ecosystems, or
standardized instrumentation.

### Worked example: scoring Project 5

Project 5, Northwind's incident-research agent (Chapter 20), plans an investigation, runs each step
as a small bounded agent, revises a cited report until a deterministic Definition of Done passes,
and posts it only after an on-call engineer approves. The approval can wait hours, through deploys
and pod restarts. Its tests replay whole trajectories offline from recorded cassettes. Three
candidates, each scored as the team would actually use it:

- **Primitives:** `agentkit`, `toolkit`, and `evalkit` as built in Chapters 16 to 24.
- **Graph library:** a graph orchestration library with its PostgreSQL checkpointer, tools still
  executed through `ToolExecutor`.
- **Provider SDK:** a provider agent SDK, with its tool hook routed through `ToolExecutor`.

Weights run from 1 to 3. Three criteria get 3: state transparency (the approver reads the state at
the pause, and the audit diffs state across steps), persistence (the approval outlives a process),
and testability (trajectory and replay tests are how Project 5 is evaluated). Ecosystem gets 1,
because every tool is internal: the runbook corpus, telemetry, and the channel.

| Criterion | Weight | Primitives | Graph library | Provider SDK |
|---|---|---|---|---|
| State transparency | 3 | 2 | 1 | 1 |
| Persistence and checkpointing | 3 | 1 | 2 | 1 |
| Streaming | 1 | 1 | 2 | 2 |
| Tracing | 2 | 2 | 1 | 1 |
| Retry semantics | 2 | 2 | 1 | 1 |
| Testability with fakes | 3 | 2 | 1 | 1 |
| Deployment model | 1 | 2 | 2 | 2 |
| Ecosystem fit | 1 | 0 | 2 | 1 |
| Upgrade churn | 2 | 2 | 1 | 1 |
| Lock-in | 2 | 2 | 1 | 0 |
| **Weighted total (max 40)** | | **34** | **26** | **20** |

The reasons behind the less obvious cells. Primitives score 1 on persistence because the event log
and resume exist but the durable backend for Project 5's store is JSON files, and a PostgreSQL
version is yours to write and operate. The graph library scores 1 on transparency because its state
is a dict whose merge rules live in reducer annotations, and 1 on testability because a fake model
exists but cassette replay of a whole trajectory is yours to add. The provider SDK scores 0 on
lock-in because its run items, hosted tools, and tracing assume one provider's model features, and
1 on persistence because a session stores history, but a pending approval must still be serialized
and resumed by your code.

Conclusion: keep the primitives for Project 5. Two cautions keep the score honest. First, the
primitives are already built and tested, so their row carries no build cost; for a team starting
from nothing, the persistence and ecosystem rows would weigh more and the gap would shrink. Second,
the result is sensitive to one cell: if approvals must survive days and redeploys and nobody wants
to own a PostgreSQL event store, persistence becomes the deciding criterion. The right move then is
not to switch wholesale but to adopt the graph library's checkpointer behind a `Workflow` port
(engineering question E3), keeping tools, policy, budgets, and evaluation in your code. The provider
SDK would win a different brief: a prototype weighted on ecosystem and time to first demo.

## Keeping domain logic framework-independent

Ports and adapters applied to an LLM application means four rules.

**The domain defines the Protocols.** `Retriever`, `LLMClient`, `Tool`, `Workflow`, `Tracer`, each
declared in `domain/` with the methods the domain calls and nothing more. A `Retriever` port says
`retrieve(query, k) -> list[Passage]`; it does not say `invoke` or `get_relevant_documents` and does
not return a framework `Document`.

**Framework objects are constructed in adapters and the composition root only.** An adapter wraps one
framework object and implements one port. The composition root (FastAPI startup, CLI entry) is the
only other place a framework is imported. A grep for the framework's package outside `adapters/` and
the composition root should return nothing; make that grep a lint rule.

**Domain types cross the boundary; framework types do not.** The adapter converts on the way in and
out. This costs a few lines per adapter and buys the ability to swap or remove the framework without
touching domain code or domain tests.

**Prompt text, policies, and budgets are domain.** Even when a framework renders the prompt or runs
the loop, the text comes from your registry, the tool policy from your `PolicyEngine`, the step and
cost budgets from your settings. The framework executes; it does not decide.

The usual objection is that this duplicates the framework's abstractions. It does, by a thin layer,
and the duplication is the point: the framework's abstractions change on its schedule, yours on yours.

## Prototype with the framework, ship the primitives

A pattern that works for many teams has three phases. Prototype with a framework to learn the
problem: which retrieval strategy helps, whether the workflow needs branching, what the tools should
be. Record traces throughout, because they tell you what the framework did for you. Then write the
domain with your primitives, porting the decisions and not the code: the chunk size you validated, the
graph shape, the prompts you extracted from the framework's templates and put under version. Finally,
keep the framework where it earned its place by the scoring table, typically for persistence and
streaming in a durable workflow, behind an adapter.

The opposite pattern is sometimes right. A team with mature primitives adopts a framework late, for
one capability it does not want to build: durable checkpointing with a hosted backend, or document
loaders for a one-time migration. The rule is the same in both directions: the framework enters
through an adapter, scored against the table, for a named reason.

The pattern that does not work is architecture by abstraction: building production out of framework
objects from day one because they were there, and discovering the hidden choices in production.

## Migration strategy

Migrating off a framework, or between two, is mechanical when the ports exist and a rewrite when they
do not, so the first step of any migration is to create them.

1. **Freeze behavior with recorded fixtures.** For each framework-backed component, record
   prompt-to-completion pairs and retrieval results on representative inputs, as `RecordingLLM`
   does. Both old and new implementations must pass them.
2. **Extract the hidden text.** Dump every prompt the framework assembled: synthesizer templates,
   agent system messages, structured-output instructions. Version each in the registry. The fixtures
   still pass, because the text has not changed, only its location.
3. **Introduce the port and adapter.** Wrap the framework object so the domain depends on the
   Protocol. Run the fixtures.
4. **Implement the port with your primitives.** Usually this is adapting a class from an earlier
   chapter. Differences in the fixtures are either bugs in the new implementation or hidden behavior
   of the old one that you have now found. Decide which to keep; update fixtures deliberately.
5. **Switch with a flag, compare in shadow.** Chapter 32's flags and shadow experiments: run both on
   live traffic, compare with Chapter 24's metrics, flip.
6. **Delete the adapter** and the dependency from the lockfile and the lint allow-list.

Step 4 is where migrations surface what the framework hid: a default `k`, a retry layer, metadata
injection, a prompt suffix. Each becomes a line in your configuration from then on.

## Production considerations

**Latency.** Composition overhead is negligible per step. The real latency cost is indirect: a
refine-mode synthesizer makes one call per chunk, a stacked retry turns one failure into many, a
session that resends unbounded history grows input tokens every turn. Measure time-to-first-token and
per-stage latency (Chapter 30) with the framework in place, not from its benchmarks.

**Cost.** Hidden retries and hidden prompt text are the two leaks. Account cost from your gateway or
tracer, not the framework's counter, so every model call is attributed to a request regardless of
which layer issued it. If the framework calls the provider directly, configure your `ModelGateway` as
its client where the framework accepts a custom client, often through a constructor argument.

**Security.** Prebuilt agents execute tool calls directly; MCP tool descriptions arrive untrusted;
framework retrievers may ignore ACL filters unless configured as pre-filters; tracing SDKs export
prompts containing customer data. Each control has a chapter (16, 26, 15, 31); confirm the framework
path honors the same controls as your primitive path, with a test per control.

**Operations.** Pin framework versions exactly. Read release notes before upgrading and run the
recorded-fixture suite after; a prompt-text change inside the framework surfaces as a `LookupError`
in replay. Keep a dependency inventory including transitive dependencies; a vulnerability in a loader
you do not use is still on your image.

## Common mistakes

**Architecture by abstraction.** Stacking components from a tutorial without being able to draw the
data flow. If you cannot name the primitive under each object, stop and learn it before shipping.

**Hidden prompt text in production.** Behavior depends on a template inside a dependency nobody has
read. Detect by diffing the prompt that reached the model (from your trace) against your registry.

**Upgrading without a net.** Bumping a framework version with no replay suite and no eval gate.
Stacked retries and prompt drift after an upgrade (both under Failure modes) are the two ways this
shows up; Chapter 25's eval gate is the second net.

**Letting the framework own the state type.** Domain state as the framework's dict, with reducers
encoding business rules; when the framework changes, the rules go with it. Define state as your
pydantic model and adapt.

**Treating the demo as the design.** Twenty lines that build a RAG pipeline are twenty lines that
chose a chunk size, a `k`, a prompt, and a synthesis mode for you.

## Failure modes

**Stacked-retry cost spike.** Signature: model calls per request rise sharply during a provider
incident while request volume is flat; your code shows one retry layer. Test: inject
`ProviderUnavailableError` on every call in a fake client; assert total attempts equal your single
policy's `max_attempts`.

**Interrupt double-execution.** Signature: a side-effecting call before an `interrupt` appears twice
in the tool audit log with the same run ID, once before the pause and once after resume. Test: a
counting fake tool before a pause node, resume, assert count is one. Fix: move side effects after the
pause or make them idempotent with Chapter 16's keys.

**Hidden metadata in the embedding.** Signature: retrieval is good on queries that mention file names
and poor on semantic queries; the embedded text in the trace contains paths and dates. Test: assert
the text passed to the embedding client equals the chunk text and nothing else.

**Prompt drift after upgrade.** Signature: eval scores shift after a dependency bump; the prompt hash
on model-call spans changes while the registry version does not. Test: the replay suite fails with
`LookupError`.

**Retriever without scores.** Signature: Chapter 14's retrieval metrics cannot be computed; the
retrieval span has IDs but no scores. Test: the adapter contract test asserts each `Passage` has a
score the framework actually returned (an adapter that defaults a missing score to 0.0, as the
minimal one in `ports.py` does, would pass silently), so an integration that returns none fails when
you integrate it, not in production.

**Session history growth.** Signature: input tokens per turn grow linearly across a conversation and
time-to-first-token with them. Test: run a twenty-turn fake conversation through the session; assert
input tokens stay within Chapter 5's budget.

## Tradeoffs

Speed of first version against understanding of the system: frameworks win the first and cost the
second unless you follow the three-question discipline. Ecosystem breadth against maintenance
surface: every integration you do not use is still in your dependency tree. Uniform abstractions
against stage visibility: a query engine is one call, and one call is one span unless the framework
exposes its stages. Durable execution for free against side-effect discipline: a checkpointer resumes
state; only your idempotency layer makes resumption safe. Metric-driven prompt optimization against
evaluation debt: compilation pays off exactly when you have the labeled data and the holdout to
support it, and misleads when you do not. None of these is a reason to avoid frameworks; each is a cost
to weigh in the scoring table.

## Evaluation and testing

Testing a framework-backed component has three layers, matching the three questions.

**Contract tests for adapters.** For each port, construct the adapter over a small in-memory
framework object and assert the port's contract: types, score presence, `k` respected,
empty-result behavior. These run offline and catch integration drift at upgrade time.

**Recorded-fixture tests for behavior.** `RecordingLLM` and `ReplayLLM`, or their retrieval
equivalents, freeze behavior on representative inputs. They are the migration safety net and the
version-drift detector. Re-record deliberately, in a reviewed change, with the diff of recorded
completions visible in the review.

**Evaluation with `evalkit` on the whole path.** Chapter 24's datasets and metrics, through the
framework path and the primitive path when both exist, with per-case deltas. For retrieval, Chapter
14's stage isolation requires the framework to expose retrieval results separately from generation;
if it does not, that is a finding against it in the scoring table. For compiled modules, the
requirements are those in the DSPy section: faithful metric, enough examples, frozen holdout, re-run
on model change.

Finally, test the hidden choices explicitly: one test per default you discovered, so that when the
framework changes one, a test says which.

## Before you ship

- [ ] Every framework dependency is pinned to an exact version, and the lockfile is reviewed on
  upgrade.
- [ ] A dump of every framework default you rely on (chunk size, `k`, timeout, retry count,
  `max_tokens`, temperature, history length) exists, and each value is set explicitly in your
  settings.
- [ ] Retries are enabled at exactly one layer; a test with a fake client that always raises
  `ProviderUnavailableError` asserts total model calls equal that layer's `max_attempts`.
- [ ] Every prompt the framework sends (synthesizer templates, agent system messages,
  structured-output instructions) is extracted, versioned in the registry, and passed back in
  explicitly; the prompt hash on model-call spans matches the registry version.
- [ ] Every tool call, including those proposed through an SDK or prebuilt agent, executes through
  `ToolExecutor`; provider-hosted tools are either disabled or explicitly allow-listed (Chapter 16).
- [ ] A grep for the framework's package outside `adapters/` and the composition root returns
  nothing, enforced as a lint rule in CI.
- [ ] Each adapter has a contract test: types, `k` respected, empty results, and real scores
  returned (not defaulted).
- [ ] A recorded-fixture replay suite runs offline on every dependency update and fails with
  `LookupError` on prompt drift.
- [ ] Side effects before any pause or interrupt are idempotent or moved after it; a counting fake
  tool proves one execution across pause and resume.
- [ ] Framework retrievers apply ACL filters before scoring, verified by a test run as a user
  without access (Chapter 15).
- [ ] Tracing exports go through your `Tracer` with redaction and sampling configured; no vendor SDK
  exports raw prompts by default.
- [ ] A session or history policy keeps input tokens within Chapter 5's budget over a twenty-turn
  test conversation.

## Exercises

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

## Key takeaways

- Every framework concept is a primitive you built: name it, picture the plain-Python version, then
  learn the vocabulary.
- Frameworks add persistence, streaming, tracing, and ecosystem. They hide defaults, retries, and
  prompt text. Make every hidden item explicit before production.
- LCEL-style composition is function composition with a calling convention and a log. Retry wrappers
  on chains re-run the whole chain and must be visible in cost accounting.
- A state graph framework is Chapter 17's engine with production backends. Only idempotency makes
  resumption safe, and interrupts re-execute the node that paused.
- Retrieval frameworks restate Chapters 11 to 13 and ship the grounding prompt. Extract it into your
  registry; insist on stage boundaries and scores so Chapter 14's evaluation works.
- DSPy-style compilation is a search over prompt contents driven by a metric and data. It needs a
  faithful metric, enough examples, a frozen holdout, and a re-run on model change.
- Agent SDKs run the tool loop without your policy layer; route execution through your
  `ToolExecutor`. MCP SDKs handle protocol correctness, not trust.
- Learn categories, not product names: each category absorbs predictable ledger rows and hides
  predictable choices. Score candidates on the ten criteria as you would use them, with your
  weights; plain primitives are a row in the table.
- Your domain owns the `Retriever`, `LLMClient`, and `Tool` Protocols; framework objects live in
  adapters. A grep for the framework outside `adapters/` should return nothing.
- Recorded fixtures that fail loudly on prompt drift are the migration safety net and the
  version-drift detector. Re-record deliberately, under review.

## Further reading

- *DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines* (Khattab et al.,
  2023): the original argument for separating signatures from prompt text and optimizing against a
  metric.
- LangGraph documentation: read the persistence, interrupt, and streaming pages with Chapter 17's
  engine beside you; they map one to one.
- LlamaIndex documentation: the response synthesizer and node parser pages show the defaults and
  prompt templates this chapter tells you to extract.
- LangChain documentation: the runnable interface and retry/fallback pages, read for what the
  wrappers retry and when.
- MCP Python SDK: a compact reference implementation to compare with Chapter 18's minimal client and
  server.
- *OpenTelemetry Semantic Conventions for Generative AI*: the attribute names to map framework
  tracing onto, so dashboards survive a framework change.
