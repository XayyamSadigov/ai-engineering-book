# Glossary

Terms used across the book, in alphabetical order. Each entry says what the term means and why it
matters to an engineer building AI systems, followed by the chapter or chapters where it is developed.
Book-specific names (libraries, classes, the running example) are included so that a reader who opens
a chapter in the middle can find where each piece was built.

---

## A

**A/B test**: An experiment that splits live traffic between two variants and compares outcome
metrics with a significance test. For AI changes it complements offline evaluation, because a prompt or
model that wins on the golden set can still lose on real users. (Ch 32)

**A2A (Agent2Agent protocol)**: An open agent-interop protocol that connects an agent to another
autonomous agent rather than to a tool: the remote side receives a task and runs its own loop with its own
model, tools, and policy. Unlike MCP, control passes to the other side, so its results are untrusted
input. (Ch 18, 22)

**Abstention**: The system's explicit decision not to answer, usually because the evidence is missing,
conflicting, or below a confidence threshold. A grounded system needs a tested abstention path; without
one, the model fills gaps with plausible fabrication. (Ch 13, 14)

**Access control list (ACL)**: Permission metadata attached to each document and chunk, such as the
groups or tenants allowed to read it. In RAG the ACL must be applied as a filter at retrieval time,
never after generation, or restricted text has already reached the model. (Ch 9, 11, 15)

**Action selector pattern**: An injection-containing design in which the model maps a request to one
action from a fixed menu and the result goes to the user, never back into the model. With no tool output
re-entering the context, an indirect injection has nothing to ride in on; the cost is that it is barely
an agent. (Ch 26)

**Adapter (fine-tuning)**: A small set of trainable weights added to a frozen base model, for example by
LoRA. Adapters are cheap to train, store, and swap, so one base model can serve several fine-tuned
behaviors. (Ch 33, 34)

**Admission control**: Deciding at the front door whether to accept a request, based on current load,
per-tenant quotas, and the request's estimated cost. Rejecting early, and shedding low-priority load
under overload, is cheaper and fairer than letting queues grow until every request times out. For a
shared serving platform the budget is KV-cache tokens, not request count. (Ch 29, 34, 36)

**Adversarial dataset**: An evaluation set built to break the system: injections, edge cases,
permission probes, ambiguous inputs. It measures bypass and failure rates that a golden set of typical
inputs never exercises. (Ch 24, 26)

**Agent**: A model, a set of tools, state, and a loop in which the model chooses the next action until
a termination condition holds. Agents trade testability and cost predictability for flexibility, so
the book treats them as the last rung of the decision ladder. (Ch 1, 19)

**Agent loop**: The repeated cycle of model call, tool proposal, policy check, execution, and
observation that drives an agent. Implemented as an explicit state machine with budgets, it becomes
debuggable and replayable instead of an opaque `while True`. (Ch 19)

**Agent-managed memory**: Memory that the agent writes itself, either through memory tools (save, search,
delete) or by editing a file it reads every session. Every write is model-inferred, so it goes through
the same write policy as any other memory, and edits to instruction files are review-gated. (Ch 21)

**Agent Skills**: A packaging format for procedural knowledge: a folder with instructions, scripts, and
resources that an agent loads on demand when a task matches the skill's description. Skills keep
rarely used procedures out of the base context and are part of the supply chain to review. (Ch 18, 38)

**Agentic RAG**: Retrieval in which an agent decides iteratively what to search for next, based on what
it has already found, under a step and token budget. It helps multi-hop questions and costs more
latency and nondeterminism than a fixed pipeline. (Ch 37)

**agentkit**: The book's bounded, event-sourced agent runtime library (`AgentRuntime`, events, budgets,
termination, replay). Built in Chapter 19 and reused by the architecture, memory, multi-agent, and
durable-agent chapters. (Ch 19, 20, 22, 38)

**aie_core**: The book's shared, provider-neutral library: message types, `LLMClient` adapters,
`FakeLLM`, `ModelGateway`, structured-output helper, token counting, embeddings, settings, and tracing.
Every project imports it instead of reinventing clients and retries. (Ch 3)

**Allowlist**: An explicit list of what is permitted (tools, argument values, URLs, domains, MCP
servers), with everything else denied. Allowlists are the dependable form of output and tool control;
denylists of bad patterns are easy to bypass. (Ch 16, 26, 27)

**Answer relevance**: Whether an answer addresses the question actually asked, regardless of whether
it is grounded or correct. A grounded answer about expense policy is irrelevant to a PTO question. Judged
with a rubric; word overlap with the question is a cheap floor. (Ch 14, 24, 25)

**Anti-corruption layer**: A translation layer that keeps a provider's or framework's types from leaking
into domain code. It is what makes switching model vendors or frameworks a contained change. (Ch 32)

**Approval binding**: Tying a human approval to the exact tool name and canonicalized arguments (for
example by hashing them), so the approved action cannot be swapped for a different one before
execution. (Ch 16, 19, 38)

**Approximate nearest neighbor (ANN)**: Index structures that find vectors close to a query without
comparing against every vector, trading a small loss of recall for large gains in latency. HNSW, IVF,
and PQ are the common families. (Ch 9)

**Architecture class**: The category a design belongs to before any boxes are drawn: LLM-enhanced
application, probabilistic workflow, RAG system, bounded agent, or platform. The class predicts where
the dominant risk lives and which design steps deserve the time. (Ch 36)

**At-least-once delivery**: A queue or execution guarantee that every job runs one or more times, never
zero. Because duplicates happen, every side effect behind it must be idempotent. (Ch 29, 38)

**Attention**: The transformer operation that lets each token weigh every earlier token when computing
its representation. Its cost grows quadratically with sequence length, which is one reason long
contexts are slow and expensive. (Ch 2)

## B

**Back-of-the-envelope sizing**: Step 9 of the design method: peak rate, tokens per second by class,
Little's Law for concurrency, daily tokens, storage, and cost per day, with every assumption labeled.
It catches budget-breaking designs before any code exists. (Ch 35)

**Backoff (exponential)**: Waiting progressively longer between retries, typically doubling a base
delay up to a cap. It gives an overloaded provider time to recover instead of hammering it. (Ch 3, 29)

**Backpressure**: Signals that flow upstream to slow producers when consumers cannot keep up, such as
bounded queues that block or reject. Without it, a slow model provider turns into unbounded memory
growth and timeouts across the system. (Ch 28, 29)

**Batch API**: A provider endpoint that accepts many requests for asynchronous processing, usually at a
discount and with a completion window of hours. It suits offline extraction and evaluation runs, not
interactive traffic. (Ch 3, 6, 30)

**BM25**: The classic lexical ranking function (from the Okapi system) that scores documents by term
frequency, inverse document frequency, and length normalization. It catches exact identifiers, error
codes, and rare words that dense embeddings often miss. (Ch 12)

**Bootstrap confidence interval**: An interval estimated by resampling the evaluation cases with
replacement and recomputing the metric many times. It tells you whether a difference between two
configurations is larger than the noise of a finite eval set. (Ch 24)

**Budget (agent)**: Hard limits on steps, tokens, cost, and wall-clock time for an agent run or a
subagent. Budgets turn runaway loops into a clean, observable termination instead of a surprise
invoice. (Ch 19, 22)

**Bulkhead**: Isolating resources (connection pools, concurrency slots, workers) per dependency or per
workload so that one slow or failing part cannot exhaust capacity for the rest. (Ch 29)

**Byte-Pair Encoding (BPE)**: A subword tokenization algorithm that starts from bytes or characters and
repeatedly merges the most frequent adjacent pair. It explains why rare words, code, and many
non-English languages cost more tokens per character. (Ch 2)

## C

**Cache break-even**: The reuse rate at which provider prompt caching starts saving money, given a
write multiplier on the first use of a prefix and a read multiplier on later hits within the TTL.
Every distinct prefix divides the traffic, so low-traffic prefixes can fall below break-even and
should leave explicit caching off. (Ch 30)

**Cache key (correctness key)**: The set of inputs a cached result depends on, such as prompt version,
model, user permissions, tenant, and index version. A key that omits any of them serves wrong or
unauthorized answers. (Ch 15, 30)

**Calibration**: The property that a score of 0.8 is correct about 80 percent of the time. Calibrated
scores are required whenever a threshold on confidence decides routing, abstention, or human review.
(Ch 6, 7, 24)

**Canary release**: Sending a small share of production traffic to a new prompt, model, or index
version and comparing its metrics with the stable version before widening the rollout. (Ch 25, 32)

**Candidate funnel**: The sequence of retrieval stages that narrows candidates, for example 100 from
hybrid retrieval, 20 after reranking, 5 packed into context. Sizing each stage is a recall, latency,
and cost decision. (Ch 12, 31)

**Capability tracking (CaMeL-style)**: A refinement of the dual LLM pattern in which the privileged
model writes a small program and an interpreter runs it while tracking the provenance (taint) of every
value. Per-tool policies then decide on provenance, for example refusing or requiring approval when a
recipient derives from untrusted data. (Ch 26)

**Cascade**: Routing that tries a cheap model first and escalates to a stronger one only when a
confidence or validation check fails. It saves cost when the check is reliable and adds latency to
escalated requests. (Ch 7)

**Catastrophic forgetting**: Loss of general capabilities after fine-tuning on a narrow dataset. It is
why a fine-tuned model must be evaluated on a broad regression suite, not only on the target task.
(Ch 33)

**Chain-of-thought**: Prompting a model to produce intermediate reasoning before its answer. It can
improve multi-step tasks, costs output tokens and latency, and the visible reasoning is not a reliable
account of how the answer was produced. (Ch 4)

**Chaos testing**: Deliberately injecting faults (provider outages, slow responses, rate limits,
malformed output) to verify that retries, fallbacks, and degraded modes actually work. (Ch 29)

**Chat template**: The model-specific format, with reserved role tokens, that serializes a list of
messages into the single token sequence the model actually reads. Serving an instruction-tuned model
with the wrong template fails silently as lower quality. (Ch 2, 7)

**Checkpoint**: A persisted snapshot of workflow or agent state at a step boundary, from which execution
can resume after a crash, an approval wait, or a deploy. (Ch 17, 23, 38)

**Chunk**: A retrievable unit of a document, with its own identifier, text, position, and inherited
metadata and ACL. Chunk boundaries decide what evidence can be retrieved together. (Ch 11)

**Chunking strategies**: The ways of cutting documents into chunks: fixed-size (simple, blind to
structure), recursive on a hierarchy of separators, sentence or paragraph, document-aware (headings,
tables, and code blocks as units), semantic (split where adjacent-sentence similarity drops), and
parent-child. Overlap between chunks protects boundary sentences at the cost of index size. Chunk size is
a recall and precision trade-off to evaluate, not a constant to copy. (Ch 11)

**Circuit breaker**: A component that stops calling a failing dependency after an error threshold,
fails fast for a cool-down period, then probes with trial requests. It prevents retry storms and gives
the dependency room to recover. (Ch 29)

**Citation precision and recall**: Deterministic citation metrics computed against gold sources.
Precision is the share of cited sources that are relevant to the question; recall is the share of
required sources the answer cites. They check the citations, not the claims: whether a claim is
supported is groundedness. See also citation validity. (Ch 14, 24)

**Citation validity**: The code check that every cited id is one the system actually showed the
generator, plus, where a case names required sources, that they are cited. It catches invented or copied
citation ids and is gated on every answered case. (Ch 14, 24, 25)

**Citation validation**: Checking after generation that every cited identifier exists in the packed
evidence and that the cited span supports the claim. It catches hallucinated and misattributed
citations before the user sees them. (Ch 13)

**Clean architecture**: Layering code into domain (pure logic), application, and adapters (providers,
databases, frameworks) so that business rules can be tested without models or infrastructure. (Ch 32)

**Client ID metadata document**: An MCP authorization mechanism in which a client's ID is an HTTPS URL
that serves a JSON description of the client, so an authorization server can trust a client it has
never seen without a separate registration step. (Ch 18)

**Cohen's kappa**: An agreement statistic between two raters that corrects for agreement expected by
chance. It is how the book checks whether an LLM judge agrees with human labels beyond what a constant
answer would achieve. (Ch 24)

**ColBERT (late interaction)**: A retrieval model that keeps one vector per token and scores a document
by matching each query token to its best document token. More precise than single-vector retrieval and
much larger to store. (Ch 37)

**Compaction**: Replacing older context (conversation turns, tool results, agent history) with a shorter
representation such as a summary or structured state. What must never be lossy (identifiers,
commitments, constraints) is kept verbatim. (Ch 5, 21, 38)

**Computer-use agent**: An agent that operates a graphical interface through screenshots and simulated
mouse and keyboard actions. Powerful for systems without APIs, slow, and hard to secure. (Ch 38)

**Confused deputy**: A privileged component that is tricked into using its authority on behalf of a
less privileged party. An agent with broad credentials acting on injected instructions is the typical
AI case. (Ch 18, 26)

**Constrained decoding**: Restricting which tokens the model may emit at each step, for example to those
valid under a JSON Schema or grammar. It guarantees syntactic validity, not semantic correctness. (Ch 2,
6)

**Context budget**: The allocation of the context window across output reserve, stable instructions,
conversation state, evidence, and tool results. Treating context as a budget forces explicit choices
about what to drop. (Ch 5, 30)

**Context engineering**: Deciding what enters the model's context on each request, in what order, with
what labels, and within what token budget. It is the main lever on answer quality after model choice.
(Ch 5)

**Context isolation**: Giving a subagent or worker a fresh, narrow context instead of the parent's full
history. It reduces distraction and limits what an injected instruction can reach. (Ch 20, 22)

**Context minimization**: Giving each step only the context it needs and dropping untrusted content once
it has served its purpose, for example composing an answer from the structured query rather than the
raw request. It shrinks both the injection surface and what an injection could exfiltrate, and is cheap
enough to apply almost everywhere. (Ch 26)

**Context relevance**: The share of retrieved context that is actually relevant to the question. Low
context relevance wastes tokens and distracts the generator even when recall is high. (Ch 14)

**Context window**: The maximum number of tokens (input plus output) a model can attend to in one call.
Quality often degrades well before the hard limit. (Ch 2, 5)

**ContextBuilder**: The book's component that assembles context under a token budget with priorities,
ordering, source labels, and compaction. It owns the untrusted-data labels of every item it assembles;
prompt templates label only their own variables. (Ch 5, 7)

**Contextual chunk header**: A short prefix added to a chunk's indexed text that restores context the
chunk lost when cut from its document, such as a heading breadcrumb or a document summary. It lets
lexical and dense search match a chunk that never names its own subject. (Ch 11, 37)

**Contextual retrieval**: Prepending a short, model-written description of where a chunk sits in its
document before indexing it, so terse chunks become findable by both lexical and dense search. (Ch 12)

**Continuous batching**: A serving technique that adds and removes sequences from the running batch at
every decode step instead of waiting for a whole batch to finish. It is the main reason modern engines
achieve high throughput. (Ch 34)

**Correctness**: Agreement with a known right answer: a label, a gold field value, a reference
answer, or an expected end state. It needs ground truth, unlike groundedness. Deterministic when the
truth is a value; judged against a reference when it is text. Chapter 14 measures it for RAG as rubric
coverage. (Ch 14, 24)

**Cosine similarity**: The cosine of the angle between two vectors, ignoring their length. The default
similarity for text embeddings; on normalized vectors it equals the dot product. (Ch 8)

**Cost per successful task**: Total cost (tokens, embeddings, reranking, tools, retries, human review)
divided by tasks that actually succeeded. More honest than cost per call, because retries and failures
are paid for too. (Ch 30)

**Cross-encoder**: A model that reads the query and a candidate document together and outputs a
relevance score. Too slow to run over a whole corpus, ideal for reranking a short candidate list.
(Ch 12)

## D

**Data/instruction separation**: Marking untrusted content (retrieved documents, tool results, user
uploads) as data with delimiters or datamarking, and instructing the model never to follow instructions
found inside it. It lowers injection success rates but is not a security control on its own. (Ch 4, 27)

**Day-one RAG evaluation**: A judge-free starter evaluation for a RAG prototype: about 30 questions
with required documents and an asker, hit@k run as each user, a leak check on every case, citation
validity, and abstention on trap questions. It can be built in an afternoon and grown later. (Ch 14)

**Dead-letter queue (DLQ)**: A queue where jobs go after exhausting retries, so they can be inspected
and replayed instead of being lost or retried forever. (Ch 29)

**Deadline propagation**: Passing one absolute deadline through every stage of a request so that
downstream timeouts shrink as time is spent. It stops a late stage from starting work the caller has
already given up on. (Ch 3, 29)

**Decision ladder**: The book's ordering of architectural options: prompt, retrieval, tools,
fine-tuning, agent. The accompanying principle of architectural restraint says to move up a rung only
when evaluation shows the lower rung cannot meet the requirement, because each rung enlarges the state
space to evaluate and secure. (Ch 1)

**Decode**: The phase of generation in which the model produces output tokens one at a time, each
depending on the previous ones. Decode is sequential and memory-bandwidth-bound, which sets
time per output token. (Ch 2, 34)

**Deduplication**: Removing exact and near-duplicate documents or chunks (for example with content
hashes or MinHash) before indexing or training. Duplicates waste context slots and skew evaluation.
(Ch 11, 33)

**Definition of Done**: An explicit, checkable acceptance condition for an agent's task, verified
against observations rather than the model's own claim of success. It is the success branch of
termination. (Ch 19, 20, 38)

**Dense retrieval**: Retrieval by embedding similarity between query and chunks. Good at paraphrase and
meaning, weak on exact identifiers and rare terms. (Ch 12)

**Determinism rule (durable workflows)**: The requirement that a replay-based durable workflow
function take the same path every time it is re-run from its history. Model and tool calls must be
recorded steps, and control code must not read the wall clock, draw random numbers, or generate fresh
ids; a violation surfaces only when a recovered run diverges. (Ch 17, 23, 38)

**Deterministic evaluation**: Scoring with code: exact match, schema validity, regex, set overlap,
numeric tolerance. Cheap, repeatable, and preferred wherever the property can be checked without a
judge. (Ch 24)

**Direct prompt injection**: An attacker typing instructions into the system's input to override its
intended behavior. (Ch 26)

**Distillation**: Training a smaller model on outputs of a larger one to reproduce its behavior on a
task at lower latency and cost. (Ch 33)

**DPO (Direct Preference Optimization)**: A preference-tuning method that trains directly on pairs of
preferred and rejected responses without a separate reward model or reinforcement learning loop.
(Ch 2, 33)

**DSPy**: A framework that treats prompts as programs with typed signatures and optimizes them
automatically against a metric. (Ch 23)

**Dual LLM pattern**: An injection-containing design with a privileged model that plans and calls tools
but never sees untrusted content, and a quarantined model that reads untrusted content but has no
tools. Code stores the quarantined outputs under symbolic names and substitutes the values only when
rendering an answer or filling a tool argument. (Ch 26)

**Durable execution**: Running long workflows so that each completed step is recorded and the workflow
resumes from the last step after a crash, with side effects not repeated. (Ch 38)

**Durable execution engine**: A separate service that owns long-running workflow runs: it records
progress in an event history, schedules steps onto your workers, fires timers, and delivers external
signals. It takes over crash recovery at the price of determinism rules, versioning rules, and payload
size limits; very long runs use a "continue as new" operation to start a fresh history. (Ch 17, 23,
38)

## E

**Egress control**: Restricting which external destinations a system or tool may contact, usually with
an allowlist. It closes the most common exfiltration channels from an injected agent. (Ch 26, 27)

**Elicitation (MCP)**: An MCP feature that lets a server ask the user a question through the host,
either as structured input against a small schema or, in URL mode, by sending the user to a URL. URL
mode keeps sensitive input such as a third-party sign-in or payment out of the host and the
model. (Ch 18)

**Embedding**: A fixed-length vector representing the meaning of a text (or image) so that similar items
are close in vector space. The foundation of semantic search, clustering, deduplication, and routing.
(Ch 8)

**Episodic memory**: Records of specific past events or interactions, such as a previous incident
investigation, retrieved when a similar situation arises. (Ch 21)

**evalkit**: The book's evaluation library: versioned case datasets, a runner with lineage,
deterministic and classification metrics, LLM judges with calibration, statistics, reports, and
release-gate thresholds. (Ch 24, 25)

**Evaluator-optimizer**: An architecture in which one model call produces a draft and another evaluates
it against criteria, looping until it passes or a budget runs out. Effective when the criteria are
checkable. (Ch 20)

**Event sourcing**: Storing every state change of an agent run as an append-only event, from which the
current state is derived. It makes runs auditable, resumable, and replayable. (Ch 19, 22, 38)

**Evidence packing**: Selecting, deduplicating, ordering, labeling, and fitting retrieved chunks into the
context budget, each with a stable id for citation. The packed evidence block is the contract between
retrieval and generation. (Ch 13)

**Exact search**: Comparing a query against every vector. Perfect recall and often fast enough up to
around a million vectors, which makes a dedicated vector database unnecessary for many systems. (Ch 9)

**Excessive agency**: Giving a model more tools, permissions, or autonomy than its task requires. It
turns any successful injection into a high-impact incident. (Ch 26)

**Exfiltration**: Leaking data out of the system, for example through a URL the model is induced to
render or a tool it is induced to call. Markdown images that load attacker URLs are a classic channel.
(Ch 26, 27)

**Expected calibration error (ECE)**: The weighted average gap between predicted confidence and observed
accuracy across confidence bins. (Ch 24, 33)

**Extraction pipeline**: A staged flow that classifies a document, extracts structured fields, validates
them with schemas and business rules, and routes uncertain results to human review. (Ch 6)

## F

**F1 score**: The harmonic mean of precision and recall. Macro-F1 averages F1 across classes equally, so
rare classes count as much as common ones. (Ch 24, 25, 33)

**Fail-closed and fail-open**: What happens when a guardrail or policy check itself errors or times
out. Fail-closed blocks the action and is the default for checks that protect permissions, money, or
data; fail-open lets it proceed and suits only low-risk checks where availability matters more. (Ch 27)

**Faithfulness**: Whether an output represents its source accurately: no contradictions, no changed
numbers or names, no dropped qualifiers. Distinct from groundedness, which asks whether each claim is
supported; an answer can be supported claim by claim and still drop a condition the source states. Many
libraries use the word for groundedness: ragkit's `faithfulness` score and Chapter 25's
`rag_faithfulness` both measure groundedness. Summaries are evaluated mainly for faithfulness.
(Ch 24, 25)

**FakeLLM and FakeEmbeddings**: The scripted `aie_core` model client that returns predefined responses
and records requests, and its deterministic embedding counterpart based on hashing or a provided
vocabulary. Together they let every test run without network access or API keys. (Ch 3, 8)

**Fallback**: An alternative used when the primary path fails: another model, another provider, a
cached answer, or a degraded response. Fallbacks can change assumptions such as context length or tool
support, so they need their own tests. (Ch 3, 7, 29)

**Fan-out and fan-in**: Running independent steps in parallel and combining their results. It cuts
latency for retrieval and research tasks at the cost of concurrency and partial-failure handling.
(Ch 17, 30)

**Feature flag**: A runtime switch that enables a prompt, model, or code path for some users or tenants.
It decouples deploy from release and makes rollback instant. (Ch 32)

**Few-shot prompting**: Including worked input and output examples in the prompt. It sets format and
style cheaply, costs tokens on every call, and biases the model toward the examples. (Ch 4)

**Fine-tuning**: Further training a model on task-specific examples to change its behavior. Useful for
stable format, style, or narrow tasks; the wrong tool for fresh facts or permissions. (Ch 33)

**Frozen holdout**: An evaluation set kept out of all prompt iteration and training and used only for
final decisions. It protects against overfitting to the dev set. (Ch 24, 33)

## G

**Golden set**: A curated evaluation dataset of representative inputs with expected outputs or
acceptance criteria, versioned and reviewed like code. (Ch 24)

**Goodput**: Throughput counting only requests that met their SLO. It, not raw tokens per second, is the
capacity you can promise users. (Ch 34)

**Graceful degradation**: Continuing to serve a reduced but useful response (smaller model, cached
answer, retrieval-only results) when a dependency fails, along pre-tested plans. (Ch 29)

**Grader**: A scoring function from a prompt, a response, and an optional reference to a number,
usually between 0 and 1, used as the reward in reinforcement fine-tuning. Graders can be string match,
field-level comparison, code that runs tests, or a model judge; they must be tested like code because
the model learns whatever they reward. (Ch 33)

**GraphRAG**: Retrieval over an entity graph extracted from the corpus, often with summaries of graph
communities, aimed at questions that span many documents. Expensive to build and maintain. (Ch 37)

**Greedy decoding**: Always picking the most probable next token. Deterministic in principle, though
provider infrastructure can still introduce variation. (Ch 2)

**Groundedness**: Whether every material claim in an output is supported (stated or directly
implied) by the evidence the system was given. It needs no reference answer, so it can run on
production traffic; a grounded answer can still be wrong if the evidence was outdated. Measured by a
lexical support check (Ch 25), a claim-level judge (Ch 14, reported as `faithfulness`), or a rubric
judge (Ch 24); Chapter 24 compares them. The book's grounded answer contract tells the generator to use
only packed evidence, cite it, treat it as data, and abstain when it is insufficient. Many sources call
this faithfulness; this book keeps the two apart. (Ch 13, 14, 24, 25)

**Grouped-query attention (GQA)**: An attention variant in which several query heads share one key and
value head, shrinking the KV cache and speeding decode. (Ch 2, 34)

**Guard model**: A classifier model, hosted or self-hosted, that scores inputs or outputs for safety
or injection categories. It is a probabilistic sensor plugged into a guardrail pipeline, not a
boundary that replaces allowlists and tool policy. (Ch 27)

**Guardrail**: A check that runs around a model or tool call (input, context, output, or tool stage) and
allows, flags, redacts, or blocks. Guardrails are layered because each one has gaps. (Ch 27)

**guardrails (package)**: The book's guardrail library: an ordered pipeline of checks at four stages,
each returning a decision with a reason and declaring whether it fails closed or open. (Ch 27)

## H

**Hallucination**: Fluent output that is not supported by the input or by facts. In RAG it shows up as
unsupported claims and invented citations, which is why citation validation and abstention exist.
(Ch 10, 13)

**Harness**: The code around a model that turns it into an agent: tools, context management, loop
control, sandboxing, and feedback such as test results. Most agent quality differences come from the
harness. (Ch 19, 38)

**Hierarchical summaries**: Section and document summaries written at ingestion and indexed alongside
chunks, optionally clustered into higher-level summaries. They answer corpus-wide questions by
retrieving a handful of summaries, without the entity extraction a graph needs. (Ch 37)

**Hit rate (hit@k)**: In retrieval evaluation, the share of questions for which at least one relevant
document appears in the top k. In caching, the share of lookups served from cache. (Ch 10, 14, 30)

**HNSW (Hierarchical Navigable Small World)**: A graph-based ANN index with layered proximity graphs.
Strong recall and latency; memory-heavy; tuned with parameters such as `M` and `ef_search`. (Ch 9)

**Human-in-the-loop**: A workflow design in which a person reviews, approves, or corrects a step. In
durable agents the approval is a first-class state the run can wait in for hours. (Ch 16, 17, 38)

**Hybrid retrieval**: Combining lexical (BM25) and dense retrieval, typically fused with Reciprocal Rank
Fusion. It is the robust default for enterprise corpora full of identifiers and paraphrases. (Ch 12)

**HyDE (Hypothetical Document Embeddings)**: Generating a hypothetical answer to the query and embedding
that instead of the query. It helps when queries are short and documents verbose, and hurts when the
hypothetical answer is confidently wrong. (Ch 12)

## I

**Idempotency key**: A unique key attached to a side-effecting request so that repeats (from retries or
duplicate deliveries) execute only once. Essential for tool calls such as creating tickets or sending
replies. (Ch 16, 29)

**Index poisoning**: Planting malicious or misleading content in a corpus so that it is retrieved and
trusted later. A form of indirect injection through the knowledge base. (Ch 21, 26)

**Indirect prompt injection**: Instructions hidden in content the system processes (web pages, documents,
emails, tool results) that the model then follows. The defining security risk of RAG and agents.
(Ch 26)

**Ingestion**: Parsing, cleaning, enriching, chunking, embedding, and indexing source documents.
Ingestion quality bounds retrieval quality. (Ch 11, 15)

**Insecure output handling**: Passing model output into an interpreter (browser, SQL engine, shell)
without encoding or validation, enabling XSS or injection through the model. (Ch 26, 27)

**Instruction file**: A human-readable file, such as an `AGENTS.md` at a repository root, that an agent
reads at the start of every session and may edit as it works. Because an injected edit becomes a
standing instruction, edits are diffed, checked against the write policy, and review-gated. (Ch 21)

**Instruction hierarchy**: The precedence order among instruction sources: system, developer, user, then
data. Prompts and policies are designed so that lower levels cannot override higher ones. (Ch 4)

**Instruction tuning**: Supervised fine-tuning of a pretrained model on curated conversations so that
it follows a chat format and answers after the assistant marker. It teaches task-following and style,
not much new knowledge. (Ch 2)

**IVF (Inverted File index)**: An ANN index that clusters vectors and searches only the clusters nearest
the query. Tuned by the number of clusters probed. (Ch 9)

## J

**Jailbreak**: Input crafted to make a model ignore its safety training or system instructions. Distinct
from injection in that the target is the model's policy rather than the application's instructions.
(Ch 26)

**Jitter**: Randomness added to retry delays so that many clients do not retry in lockstep and create
synchronized load spikes. (Ch 3, 29)

**JSON Schema**: A standard vocabulary for describing JSON structure. It defines tool parameters and
structured-output formats across providers. (Ch 3, 6, 16)

**JSON-RPC**: A lightweight remote procedure call protocol encoded in JSON. MCP messages are JSON-RPC
requests, responses, and notifications. (Ch 18)

**JWT (JSON Web Token)**: A signed token carrying identity claims such as user, groups, and tenant. The
capstone uses it to derive permissions for retrieval and tools. (Ch 28, 39)

## K

**Knowledge cutoff**: The date after which a model's training data has no information. Facts that change
after it must come from retrieval or tools. (Ch 2)

**KV cache**: Stored key and value tensors for previous tokens so they are not recomputed at each decode
step. Its size grows with layers, heads, sequence length, and batch size and often limits how many
requests a GPU can serve. (Ch 2, 34)

## L

**LangChain**: A framework of composable components (prompts, models, parsers, retrievers) for LLM
applications. (Ch 23)

**LangGraph**: A framework for building workflows and agents as state graphs with checkpoints and
interrupts. (Ch 17, 23)

**Latency budget**: An allocation of the end-to-end latency target across stages such as retrieval,
reranking, prefill, decode, and tools. A stage that overspends is visible immediately. (Ch 12, 30, 35)

**Latent attention (MLA)**: An attention variant, multi-head latent attention, that caches one compressed
latent vector per token per layer instead of separate keys and values for every head. It shrinks the KV
cache well below the grouped-query formula, so KV sizing must follow the model's configuration.
(Ch 34)

**Learned sparse retrieval**: Retrieval with a transformer that outputs a weight per vocabulary term,
including related terms absent from the text; SPLADE is the best-known example. The result is still a
sparse vector served from an inverted index, sitting between BM25 and dense retrieval. (Ch 12, 37)

**Least privilege**: Granting each tool, agent, and credential only the permissions its task needs.
(Ch 16, 26)

**Lexical retrieval**: Retrieval by matching words, typically BM25 or a database full-text index such as
PostgreSQL `tsvector`. (Ch 12)

**Little's Law**: In a stable system, average concurrency equals arrival rate times average time in
system. Used for capacity planning: it tells you how many concurrent sequences a target load implies.
(Ch 29, 34, 35, 36)

**LlamaIndex**: A framework focused on data ingestion, indices, and query engines for RAG. (Ch 23)

**LLM-as-judge**: Using a model with a rubric to score outputs on qualities code cannot check, such as
groundedness, faithfulness, or answer relevance. Useful only after calibration against human labels and
with known biases controlled. (Ch 14, 24)

**LLMClient**: The `aie_core` protocol with synchronous, asynchronous, and streaming completion methods
that all provider adapters implement. (Ch 3)

**Logits**: The raw, unnormalized scores a model outputs for every token in its vocabulary at each step.
Softmax turns them into probabilities that sampling draws from. (Ch 2)

**Long context versus RAG**: The choice between placing whole documents in a large context window and
retrieving selected chunks. Long context is simpler for small, stable corpora; RAG wins on cost,
freshness, permissions, and citations. (Ch 10, 37)

**LoRA (Low-Rank Adaptation)**: A fine-tuning method that freezes the base weights and trains small
low-rank matrices added to selected layers. It reduces trainable parameters by orders of magnitude.
(Ch 33)

**Lost in the middle**: The observed tendency of models to use information at the start and end of a
long context better than information in the middle. It motivates careful ordering of evidence. (Ch 2,
5)

## M

**Managed open-weight endpoint**: A cloud or inference provider serving open-weight models for you,
billed per token or per reserved GPU-hour. It gives model choice, portability, and often your own
adapters without running GPUs, with engine and quantization changes on the provider's schedule.
(Ch 7, 34)

**Map-reduce**: Processing many items or chunks independently (map) and combining the partial results
(reduce). Used for long-document summarization and parallel research. (Ch 17, 37)

**Matryoshka embeddings**: Embeddings trained so that their leading dimensions form a usable smaller
embedding. They allow truncation to cut storage and latency with a measurable quality loss. (Ch 8)

**MCP (Model Context Protocol)**: An open protocol through which AI applications connect to external
tools, resources, and prompts in a uniform way. It standardizes discovery and invocation; authorization
remains your responsibility. (Ch 18)

**MCP client**: The connector inside a host that maintains a session with one MCP server. (Ch 18)

**MCP host**: The AI application (assistant, IDE, agent runtime) that the user interacts with and that
runs MCP clients. The host owns policy and user consent. (Ch 18)

**MCP registry**: The MCP project's public catalog of server metadata (name, version, package or
remote endpoint), which hosts and private subregistries can consume. An entry is provenance, not
approval: it says who published a server, not that its descriptions or code were reviewed. (Ch 18)

**MCP server**: A program that exposes tools, resources, and prompts over MCP, locally over stdio or
remotely over HTTP. Its descriptions are untrusted input. (Ch 18)

**MCP tasks**: An MCP feature, introduced as experimental and moved into a formal extension in the
2026-07-28 revision, in which a request runs as a task: the requester gets a task handle at once and
polls for status and the eventual result, or cancels it. It fits long-running tool calls that a single
waiting request does not. (Ch 18)

**Memory poisoning**: Getting false or malicious content written into an agent's long-term memory so
it influences future sessions. Write policies and provenance are the defenses. (Ch 21, 26)

**memorykit**: The book's memory library: conversation, semantic, episodic, and user-profile stores with
provenance, confidence, expiry, tenant scope, write policy, and hard deletion. (Ch 21)

**Mental models (the ten)**: The book's recurring principles, such as "LLM output is probabilistic" and
"the model proposes, code authorizes", each tied to an engineering consequence. (Ch 1)

**Metadata filtering**: Restricting vector search by attributes such as tenant, ACL group, date, or
document type. Pre-filtering applies the restriction during search; post-filtering applies it after and
can return too few results. (Ch 9)

**Minimum viable evaluation**: The week-one evaluation a team can build from nothing: 30 to 50 real
cases, deterministic checks first, and one judge calibrated on about 30 labels that reports but does not
gate. Every later evaluation practice upgrades one part of it. (Ch 24)

**Mixture of experts (MoE)**: An architecture whose blocks hold many feed-forward "experts" plus a
router that sends each token through only a few. Total parameters (every expert) set memory; active
parameters (what one token passes through) set compute per token, so the two must not be
confused when sizing. (Ch 2, 34)

**ModelGateway**: The `aie_core` component that wraps a primary client with retries, fallbacks,
rate limiting, caching, concurrency limits, cost accounting, and tracing. Every model call in the book
goes through it. (Ch 3)

**Model hint**: Metadata in a prompt file (a size tier, whether structured output is needed) that
advises the router which models suit the prompt, instead of naming a vendor model. It is advice: policy
routing rules outrank it. (Ch 4, 7)

**Model pinning**: Referencing an exact model version rather than an alias that the provider may update.
It keeps behavior stable between deliberate, evaluated upgrades. (Ch 7)

**Model routing**: Sending each request to the model best suited by rules, a classifier, embedding
similarity, or confidence. It balances quality, latency, and cost. (Ch 7)

**MRR (Mean Reciprocal Rank)**: The average of 1 divided by the rank of the first relevant result. It
rewards putting a relevant document at the top. (Ch 14)

**Multi-adapter serving**: Loading one base model once and keeping many LoRA adapters resident, applying
each request's adapter within the same batch. Each adapter looks like a separate model on one endpoint,
so many low-traffic fine-tunes cost roughly one replica. (Ch 34)

**Multi-agent system**: Several agents with separate contexts and roles coordinated by a supervisor,
pipeline, or shared log. Justified by parallelism, isolation, independent verification, or permission
domains, not by default. (Ch 22)

**Multi-query retrieval**: Generating several reformulations of a query, retrieving for each, and fusing
the results to raise recall. (Ch 12)

**Multi round-trip request (MCP)**: The 2026-07-28 MCP mechanism that replaces server-to-client calls
in a stateless protocol: the server returns a result asking for input (for example an elicitation)
plus an opaque request state, and the client re-issues the call with the answers and the echoed
state. (Ch 18)

**Multi-tenancy**: Serving several customers or business units from one system while keeping their data,
quotas, and costs isolated. (Ch 9, 15, 29, 30)

## N

**nDCG (normalized Discounted Cumulative Gain)**: A ranking metric that rewards relevant results near the
top, supports graded relevance, and is normalized against the ideal ordering. (Ch 14)

**Northwind Assist**: The book's running example: an internal assistant for a fictional company with two
tenants, `retail` and `logistics`, built across the projects and assembled in the capstone. (Ch 1, 39)

## O

**Online evaluation**: Measuring quality on live traffic through user feedback, corrections, sampled
judging, and canary comparisons. (Ch 25, 32)

**Opaque reasoning items**: Encrypted, signed, or server-referenced reasoning blocks that some provider
APIs return alongside tool calls and expect back unchanged on the next turn. A loop that rebuilds its
transcript without them quietly degrades or fails, so they need their own field in the recorded
event. (Ch 19)

**OpenTelemetry (OTel)**: A vendor-neutral standard and SDK for traces, metrics, and logs. The book's
tracer can export to it so AI spans join the rest of the system's telemetry. (Ch 28, 31)

**Outbox pattern**: Writing an intended side effect to a durable table in the same transaction as the
state change, then delivering it separately. It prevents lost or duplicated actions when a process
crashes between steps. (Ch 16)

**OWASP Top 10 for LLM Applications**: A widely used list of the main security risks of LLM
applications, such as prompt injection and excessive agency. Reviewers and questionnaires use its
vocabulary, so a threat model should map onto it. (Ch 26)

## P

**PagedAttention**: A KV cache memory manager that stores keys and values in fixed-size blocks, like
virtual memory pages, reducing fragmentation and allowing more concurrent sequences. (Ch 34)

**Pairwise judging**: Asking a judge which of two outputs is better instead of scoring each alone. More
sensitive to small differences; requires randomizing order to control position bias. (Ch 24)

**Parent-document retrieval**: Indexing small chunks for precise matching but returning their larger
parent section to the generator for context. (Ch 11, 12)

**PEFT (Parameter-Efficient Fine-Tuning)**: The family of methods, including LoRA, that adapt a model by
training a small fraction of its parameters. (Ch 33)

**pgvector**: A PostgreSQL extension that adds vector columns, distance operators, and ANN indexes. It
lets one database hold documents, metadata, ACLs, and embeddings. (Ch 9, 28)

**PII (Personally Identifiable Information)**: Data that identifies a person, such as names, emails, and
phone numbers. It must be detected and redacted before it reaches models, logs, or memory where policy
forbids it. (Ch 21, 27)

**Plan-then-execute**: An injection-containing design in which the model commits to a plan before it
reads any untrusted content and code executes that plan. Injected content can no longer add steps, but
it can still corrupt the arguments and content of planned steps. (Ch 26)

**Planner-executor**: An architecture in which one step produces an explicit plan and an executor
carries it out step by step, replanning when observations contradict the plan. (Ch 20)

**Position bias**: A judge's tendency to prefer an answer because of its position, such as the first of
two. Controlled by randomizing or swapping order. (Ch 24)

**Precision@k**: The share of the top k results that are relevant. (Ch 14)

**Preference tuning**: Training that shapes which of several acceptable answers a model prefers, from
compared pairs of answers, through RLHF or DPO-style methods. Refusal habits and sycophancy are among
the tendencies it instills. (Ch 2, 33)

**Prefill**: The phase in which the model processes all input tokens in parallel and builds the KV cache.
Prefill is compute-bound and dominates time to first token for long prompts. (Ch 2, 34)

**Procedural memory**: Stored knowledge of how to do things: instructions, playbooks, skills. (Ch 21, 38)

**Product quantization (PQ)**: Compressing vectors by splitting them into sub-vectors and replacing each
with the id of its nearest codebook entry. It cuts memory sharply at some cost in recall. (Ch 9)

**Projects (P1 to P6)**: The six runnable systems the capstone assembles. P1, structured extraction API,
`book/projects/p1-extraction-api`, package `extraction_api` (Ch 6). P2, semantic search,
`p2-semantic-search`, package `semsearch` (Ch 9). P3, production RAG assistant, `p3-rag-assistant`,
package `rag_assistant` (Ch 15). P4, tool-using support assistant, `p4-support-assistant`, package
`support_assistant` (Ch 16). P5, incident-research agent, `p5-incident-agent`, package
`incident_agent` (Ch 20). P6, multi-agent research team, `p6-research-team`, package `research_team`
(Ch 22). The capstone lives in `book/capstone/northwind-assist`. (Ch 6, 9, 15, 16, 20, 22, 39)

**Prompt caching**: Reusing the provider's computation for a repeated prompt prefix, which reduces cost
and time to first token. It requires stable content at the start of the prompt. (Ch 3, 5, 30, 34)

**Prompt injection**: Input that causes a model to follow an attacker's instructions instead of the
application's. Prompt wording cannot fully prevent it; containment comes from permissions, policy, and
egress control. (Ch 26, 27)

**Prompt registry**: A versioned store of prompt templates with metadata, so that every request records
which prompt version produced it and changes are tested and reviewed. (Ch 4)

**Prompt template**: A parameterized prompt rendered with request data, with escaping so that inserted
data cannot break the prompt's structure. (Ch 4)

**Protected resource metadata**: A document a remote MCP server points to in its `401` response that
names the authorization servers it trusts. It is the first step of the MCP authorization flow. (Ch 18)

**Provenance**: The recorded origin of a piece of content or memory: source, author, time, and how it was
derived. It drives trust decisions, citation, and deletion. (Ch 21, 26)

**Provider abstraction**: A provider-neutral interface such as `LLMClient` behind which vendor SDKs sit,
so applications can switch or combine providers. (Ch 3, 32)

**Provider-hosted tools**: Tools the model provider runs mid-generation, such as web search, code
execution, file search, and remote MCP connectors, enabled in the request. They bypass your tool
policy, idempotency, and audit, so enabling one is a per-request policy decision. (Ch 16, 18, 23)

## Q

**QLoRA**: LoRA fine-tuning on a base model quantized to 4 bits, which lets large models be adapted on a
single GPU. (Ch 33)

**Quantization**: Storing model weights (and sometimes activations or KV cache) in fewer bits, such as
8-bit or 4-bit. It reduces memory and speeds serving with a quality cost that must be measured on your
task. (Ch 34)

**Query rewriting**: Transforming a user query before retrieval, for example making a follow-up question
self-contained using conversation history. (Ch 12)

## R

**RAG (Retrieval-Augmented Generation)**: Retrieving relevant documents at request time and giving them
to the model as evidence for its answer. It provides fresh knowledge, citations, and permission-aware
answers without retraining. (Ch 10)

**ragkit**: The book's RAG library, built in layers: parsers, the document model, cleaning,
deduplication, and chunking strategies (Ch 11); `ragkit.retrieval` with BM25, dense, hybrid fusion,
reranking, and query transforms (Ch 12); `ragkit.generation` with evidence packing, citation
validation, and abstention (Ch 13); and `ragkit.eval` with retrieval metrics and stage isolation
(Ch 14). Project 3 composes it into a service. (Ch 11, 12, 13, 14, 15)

**Rate limit**: A cap on requests or tokens per unit of time imposed by a provider or by your own
gateway. Exceeding it returns errors (often HTTP 429) that should be retried after the indicated delay.
(Ch 3, 29)

**ReAct**: An agent pattern that interleaves reasoning steps with tool actions and observations in one
loop. The simplest agent architecture and the baseline for others. (Ch 20)

**Reasoning model**: A model that spends additional generated tokens on internal reasoning before
answering, trading latency and cost for accuracy on multi-step problems. Many expose a reasoning-effort
setting, and some return opaque reasoning items a tool loop must preserve. (Ch 2, 7, 19)

**Recall@k**: The share of relevant documents that appear in the top k results. The primary retrieval
metric, because evidence that is not retrieved cannot be used. (Ch 9, 14)

**Reciprocal Rank Fusion (RRF)**: A method that merges ranked lists by summing 1 divided by (k plus rank)
for each document across lists. It needs no score calibration, which makes it the default for hybrid
retrieval; weighted score fusion is the more tunable and more fragile alternative. (Ch 12)

**Red teaming**: Structured adversarial testing by people or automated generators to find security and
safety failures before attackers or users do. (Ch 26, 27)

**Redaction**: Replacing sensitive values such as PII or secrets with placeholders before text reaches a
model, a log, or a trace. (Ch 27, 31)

**Reflection**: An architecture in which the model critiques and revises its own output. Without new
evidence, self-critique tends to reinforce the original mistake. (Ch 20)

**Regression suite**: A set of cases that previously passed and must keep passing after every prompt,
model, or code change. (Ch 4, 24, 25)

**Reinforcement fine-tuning (RFT)**: Fine-tuning in which a provider samples several responses per
prompt, scores them with your grader, and moves the weights toward the higher-scoring ones. It suits
tasks whose answers are checkable but hard to demonstrate, and its typical failure is reward
hacking. (Ch 33)

**Release gate**: An automated CI check that blocks a change unless evaluation metrics meet configured
thresholds, compared with the current baseline. (Ch 24, 25)

**reliability (package)**: The book's library of system-level resilience primitives: deadlines, circuit
breakers, bulkheads, admission control, job queues, workers, degradation policies, and fault injection.
(Ch 29)

**Repair loop**: Re-asking the model with the validation error when its structured output fails to
parse or validate, up to a fixed number of attempts. (Ch 6)

**Replay**: Re-running an agent or workflow from its recorded events, with recorded model and tool
responses, to debug or to evaluate a change against real trajectories. (Ch 19, 25)

**Request lineage**: The record of everything that produced a response: prompt version, model, retrieved
evidence ids, tool calls, policy decisions, cost. You must be able to answer these for any production
request. (Ch 1, 31)

**Reranking**: Re-scoring a short list of retrieval candidates with a more precise model, usually a
cross-encoder. It often gives the largest precision gain per unit of effort in RAG. (Ch 12)

**Resource indicator**: A parameter in an OAuth token request naming the target server's canonical
URL, so the issued token is bound to that server as its audience. In MCP it makes a token stolen from
one server useless against another. (Ch 18)

**Response cache**: A cache of complete model responses keyed on the exact request and its correctness
inputs. Safe only when the key includes everything the answer depends on. (Ch 3, 30)

**Result equivalence**: Judging generated SQL by whether its executed result matches the gold result
(normalized column names, rows compared as a multiset, numeric tolerance), not by how the SQL is
written. The primary metric for text-to-SQL. (Ch 36)

**Retry storm**: Load amplification when every layer of a call chain retries independently, multiplying
traffic on an already struggling dependency. (Ch 29)

**Retryable error**: An error worth retrying, such as a timeout, 429, or 5xx, as opposed to an invalid
request or a content-filter refusal. The book's error types carry a `retryable` flag. (Ch 3, 29)

**RLHF (Reinforcement Learning from Human Feedback)**: Training a reward model on human preference
comparisons and optimizing the language model against it with reinforcement learning. (Ch 2, 33)

**Router**: A component that chooses the model, prompt, or path for a request. Also an agent
architecture in which a classifier step dispatches to specialized handlers. (Ch 7, 20)

**Rubric**: The written criteria and score levels a judge applies. One dimension per rubric gives more
reliable judgments than a single holistic score. (Ch 24)

## S

**Sampling**: Drawing the next token from the model's probability distribution, shaped by temperature,
top-k, and top-p. It is the source of output variability. (Ch 2)

**Sandbox**: An isolated execution environment with restricted filesystem, network, and resources for
running model-generated code or risky tools. (Ch 16, 38)

**Self-consistency**: Sampling several answers and taking the majority or checking agreement. It raises
accuracy and flags uncertainty at a multiple of the cost. In text-to-SQL, candidates are compared by
executed result rather than by text. (Ch 13, 36)

**Self-hosting**: Running open-weights models on your own infrastructure instead of calling a hosted API.
Driven by data control, cost at scale, or latency, and paid for in operations work. (Ch 34)

**Semantic cache**: A cache that returns a stored answer for queries whose embeddings are similar to a
previous query. It risks answering a question nobody asked when the threshold is loose. (Ch 15, 30)

**Semantic conventions (GenAI)**: OpenTelemetry's standard attribute names for model calls, such as
model, token counts, and operation. Using them keeps traces portable across tools. (Ch 23, 31)

**Semantic layer**: A curated model of business metrics and entities over raw tables. Text-to-SQL
against a semantic layer is safer and more accurate than against raw schemas. (Ch 36, 37)

**Semantic memory**: Stored facts and knowledge about the user, domain, or world, independent of when
they were learned. (Ch 21)

**semsearch**: The package of Project 2: a `VectorStore` protocol with NumPy and pgvector
implementations, namespaces per embedding model and index version, filtered and hybrid search, and
retrieval metrics. Later RAG code reuses it rather than re-creating a store. (Ch 9)

**Server-sent events (SSE)**: A one-way HTTP streaming format of text events. Model providers stream
token deltas over it, and many AI backends use it to stream answers to browsers. (Ch 3, 28)

**SFT (Supervised Fine-Tuning)**: Training on input and target output pairs so the model imitates the
targets. The base method behind most task fine-tuning. (Ch 33)

**Shadow traffic**: Sending a copy of production requests to a new version without returning its
responses to users, to compare behavior safely. (Ch 32)

**Side-effect class**: The book's classification of tools as read-only, reversible write, irreversible,
or external. The class determines retries, approval, and idempotency requirements. (Ch 16, 19)

**Slice analysis**: Breaking evaluation results down by segment (language, tenant, document type, tag).
Aggregate scores hide regressions on important slices. (Ch 14, 24)

**Sliding-window attention**: Attention layers that attend only to the last W tokens and so keep at most
W tokens of KV cache regardless of context length. Models interleaving local and global layers need
far less cache than the naive formula, if the engine implements the window. (Ch 34)

**SLO (Service Level Objective)**: A target for a measured indicator, such as p95 time to first token
under 2 seconds for 99 percent of minutes. SLOs turn reliability into budgets and alerts. (Ch 29, 34)

**Small language model (SLM)**: A model small enough to run cheaply or locally, often good enough for
classification, extraction, and routing. (Ch 2, 7)

**Softmax**: The function that converts logits into a probability distribution. Temperature divides the
logits before softmax. (Ch 2)

**Span**: One timed operation within a trace, such as a retrieval, a model call, or a tool call, with
attributes and status. (Ch 31)

**Speculative decoding**: Using a fast draft model to propose several tokens that the large model verifies
in one pass. It reduces latency without changing the output distribution when implemented exactly.
(Ch 34)

**SQL guard**: A validator that parses model-generated SQL and approves only a single read-only query
over allowlisted tables, without blocked columns or functions, with a bounded LIMIT. It complements, and
never replaces, a read-only database role with row-level security. (Ch 36)

**Stable prefix**: Placing unchanging content (system instructions, tool definitions) at the start of the
prompt so prompt caching applies across requests. (Ch 5, 30)

**Stage isolation**: Evaluating each RAG stage separately to find where the evidence was lost: not
ingested, not retrieved, ranked out, not packed, or ignored by the generator. (Ch 14)

**Stop sequence**: A string that ends generation when the model emits it. One of the stop conditions
together with the token limit and the model's own end token. (Ch 2)

**Streaming**: Delivering output incrementally as tokens are generated. It improves perceived latency
and complicates validation, citations, and error handling mid-stream. (Ch 3, 13, 28, 30)

**Structured output**: Model output constrained to a schema, usually JSON, via provider schema modes,
tool calling, or constrained decoding, then validated in code. (Ch 3, 6)

**Supervisor-worker**: An architecture in which a supervisor decomposes a task, delegates parts to
worker agents, and integrates their results. (Ch 20, 22)

**Sycophancy**: A model's learned tendency to agree with the user and sound confident, accepting false
premises and abandoning correct answers under pushback. It comes from preference tuning and is
countered with grounding, neutral prompts, and evaluation cases that push back. (Ch 2, 20)

**Synthetic data**: Evaluation or training examples generated by a model, for example questions written
from chunks. Fast to produce, biased toward what the generator finds easy, and in need of validation.
(Ch 14, 25)

**System design method (ten steps)**: Requirements, architecture, models, retrieval, tools, memory,
security, evaluation, scaling, and failure modes, in that order, each producing a reviewable
artifact. Later steps may send you back; you never start at step 3. (Ch 35, 36)

**System prompt**: The highest-priority instructions in a request, set by the application, defining
role, rules, and output format. (Ch 3, 4)

## T

**Temperature**: A sampling parameter that sharpens (low) or flattens (high) the probability
distribution. Low temperature for extraction and classification, higher for varied generation. (Ch 2)

**Tenant isolation**: Guaranteeing that one tenant's data, memory, caches, and tool access never reach
another tenant. Enforced in queries and keys and verified with tests. (Ch 26, 27, 29)

**Termination condition**: A rule that ends an agent run: success validated, step or budget limit,
timeout, repeated state, no progress, or approval required. (Ch 19)

**Text-to-SQL**: Translating a natural-language question into a SQL query. Needs read-only credentials,
query validation, and result verification. (Ch 36, 37)

**Thinking tokens**: The intermediate tokens a reasoning model generates before its answer. They are
decoded like any other output, occupy KV cache, and are usually billed as output, so they must be
counted from the provider's usage field. (Ch 2)

**Threat model**: A structured account of assets, actors, trust boundaries, and threats for a system, used
to choose controls. (Ch 26)

**Time per output token (TPOT)**: The average time between output tokens after the first. It sets how
fast a streamed answer appears and is bound by decode speed. (Ch 2, 34)

**Time to first token (TTFT)**: The time from sending a request to receiving the first output token. It
includes queueing and prefill and is the latency users feel most. (Ch 2, 30, 34)

**Token**: The unit a model reads and writes, typically a word piece. Cost, latency, and context limits
are all measured in tokens. (Ch 2)

**Token bucket**: A rate-limiting algorithm in which tokens refill at a fixed rate and each request
spends some. It allows short bursts while enforcing an average rate. (Ch 3, 29)

**Tokenizer**: The component that converts text to token ids and back. Different models use different
tokenizers, so token counts are model-specific. (Ch 2)

**Tool calling**: The protocol by which a model emits a structured request to call a named function with
arguments, and the application executes it and returns the result. The model proposes; code decides.
(Ch 3, 16)

**Tool poisoning**: Malicious instructions embedded in a tool's description or results, read by the model
as guidance. The rug pull variant serves a clean description at review time and a poisoned one later;
pinning reviewed descriptions by fingerprint detects it. (Ch 18, 26)

**Tool policy**: Code that decides whether a proposed tool call may run, based on identity, permissions,
argument constraints, side-effect class, rate limits, and approvals. (Ch 16, 27)

**Tool registry**: The catalog of tools with their schemas, side-effect classes, and handlers, from which
a request's allowed tool set is selected. (Ch 16)

**Tool schema**: The name, description, and JSON Schema for a tool's parameters. Narrow schemas with
enums and clear descriptions improve tool selection and limit misuse. (Ch 16)

**toolkit**: The book's governed tool-calling library: registry, policy, idempotency store, sandbox
runner, and audit. (Ch 16)

**Top-k sampling**: Sampling only from the k most probable next tokens. (Ch 2)

**Top-p sampling**: Sampling from the smallest set of tokens whose cumulative probability exceeds p, also
called nucleus sampling. (Ch 2)

**Trace**: The full tree of spans for one request or agent run. For AI systems it must include the exact
context and versions to make semantic failures debuggable. (Ch 31)

**Trajectory evaluation**: Scoring an agent's sequence of steps, not just its final answer: tool
correctness, efficiency, policy violations, and whether required steps happened. (Ch 25)

**Transformer**: The neural network architecture behind modern LLMs, built from stacked attention and
feed-forward blocks. (Ch 2)

**Trust boundary**: A line across which data moves from less trusted to more trusted parts of a system.
In AI systems every retrieved document, tool result, and MCP server sits on the untrusted side. (Ch 26,
28, 35)

## U

**User profile memory**: Durable facts and preferences about a user, written under a policy and
deletable on request. (Ch 21)

## V

**Vector database**: A system specialized for storing and searching embeddings with ANN indexes and
metadata filters. Often unnecessary at modest scale, where PostgreSQL with pgvector or exact search
suffices. (Ch 9)

**Vectorless retrieval**: Retrieval without embeddings: SQL, metadata filters, full-text search, or
navigating a document hierarchy. Often better for structured questions. (Ch 37)

**Voice agent**: An agent that converses in speech, with streaming speech recognition and synthesis and a
tight latency budget per turn. It must handle barge-in: stop audio, cancel in-flight generation, and
treat the interruption as new input. (Ch 35, 38)

## W

**Worker**: A process that pulls jobs from a queue, executes them idempotently, and acknowledges or
retries them. Also a subordinate agent under a supervisor. (Ch 22, 28, 29)

**Workflow**: A predefined sequence or graph of steps, some of which call models, where code decides the
path. The book's default control structure before agents. (Ch 17)

**Working memory**: The information held in the current context window for the task at hand. (Ch 2, 21)

**Write policy (memory)**: Rules that decide what may be written to long-term memory, with what
provenance and confidence. It blocks secrets, unapproved PII, and injected instructions. (Ch 21)
