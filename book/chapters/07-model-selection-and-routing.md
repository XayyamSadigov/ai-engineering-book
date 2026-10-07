# Chapter 7 — Model Selection and Routing

This chapter is about choosing a model for a workload from evidence rather than reputation, and about building the component that makes that choice again for every request. Getting it wrong rarely breaks anything visibly: the bill grows, or the hardest requests get worse while the average looks fine.

**You will be able to:**
- Record each model's capabilities, prices, and data zone in a catalog that application code reaches only through pinned aliases.
- Run one evaluation set across candidates and choose from the Pareto front, using interval lower bounds and paired counts.
- Decide between a hosted API, a managed open-weight endpoint, and self-hosting from cost at volume, data control, and operating burden.
- Build a router that combines policy rules, an optional classifier, and a confidence cascade, and that refuses fallbacks which would quietly break a request.
- Choose a cascade threshold by end-to-end utility, pricing in what a misrouted request costs.
- Diagnose routing incidents (escalation storms, hidden fallbacks, retired pins) from router telemetry.

**Prerequisites:** Chapters 3 and 6 (the `aie_core` client, gateway, and pricing table; confidence signals and calibration). | **Code:** `book/projects/examples/ch07/` (run: `cd book/projects/examples/ch07 && pytest -q`) | **Builds:** the chapter's model catalog, selection harness, `Router`, and cascade evaluator, running offline against `FakeLLM` instances that act as a small, a general, and a reasoning model on the Northwind ticket set.

## Why this matters

Model choice is often the largest lever on the cost and latency of an AI feature, and it is usually pulled once, early, by intuition. A team prototypes with the strongest model it can reach, the prototype works, and that model becomes the production default for every request: the password-reset question, the 400-page contract, the ticket that only needs a category. Months later the bill arrives, or the p95 latency target is missed, and nobody can say which requests needed the expensive model, because nothing measured it.

The opposite mistake is just as common and harder to see. A team reads that a small model is "nearly as good" on a public benchmark, switches, and watches the aggregate quality score barely move. Meanwhile the hardest tenth of traffic, the requests where a wrong answer creates a human escalation or a customer complaint, quietly gets worse. Averages hide that, and benchmarks measure someone else's distribution.

Three facts make selection an engineering problem. Workloads are mixed: Northwind Assist serves ticket classification (short input, high volume), cited policy answers, incident research (long context, tools), and occasional high-stakes refund exceptions, and no single model is the best trade-off for all four. Models differ in contracts, not only quality: window, tool calling, schema output, image input, and where inference may run. A request that needs a missing capability fails or, worse, succeeds wrongly. And models change underneath you, as providers update, rename, and retire them.

The answer is machinery most teams build eventually and usually too late: a catalog of what each model can do, a harness that measures candidates on your data, a router that decides per request, and an evaluation that treats the router as part of the system's quality. This chapter builds all four.

## Mental model

> **Mental model:** A model is a dependency with a contract. Choose it per request, from evidence, and treat the choice itself as code that can be wrong.

The contract part means you write down what each model guarantees (window, tools, schema mode, modalities, data zone) and check it before every call, the way you would check an API version. The evidence part is the book's eighth mental model applied directly: model quality alone does not determine application quality, so selection comes from a task-level evaluation, not a leaderboard. The last part is the one teams skip. A router is a classifier with a cost function. It makes errors, its errors have prices, and routing quality is a component of end-to-end quality. A cheap model that handles easy traffic well can still lose money overall if the router sends it the wrong requests.

## Core concepts

### The selection axes

Ten properties decide whether a model fits a workload. They split into two groups. Hard constraints remove candidates outright. Trade-off axes are measured and balanced.

| Axis | Kind | What to measure or check | What happens if you ignore it |
|---|---|---|---|
| Task capability | trade-off | pass rate on your evaluation set, with an interval | you ship on benchmark reputation |
| Reasoning depth | trade-off | pass rate on the hard slice, per effort setting | hard cases fail silently or easy ones overpay |
| Latency | trade-off | p50 and p95 end to end, time to first token for streaming | SLO misses that no prompt change can fix |
| Price | trade-off | cost per request and per correct answer, at your token mix | the bill scales with traffic, not with value |
| Context window | hard | prompt plus output tokens versus the window | truncation, rejected requests, or lost-in-the-middle at the edge |
| Structured-output quality | hard or soft | schema validity rate, native schema mode or not | repair loops, parse failures in production |
| Tool use | hard | tool calling supported, argument validity rate | tool-using prompts answered without tools |
| Multimodality | hard | image or audio input supported, perception accuracy | images silently dropped or misread |
| Reliability and availability | trade-off | error rate, rate-limit headroom, regional outages | a single provider incident takes the feature down |
| Data residency | hard | where inference runs, retention terms | regulated data leaves its permitted zone |

Two of these deserve a note. Price means cost per correct answer, not per token. A model that is cheaper per token but writes three times as many output tokens, or needs a repair round trip twice as often, can cost more per correct answer, which is the number that matters. And reliability is a property of the deployment, not the weights: the same model behind two providers, or self-hosted versus hosted, has different availability, rate limits, and latency tails.

### Model classes as workload shapes

Chapter 2 introduced the model classes. For selection, what matters is the workload each one fits.

A **small model** (often called an SLM, small language model) is the right default for narrow, stable, high-volume tasks: classification, extraction into a fixed schema, intent routing, guard checks. Small does not mean weak on every task. A compact model prompted or fine-tuned for one well-defined job often matches a general model on that job at a fraction of the cost and latency, and it can run on hardware you control, which matters for privacy. Its weakness is breadth: unusual inputs, multi-topic requests, and anything needing world knowledge it was not trained on.

A **general model** is the default when the shape of the task is not known in advance: open-ended questions, drafting, synthesis across sources.

A **reasoning model** spends extra inference compute on multi-step problems. It earns its cost on planning, multi-constraint decisions, mathematics, and hard debugging. It wastes it on classification and extraction, where the answer does not get better with more deliberation.

A **decision-style model** returns a typed decision (a class, a score, a probability) rather than prose. For routing, moderation, eligibility, and triage, a bounded output is cheaper, easier to validate, and can be calibrated. Do not use a text generator when the product contract is a decision.

A recurring selection question is whether a small model with retrieval beats a reasoning model. It does when the hard part of the task is knowledge access rather than reasoning: the task is bounded, the answer is in your documents, latency and cost matter, and your evaluation shows the small model is sufficient once it has the evidence. Reasoning effort cannot compensate for missing facts, and retrieval cannot compensate for a problem that genuinely needs several dependent inference steps.

Newer classes keep arriving. Recursive approaches let a model query a huge input piece by piece instead of loading it whole, separating the available corpus from the active context. Diffusion-style models refine many positions in parallel, which changes the latency profile. Keep the router open to new classes, but admit one only after it passes the same evaluation as everything else, and record the evidence level of every claim: a vendor report, a preprint, and an independent evaluation on a workload like yours are three different things.

### Reasoning effort as a knob

Many models now expose a reasoning budget: an effort level, a thinking-token limit, or a separate reasoning mode. This turns inference compute into a per-request parameter. The same model asked to think longer gets better on hard multi-step problems and worse on latency and cost, often by large factors, because the extra deliberation is billed and generated serially as output tokens.

Treat effort as a route attribute, not a global setting. The Northwind high-assurance route uses high effort for refund exceptions; the general route uses none; a classifier-selected reasoning route uses medium. Three rules keep the knob honest. Evaluate the outcome, not the length of the reasoning: long chains can be confidently wrong, and visible deliberation is not evidence of correctness. Cap the budget: an effort setting without a token ceiling is an unbounded cost on adversarial or degenerate inputs. And measure each effort level as a separate candidate in the selection harness, because "the reasoning model" at low and high effort are two points on the Pareto front (defined below), not one.

The `aie_core` request type has no effort field, because providers expose it differently. The router passes `reasoning_effort` in `req.metadata`, and only to models whose catalog profile lists the requested level; adapters that support the knob read it from there. A request that asks for an effort level the model does not offer, including a model with no knob at all, gets a soft warning instead of a silent drop.

### Multimodal inputs

Images, scans, and audio enter the model as extra tokens produced by a modality-specific encoder, so they consume context, cost, and latency like text. Selection for multimodal work has three parts beyond "does it accept images."

First, preprocess before you pay. Resize, crop to the region of interest, sample video frames, or run OCR or object detection first when that reduces cost without losing evidence. A photographed invoice often goes through OCR and a text model more cheaply and accurately than through a vision model; the selection harness can tell you which. Second, preserve provenance: keep a reference to the exact image, page, frame, or region so the answer can point at it. Third, evaluate perception and reasoning separately. An answer can be wrong because OCR misread a digit, because the model grounded on the wrong region, or because it reasoned badly from correct features, and one end-to-end score cannot tell these apart.

Multimodal inputs also widen the attack surface: text hidden in images, adversarial perturbations, manipulated metadata, and instructions embedded in a screenshot or a QR code (Chapter 26). For the router, an image part in the request is a hard requirement: the request goes only to models whose profile has `supports_vision`, and a text-only fallback is never acceptable, because it would answer without seeing the input.

### Hosted API, managed open-weight, or self-hosted

Every candidate also comes with a deployment choice, and that choice moves several axes at once. There are three common options. A **hosted API** serves a proprietary model behind a provider's endpoint, billed per token. A **managed open-weight endpoint** serves a model whose weights are published, run by a cloud or inference provider and billed per token or per GPU-hour; you could take the same weights elsewhere, and you do not run GPUs. **Self-hosting** runs open weights on hardware you control, behind an inference server such as the ones Chapter 34 covers.

| Factor | Hosted API | Managed open-weight | Self-hosted |
|---|---|---|---|
| Licensing | provider terms of service | the model's license plus provider terms | the model's license; some restrict use, scale, or derivatives |
| Cost at volume | per token, scales linearly with traffic | per token or per hour | mostly fixed: GPUs cost the same idle or busy |
| Data control | provider's retention terms and regions | provider's regions, often configurable | data never leaves your network |
| Fine-tunability | only what the provider offers | often adapters, sometimes full | full control (Chapter 33) |
| Operations burden | none beyond the client | low | high: capacity, upgrades, on-call, quantization checks |
| Versions | provider retires pins on its schedule | provider may retire hosted versions | you keep weights as long as you like |
| Strongest available quality | usually here | often behind | often behind |

The cost row is where teams go wrong, in both directions. A hosted price scales with tokens. A self-hosted node costs the same per hour whether it is busy or idle, so its cost per token depends on utilization. Illustrative numbers: a GPU node at $8 per hour that sustains 2,000 output tokens per second at full load produces 7.2 million tokens an hour, about $1.10 per million at full utilization. At 25% average utilization the same node costs about $4.40 per million, roughly the general model's illustrative output price, before counting the engineers who keep it running. Compare at your measured utilization, including nights and weekends, never at peak throughput (Chapter 34 does the serving math in detail).

Self-hosting is justified when data must not leave your network (Northwind's `nw-small` runs on-premises for exactly this reason), when a narrow, high-volume task runs steadily enough to keep GPUs busy and a small model passes the evaluation, or when you need control over weights for fine-tuning or for keeping a version indefinitely. It is a poor choice for spiky or low-volume traffic, for tasks that need the strongest available model, and for teams with no one to own GPU capacity. A managed open-weight endpoint is the middle path: open weights and portability without the operations, and a common first step before self-hosting.

Two failure patterns recur. A model evaluated at full precision is deployed quantized to fit the hardware, and quality drops on the hard slice; evaluate the exact deployment, quantization and server version included. And an inference server upgrade changes outputs (a new default sampling setting, a different chat template) without anyone touching the model; pin the server version alongside the weights. In the catalog, deployment is part of the profile: the same weights served by a provider and by you are two aliases, two candidates in the harness, and two rows on the Pareto front, because reliability, latency, data zone, and price all belong to the deployment.

### Evaluation-driven selection

The selection procedure is the same whether you are choosing a first model or deciding whether to switch.

1. **Write the task contract.** Inputs, outputs, the metric that defines correct, and the cost of each kind of error. For ticket classification: exact category match, and a wrong category costs a re-triage.
2. **Build the evaluation set from your workload.** Real or realistic requests, with the hard and rare slices represented. Sixty tickets is enough to start and too few to separate close candidates, as the intervals below show (Chapter 24 covers sample size).
3. **Fix the constraints.** Latency SLO, cost ceiling, data zone, required capabilities. These come from the product, not from the candidates.
4. **Filter candidates by hard constraints.** A model that cannot run in the required zone or lacks a required capability is not a candidate, however good its scores.
5. **Run every candidate on the same set in the configuration you will ship:** same prompt, same parameters, same effort setting. Record quality, latency percentiles, cost, and failures per case.
6. **Read the Pareto front.** The Pareto front is the set of candidates you cannot improve on one axis without losing on another. Discard the rest, the dominated candidates: anything another candidate beats or matches on quality, cost, and latency at once.
7. **Choose the cheapest candidate that meets the quality floor on the lower bound of its interval**, not on its point estimate, and within the latency ceiling.
8. **Compare close candidates with paired counts.** On the same cases, how many does A get right that B gets wrong, and the reverse? Two models with similar averages can fail on entirely different cases, which is exactly the situation where routing pays.
9. **Record the decision** (set version, prompt version, model versions, results) and re-run it on every model, prompt, or traffic change.

Here is the harness's output on the Northwind ticket set, with illustrative prices and latencies:

| candidate | quality (95% CI) | p50 ms | p95 ms | $/request | $/correct | errors | Pareto |
|---|---|---|---|---|---|---|---|
| nw-reasoning | 0.950 (0.86-0.98) | 6500 | 6500 | 0.000928 | 0.000977 | 0.0% | yes |
| nw-general | 0.933 (0.84-0.97) | 1800 | 1800 | 0.000186 | 0.000199 | 0.0% | yes |
| nw-small | 0.800 (0.68-0.88) | 250 | 250 | 0.000019 | 0.000023 | 0.0% | yes |

The 95% interval is a Wilson score interval on the pass rate; it stays honest at small sample sizes and near 0 or 1, where the textbook normal approximation does not (Chapter 24 covers intervals and paired tests in depth). The fakes report fixed latencies, so p50 equals p95 here; a real bake-off shows a gap between them, and the gap is often what disqualifies a candidate.

All three are on the front: each is the best at something. The intervals overlap heavily between the reasoning and general models, so sixty cases cannot say the reasoning model is better at classification; they can say it costs five times as much and takes more than three and a half times as long. With a quality floor of 0.80 on the lower bound and a p95 ceiling of 3 seconds, the procedure picks `nw-general`: the small model's point estimate meets the floor but its lower bound (0.68) does not.

The harness also prints paired counts: the small model gets two tickets right that the general model misses, and the general model gets ten right that the small one misses. Neither model's correct set contains the other's, and most of the small model's errors are concentrated in a recognizable slice: multi-topic tickets whose obvious keywords point the wrong way. That is the signature of a workload where a cascade (cheap model first, escalate when unsure; see Routing strategies) might beat both single-model options. Whether it does depends on what an error costs, which is the subject of the cascade evaluation below.

### Routing strategies

Routing chooses the model, provider, or pipeline for each request. The strategies form a ladder of increasing power and decreasing explainability.

**Static mapping by use case.** Each endpoint or task type has a model, chosen by the procedure above. This is where every system should start, and many never need more. It is trivially explainable and testable.

**Rules.** Deterministic predicates over request features that code already knows: task type, tenant, data zone, risk flag, prompt size, presence of tools or images, customer tier. Rules encode policy (restricted data stays on-premises) and obvious economics (classification goes to the small model). They are cheap, fast, and auditable. Put policy rules before cost rules, so a cost optimization can never override a compliance requirement.

**Classifiers.** A small model predicts difficulty, domain, or route from the request text. This handles traffic that rules cannot separate, such as general questions that are sometimes easy and sometimes need reasoning. A classifier must abstain: when its confidence is low, the request takes the default route rather than a guess.

**Embedding similarity.** Embed labeled example requests per route, compute a centroid per route, and send each request to the nearest centroid, with the margin over the runner-up as confidence. It needs a few dozen examples instead of a trained model, and requests between two centroids visibly report low margin. Chapter 8 covers the embedding side; the router implements it as `EmbeddingRouteClassifier`.

**Model-as-router.** A language model reads the request and picks a route. It handles nuance, but adds a full model call to every request and is the easiest router to manipulate through the request text. Use it only when cheaper strategies have measurably failed, and map its output to an enumerated route in code.

**Confidence cascades.** Instead of predicting difficulty up front, run the cheap model first and escalate when its answer looks unreliable: low confidence, a failed validator, or an output outside the allowed set. A cascade learns difficulty from the attempt itself, which is often more accurate than predicting it from the input, at the cost of paying for the cheap call on every escalated request and adding its latency to the escalated path.

Each strategy can choose along more than one dimension. A route names a model, but it can also name an effort level, so "the reasoning model at medium effort" and "the reasoning model at high effort" are different routes, and the cheapest escalation is sometimes more effort on the same model rather than a different model.

Real routers combine these. The Northwind router evaluates rules first, then an optional classifier, then a default route, and any route can carry an escalation target that turns it into a cascade.

### Confidence signals and calibration

A cascade is only as good as the signal that decides escalation. The candidates, from cheapest to most expensive:

- **Self-reported confidence.** A confidence field in the output. Free, and often miscalibrated.
- **Token log-probabilities.** Where exposed, the probability of the chosen label token is better founded than a self-report.
- **Validators.** Deterministic checks: the output parses, the label is in the allowed set, the extracted total equals the sum of line items. A validator failure is the most reliable trigger there is, because it is not a guess.
- **Agreement.** Sample the cheap model twice, or ask two cheap models, and escalate on disagreement. It doubles the cheap cost and misses consistently wrong answers.
- **A learned verifier.** A small model trained to predict whether an answer is correct. The most accurate and the most work to maintain.

Chapter 6 explains these signals in depth and owns calibration: a reliability table bins cases by confidence and compares each bin's mean confidence with its accuracy, and the expected calibration error (ECE) is the count-weighted average gap. What matters for routing is to run that table on your own distribution, for the exact signal the cascade uses. On the Northwind set, the small model's self-reported confidence has these bins (illustrative):

| confidence bin | cases | mean confidence | accuracy |
|---|---|---|---|
| 0.2-0.4 | 1 | 0.20 | 0.00 |
| 0.4-0.6 | 7 | 0.52 | 0.71 |
| 0.6-0.8 | 15 | 0.76 | 0.73 |
| 0.8-1.0 | 37 | 0.96 | 0.86 |

ECE is about 0.09. The signal is informative (higher bins are more accurate) but overconfident at the top. The top bin holds five wrong answers, four of them at 0.97, the highest confidence the model ever reports. No threshold short of escalating everything will catch those four, and one of them is wrong on the large model too. Those confident errors are the number to watch in cascade design: escalation can only fix errors the signal can see. Confident mistakes pass straight through, and their price decides whether the cascade is worth building.

### Routing error cost

A router makes two kinds of mistakes. A **false accept** (false "easy") lets a cheap model's wrong answer through. It costs quality, often silently, or a retry if something downstream catches it. A **false escalation** (false "hard") sends a request the cheap model had answered correctly to the expensive model anyway. It costs money and latency, never quality. Router accuracy, the share of requests where the escalate-or-not decision was right, counts these two equally. Their prices are nothing alike: at the illustrative Northwind prices a false escalation costs about $0.00017 extra, while a silent wrong answer can cost a re-triage worth hundreds of times that.

The correct objective is end-to-end utility per request, in one currency:

```text
# pseudocode
utility = value of correct answers
        - cost of silent errors (false accepts nobody catches)
        - cost of caught errors (rework plus the re-run on the strong model)
        - model spend on every stage that ran
        - cost of latency, if waiting has a price
```

False accepts land in the two error terms; a false escalation shows up only as extra model spend and latency. The misroute cost belongs in this sum even when the cheap answer is eventually fixed. A cheap model that fails, is caught by a validator downstream, and triggers a retry on the strong model costs both calls plus the rework, which can exceed calling the strong model once. That is the warning attached to every routing recipe: evaluate misroutes, not just routes.

The cascade sweep on the ticket set makes this concrete. Running each model once and simulating every threshold offline gives, for three illustrative prices of a silent error (the value of a correct answer and the caught share stay fixed):

| cost of a silent error | best policy by utility | accuracy | $/request | escalated |
|---|---|---|---|---|
| 0.0002 | always small | 0.800 | 0.000019 | 0% |
| 0.001 | cascade, threshold 0.78 | 0.883 | 0.000066 | 25% |
| 0.05 | always large | 0.933 | 0.000186 | 100% |

Three prices, three different winners. When errors are nearly free, the cheap model's savings dominate. In a middle band, the cascade beats both single models. When errors are expensive, the small model's confident mistakes cost more than every escalation saves, and calling the large model directly wins: the best cascade escalates everything, which makes it the large model plus a wasted small call on every request. Meanwhile the threshold that maximizes router accuracy is the same, about 0.53, in all three regimes, because router accuracy ignores prices.

### Fallbacks that change assumptions

A fallback is a second model used when the first is unavailable or unfit. The gateway in Chapter 3 already falls back on retryable errors. What the gateway cannot know is whether the fallback can actually serve this request. Fallbacks change assumptions in ways that break requests:

- **Context length.** The primary has a 200k-token window and the fallback 32k (illustrative sizes). Falling back either fails with a context error or, if some layer truncates to fit, answers from a prompt that lost its evidence.
- **Tool support.** A tool-using request sent to a model without tool calling gets a confident prose answer that never looked anything up.
- **Structured output.** The fallback lacks a native schema mode. The request can still work through prompt-and-parse with validation and repair (Chapter 6), but its malformed-output rate changes.
- **Data residency.** The only model allowed for restricted data is on-premises. Falling back to a cloud model during an outage is a compliance incident, not a reliability feature.
- **Reasoning effort, tokenizer, and prompt format.** The fallback has no effort knob, counts tokens differently, or follows the system prompt differently. A prompt tuned for one model family can underperform on another (Chapter 4).

The router therefore classifies every gap as hard or soft. Hard gaps (context, output length, tools, vision, data zone) remove a model from the candidate list. Soft gaps (no native schema mode, unsupported effort level) keep it but attach a warning to the decision, so a dashboard can show how often requests ran under changed assumptions.

Two fallbacks remain when the plan fails. When every planned model has a hard gap, the router looks across the catalog, among models that have a registered client, for the cheapest compatible one and records the substitution. A model whose data zone is `any` makes no residency promise, so it never satisfies a request that requires a specific zone, and the substitution can never pick it for an on-prem request. When nothing is compatible, it raises `NoCompatibleModelError` instead of sending the request somewhere it cannot succeed. The caller can then shrink the context, drop the tools, or tell the user, all better than a wrong answer.

### Vendor neutrality and model pinning

Every model reference in application code should be an alias owned by the catalog (`nw-small`, `nw-general`), never a provider's model name. The catalog resolves each alias to a pinned, versioned identifier. Swapping a provider or version becomes a one-line, reviewed catalog change. A provider updating a floating name cannot change your behavior without a deploy. And every trace records the exact version that served the request, so a regression can be tied to a version change.

Repointing an alias is a release: run the selection harness on the new version, compare per-category deltas, canary it (serve it to a small share of traffic first), and keep the old pin for rollback (Chapter 32). Provider-neutral `aie_core` types keep the router independent of any vendor SDK. Track deprecation dates for every pin, because a retired pin becomes an outage on a date you were told about months earlier.

A migration off a retired or superseded pin follows a fixed procedure, and it starts weeks before the retirement date:

1. **Add the new version as a separate alias** (`nw-general-next`) in the catalog, with its own profile. Capabilities change between versions too: a new version can have a different window, output limit, or schema support, and the router's capability check only protects you if the profile is accurate.
2. **Run the selection harness** with both pins on the current evaluation set, in the shipped configuration. Read per-slice deltas and paired counts, not only the overall score, and re-run the cascade sweep if the alias is a cascade stage: a new version's confidence distribution can move every threshold.
3. **Port the prompt deliberately.** Prompts tuned on one version can regress on its successor, so every prompt that the alias serves runs its regression suite against the new pin before anything ships (Chapter 4). If the new version needs a retuned prompt, evaluate the pair offline, then roll the changes out one at a time where possible, so a regression can be attributed to the prompt or the model rather than to both.
4. **Shadow, then canary.** Send a sample of live traffic to the new alias without serving its answers, compare format-error rate, latency, and judged quality, then serve a small share and watch the route's telemetry.
5. **Repoint the alias, keep the old pin** until the retirement date, so rollback is a catalog change rather than an emergency.

Error mapping matters on retirement day. A provider usually rejects a retired model with a "model not found" class of error, which `aie_core` maps to a non-retryable `InvalidRequestError`: the route fails loudly and an alert fires. That is the correct behavior, because a missing model will not come back on retry. Any layer that turns it into a retryable error, such as an internal proxy that answers 503 for every upstream failure, converts the outage into a silent fallback on every request, which shows up only as a cost increase.

## How it works

Follow one Northwind request through the router.

1. **Requirements are derived from the request, not declared by the caller.** `Requirements.from_request` counts prompt tokens plus `max_tokens`, and detects tools, a response schema, and image parts; data zone and effort come from metadata. A caller cannot forget to declare that its prompt is 150k tokens.
2. **A route is selected.** Rules run in order: restricted data, high risk, long context, narrow task. If none matches and a classifier is configured, the classifier predicts a route; a prediction below its confidence floor falls through to the default route. The decision records which stage chose the route.
3. **The route becomes a candidate list.** The primary model and its fallbacks are checked against the requirements. Models with hard gaps are skipped, each with a recorded reason. If none survive, the cheapest compatible model in the catalog that has a client is substituted; if there is none, the request fails with `NoCompatibleModelError`. Soft gaps become warnings. The escalation target, if any, is checked too and disabled if incompatible.
4. **Candidates are called in order.** Each call goes through that alias's client, which in production is a `ModelGateway` with its own retries. The router sets `req.model` to the pinned identifier and passes the effort level where supported. A retryable error moves to the next candidate; a non-retryable one is raised, because a bad request will fail on every model.
5. **The answer is checked for escalation.** If the route has an escalation target that did not already serve the request, the validator runs first and the confidence function second. A failed validator or low confidence triggers one call to the stronger model. If that call fails, or the escalation target was disabled because it cannot serve this request, the router returns the cheap answer marked `degraded`, so the caller knows it is below the route's quality bar.
6. **Everything is recorded.** The result carries the decision, every attempt with its outcome, latency, cost, and confidence, and the model that finally served. The router also compares the model the completion reports with the pin it planned for; a difference means something below the router, usually a gateway with its own fallback list, served the request from a model the capability check never saw. A tracing span carries the route, stage, serving alias and the model id the completion reported, attempts, warnings, escalation, degradation, mismatch, and cost as attributes.

## Architecture

The router's decision and execution path, with the hard capability check between selection and execution:

```mermaid
flowchart TD
    REQ["CompletionRequest"] --> NEED["derive Requirements"]
    NEED --> RULES{"rule matches?"}
    RULES -- yes --> ROUTE["route"]
    RULES -- no --> CLF{"classifier confident?"}
    CLF -- yes --> ROUTE
    CLF -- no --> DEF["default route"] --> ROUTE
    ROUTE --> CAP["filter primary and fallbacks by hard gaps"]
    CAP --> ANY{"any compatible?"}
    ANY -- no --> SUB{"catalog has one?"}
    SUB -- no --> FAIL["NoCompatibleModelError"]
    SUB -- yes --> CALL
    ANY -- yes --> CALL["call candidates in order"]
    CALL -- retryable error --> CALL
    CALL -- answer --> ESC{"escalation target and low confidence or invalid?"}
    ESC -- no --> OUT["RoutedCompletion"]
    ESC -- yes --> STRONG["call strong model"]
    STRONG --> OUT
```

The cascade on one ticket, showing why escalated requests pay for both stages in money and latency:

```mermaid
sequenceDiagram
    participant App as Application
    participant R as Router
    participant S as nw-small
    participant V as Validator and confidence
    participant L as nw-general
    App->>R: classify ticket
    R->>S: request, model small-instruct-2026-03
    S-->>R: category hardware, confidence 0.41
    R->>V: check answer
    V-->>R: valid, below 0.80
    R->>L: same request, model general-2026-02
    L-->>R: category vpn_network, confidence 0.93
    R-->>App: answer from nw-general, two attempts, cost of both
```

Selection and routing are two halves of one loop. Offline evaluation sets the catalog and the thresholds; production telemetry feeds new cases back into the evaluation set:

```mermaid
flowchart LR
    subgraph Offline
        SET["evaluation set"] --> HARNESS["selection harness"]
        HARNESS --> FRONT["Pareto front"]
        FRONT --> CAT["catalog pins"]
        SET --> SWEEP["cascade sweep"]
        SWEEP --> THR["thresholds"]
    end
    subgraph Online
        ROUTER["router"] --> TRACE["route, model, escalation, cost per request"]
    end
    CAT --> ROUTER
    THR --> ROUTER
    TRACE --> SAMPLE["sampled and failed cases"]
    SAMPLE --> SET
```

## Implementation

```text
book/projects/examples/ch07/
├── catalog.py        ModelProfile, Requirements, capability gaps, ModelCatalog
├── catalog.json      the illustrative Northwind catalog as data
├── tasks.py          ticket-classification cases, label schema, scorer, confidence reader
├── fakes.py          make_small_model and make_large_model: FakeLLM handlers
├── selection.py      run_selection, Wilson intervals, percentiles, Pareto front, choose
├── router.py         Router, Route, Rule, EmbeddingRouteClassifier, Northwind routes and rules
├── cascade_eval.py   collect_outcomes, simulate, sweep, utility, calibration
├── config.py         RouterSettings from environment variables
├── demo.py           selection table, cascade sweeps, routing decisions
├── test_ch07.py      offline tests
├── pyproject.toml
├── .env.example
└── README.md
```

Install and run from the repository root:

```bash
uv pip install --python .venv/bin/python -e book/projects/aie_core   # or: pip install -e book/projects/aie_core
.venv/bin/python -m pytest book/projects/examples/ch07 -q
cd book/projects/examples/ch07 && ../../../../.venv/bin/python demo.py
```

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_CATALOG_PATH` | unset | JSON catalog; the built-in illustrative catalog when unset |
| `ROUTER_MIN_CONFIDENCE` | `0.8` | escalate below this confidence on cascade routes |
| `ROUTER_LONG_CONTEXT_TOKENS` | `100000` | prompt plus output size that selects the long-context route |
| `ROUTER_CLASSIFIER_MIN_CONFIDENCE` | `0.5` | below this, the classifier abstains |
| `LLM_PROVIDER`, `LLM_MODEL`, API keys | see `aie_core` | used when fakes are replaced by real clients |

### The catalog

The catalog holds one `ModelProfile` per alias and derives a request's `Requirements`. Read `capability_gaps` first: it is the single definition of compatible. The excerpt below shows the profile, requirement derivation, the gap check, and the compatibility query; the `Tier` enum, JSON loading, and the illustrative `northwind_catalog()` (four aliases: `nw-small` on-premises, `nw-general` and `nw-reasoning` in the EU, `nw-longctx` with a large window but no tools) are on disk.

```python
# path: book/projects/examples/ch07/catalog.py (excerpt; full file on disk)
class ModelProfile(BaseModel):
    alias: str                                   # what application code and routes use
    model_id: str                                # pinned, versioned identifier sent to the provider
    provider: str                                # key into the client registry
    context_tokens: int                          # total window: prompt plus output
    max_output_tokens: int
    supports_tools: bool = False
    supports_json_schema: bool = False           # native schema-constrained output
    supports_vision: bool = False
    reasoning_efforts: tuple[str, ...] = ()      # e.g. ("low", "medium", "high"); empty = no knob
    data_zones: tuple[str, ...] = ("any",)       # where inference runs, e.g. ("eu",); "any" = no residency guarantee
    cost_tier: Tier = Tier.MEDIUM
    latency_tier: Tier = Tier.MEDIUM
    input_per_1m: float = 0.0                    # illustrative USD per million input tokens
    output_per_1m: float = 0.0                   # illustrative USD per million output tokens
    # ...

class Requirements(BaseModel):
    # ...
    @classmethod
    def from_request(cls, req: CompletionRequest) -> "Requirements":
        """Derive requirements from the request itself, so callers cannot forget them."""
        has_image = any(
            not isinstance(m.content, str) and any(p.type == "image_url" for p in m.content)
            for m in req.messages
        )
        prompt_tokens = count_message_tokens(req.messages, req.model)
        return cls(
            min_context_tokens=prompt_tokens + req.max_tokens,
            min_output_tokens=req.max_tokens,
            needs_tools=bool(req.tools) and req.tool_choice != "none",
            needs_json_schema=req.response_schema is not None,
            needs_vision=has_image,
            data_zone=req.metadata.get("data_zone"),
            reasoning_effort=req.metadata.get("reasoning_effort"),
        )

def capability_gaps(profile: ModelProfile, need: Requirements) -> list[Gap]:
    gaps: list[Gap] = []
    if need.min_context_tokens > profile.context_tokens:
        gaps.append(Gap(capability="context",
                        detail=f"needs {need.min_context_tokens} tokens, window is {profile.context_tokens}"))
    # ... output length, tools, and vision follow the same pattern
    if need.data_zone and need.data_zone not in profile.data_zones:  # "any" never satisfies an explicit zone
        gaps.append(Gap(capability="data_zone",
                        detail=f"request must stay in {need.data_zone}; model runs in {list(profile.data_zones)}"))
    if need.needs_json_schema and not profile.supports_json_schema:
        # Soft: structured output can fall back to prompt-and-parse with validation and repair
        # (Chapter 6). The request still works, but its failure rate changes, so it is flagged.
        gaps.append(Gap(capability="json_schema", hard=False,
                        detail="no native schema mode; falls back to prompt+parse with repair"))
    if need.reasoning_effort and need.reasoning_effort not in profile.reasoning_efforts:
        gaps.append(Gap(capability="reasoning_effort", hard=False,
                        detail=f"effort {need.reasoning_effort!r} unsupported; model offers {list(profile.reasoning_efforts)}"))
    return gaps

def is_compatible(profile: ModelProfile, need: Requirements) -> bool:
    return not any(g.hard for g in capability_gaps(profile, need))

class ModelCatalog(BaseModel):
    # ...
    def compatible(self, need: Requirements, among: Iterable[str] | None = None) -> list[ModelProfile]:
        """Compatible profiles, cheapest first, then fastest. Stable for equal tiers."""
        names = list(among) if among is not None else list(self.profiles)
        found = [self.get(n) for n in names if is_compatible(self.get(n), need)]
        return sorted(found, key=lambda p: (p.cost_tier, p.latency_tier))
```

### The selection harness

The harness runs every candidate on the same cases, scores each answer, and summarizes quality with a Wilson interval, latency percentiles, and cost. The excerpt shows the three functions that carry the method: `run_case`, where a failure becomes a scored zero instead of a crash; `dominates`, which defines the Pareto front; and `choose`, which applies the quality floor to the interval's lower bound. The result models, `summarize`, `percentile`, `run_selection`, and `paired_disagreements` (the paired counts) are on disk.

```python
# path: book/projects/examples/ch07/selection.py (excerpt; full file on disk)
def wilson_interval(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval: honest at small n and near 0 or 1, unlike the normal approximation."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)

def run_case(name: str, client: LLMClient, case: EvalCase, scorer: Scorer,
             pricing: PricingTable, model: str | None = None) -> CaseResult:
    req = case.request.model_copy(update={"model": model}) if model else case.request
    started = time.perf_counter()
    try:
        completion = client.complete(req)
    except LLMError as exc:  # a failed call is a scored outcome, not a crash of the harness
        return CaseResult(case_id=case.id, candidate=name, score=0.0,
                          latency_ms=(time.perf_counter() - started) * 1000,
                          cost_usd=0.0, error=type(exc).__name__)
    wall_ms = (time.perf_counter() - started) * 1000
    # Providers report server-side latency; fakes report a scripted one. Prefer the reported
    # figure when present so fakes are deterministic; real runs should also log wall time.
    latency = completion.latency_ms or wall_ms
    return CaseResult(
        case_id=case.id, candidate=name, score=scorer(case, completion), latency_ms=latency,
        cost_usd=pricing.cost_usd(completion.model, completion.usage),
        input_tokens=completion.usage.input_tokens, output_tokens=completion.usage.output_tokens,
        output=completion.text[:500],
    )

def dominates(a: CandidateSummary, b: CandidateSummary) -> bool:
    """a dominates b if it is no worse on every axis and strictly better on at least one."""
    no_worse = (a.quality >= b.quality and a.cost_per_request_usd <= b.cost_per_request_usd
                and a.p95_latency_ms <= b.p95_latency_ms)
    better = (a.quality > b.quality or a.cost_per_request_usd < b.cost_per_request_usd
              or a.p95_latency_ms < b.p95_latency_ms)
    return no_worse and better

def choose(report: SelectionReport, *, min_quality: float, max_p95_ms: float,
           use_lower_bound: bool = True) -> CandidateSummary | None:
    """Cheapest candidate meeting the quality floor and latency ceiling. With use_lower_bound
    the floor applies to the lower end of the confidence interval, which refuses to pick a
    model on luck when the evaluation set is small."""
    ok = [s for s in report.summaries
          if (s.quality_low if use_lower_bound else s.quality) >= min_quality and s.p95_latency_ms <= max_p95_ms]
    return min(ok, key=lambda s: (s.cost_per_request_usd, s.p95_latency_ms), default=None)
```

### The router

The router has two halves. `route` decides: it picks a route, then filters the route's models by hard capability gaps. `complete` executes: it calls the surviving candidates in order and runs the cascade. A `Route` is plain data, and the Northwind rules show the ordering principle, policy before cost.

```python
# path: book/projects/examples/ch07/router.py (excerpt; full file on disk)
class Route(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    model: str                                  # catalog alias
    fallbacks: tuple[str, ...] = ()             # tried on retryable errors or capability gaps
    reasoning_effort: str | None = None         # passed to models that expose the knob
    escalate_to: str | None = None              # confidence cascade target
    min_confidence: float = 0.0                 # escalate when confidence is below this

def northwind_rules(long_context_threshold: int = 100_000) -> list[Rule]:
    """Rules run in order; put the ones that encode policy before the ones that save money."""
    return [
        Rule("restricted_data", "private", lambda req, need: need.data_zone == "onprem"),
        Rule("high_risk", "high_assurance", lambda req, need: req.metadata.get("risk") == "high"),
        Rule("long_context", "long_context", lambda req, need: need.min_context_tokens > long_context_threshold),
        Rule("narrow_task", "small_first", lambda req, need: req.metadata.get("task") in NARROW_TASKS),
    ]
```

The decision half. `_pick_route` (on disk) returns the first matching rule's route, else a classifier prediction above its confidence floor, else the default route. Everything after that is capability filtering:

```python
# path: book/projects/examples/ch07/router.py (excerpt; full file on disk)
    def route(self, req: CompletionRequest) -> RouteDecision:
        need = Requirements.from_request(req)
        route, stage, rule, conf = self._pick_route(req, need)
        effort = need.reasoning_effort or route.reasoning_effort
        need = need.model_copy(update={"reasoning_effort": effort})

        substitutions: list[str] = []
        planned = [route.model, *route.fallbacks]
        candidates: list[str] = []
        for alias in planned:
            gaps = [g for g in capability_gaps(self.catalog.get(alias), need) if g.hard]
            if gaps:
                substitutions.append(f"skip {alias}: " + "; ".join(f"{g.capability}: {g.detail}" for g in gaps))
            else:
                candidates.append(alias)
        if not candidates:
            # Every planned model is unfit. Look across the whole catalog, cheapest first, but only
            # among models we can actually call.
            for p in self.catalog.compatible(need, among=self.clients):
                candidates.append(p.alias)
                substitutions.append(f"catalog substitute {p.alias} for route {route.name}")
                break
        if not candidates:
            raise NoCompatibleModelError(
                f"no model satisfies route {route.name!r}: " + " | ".join(substitutions))

        escalate_to = route.escalate_to
        escalation_disabled = False
        if escalate_to is not None and not is_compatible(self.catalog.get(escalate_to), need):
            substitutions.append(f"escalation to {escalate_to} disabled: capability gap")
            escalate_to, escalation_disabled = None, True
        # ... soft gaps on surviving candidates become warnings; the RouteDecision is returned
```

The execution half, with the span attributes elided. `_call` (on disk) sends the request through the alias's client with `req.model` set to the pin and the effort level in metadata where the profile supports it, and prices the completion.

```python
# path: book/projects/examples/ch07/router.py (excerpt; full file on disk)
    def _needs_escalation(self, completion: Completion, decision: RouteDecision, attempt: Attempt) -> bool:
        if self.validator is not None and not self.validator(completion):
            attempt.outcome, attempt.detail = "invalid", "validator rejected output"
            return True
        if self.confidence is not None:
            attempt.confidence = self.confidence(completion)
            if attempt.confidence < decision.min_confidence:
                attempt.outcome = "low_confidence"
                attempt.detail = f"confidence {attempt.confidence:.2f} < {decision.min_confidence:.2f}"
                return True
        return False

    def complete(self, req: CompletionRequest) -> RoutedCompletion:
        with self.tracer.span("router.complete") as span:
            decision = self.route(req)
            # ...
            for alias in decision.candidates:
                try:
                    completion, attempt = self._call(alias, req, decision)
                except LLMError as exc:
                    attempts.append(Attempt(model=alias, outcome="error", detail=type(exc).__name__))
                    if not exc.retryable:
                        span.record_exception(exc)
                        raise
                    continue
                attempts.append(attempt)
                served_by = alias
                break
            # ... no completion: raise ProviderUnavailableError
            escalated = degraded = False
            target = decision.escalate_to
            if target and target != served_by and self._needs_escalation(completion, decision, attempts[-1]):
                try:
                    strong, strong_attempt = self._call(target, req, decision)
                    attempts.append(strong_attempt)
                    completion, served_by, escalated = strong, target, True
                except LLMError as exc:
                    # Keep the cheap answer but say it is below the route's bar.
                    attempts.append(Attempt(model=target, outcome="error", detail=type(exc).__name__))
                    degraded = True
            elif decision.escalation_disabled and self._needs_escalation(completion, decision, attempts[-1]):
                # The cascade could not run for this request, and the answer is below its bar.
                degraded = True
            mismatch = completion.model != self.catalog.get(served_by).model_id
            # ... build the RoutedCompletion, set span attributes, return it
```

### Cascade evaluation

The evaluator runs each model once per case, then replays every threshold offline. `collect_outcomes` (on disk) records, per case, whether each model was right, the small model's confidence, and both stages' cost and latency. `_per_case` is where the utility formula above becomes code; `simulate` applies one threshold to the collected table without calling a model.

```python
# path: book/projects/examples/ch07/cascade_eval.py (excerpt; full file on disk)
class UtilityModel(BaseModel):
    """All values illustrative. Set them with the product owner, not by the engineer alone."""

    value_correct: float = 0.0          # often zero: the baseline is "the task got done"
    cost_silent_error: float = 0.05     # a mis-triaged ticket waits in the wrong queue
    detect_rate: float = 0.0            # share of accepted wrong answers caught downstream
    rework_cost: float = 0.01           # cost of a caught error, on top of re-running large
    latency_cost_per_s: float = 0.0     # what a second of waiting is worth, if anything

def _per_case(o: CaseOutcome, escalate: bool, u: UtilityModel) -> tuple[float, float, float, float]:
    """(expected correctness, cost, latency_ms, penalty) for one case under one decision."""
    if escalate:
        cost = o.small_cost_usd + o.large_cost_usd
        latency = o.small_latency_ms + o.large_latency_ms
        return float(o.large_correct), cost, latency, 0.0 if o.large_correct else u.cost_silent_error
    if o.small_correct:
        return 1.0, o.small_cost_usd, o.small_latency_ms, 0.0
    # False accept. A share d of these is caught downstream and redone on the large model,
    # paying rework plus a second call; the rest stays wrong and silent.
    d = u.detect_rate
    redo_penalty = u.rework_cost + (0.0 if o.large_correct else u.cost_silent_error)
    return (d * float(o.large_correct),
            o.small_cost_usd + d * o.large_cost_usd,
            o.small_latency_ms + d * o.large_latency_ms,
            d * redo_penalty + (1 - d) * u.cost_silent_error)

def simulate(outcomes: Sequence[CaseOutcome], threshold: float, u: UtilityModel) -> PolicyMetrics:
    """Accept the small answer when its confidence >= threshold, otherwise escalate."""
    # ...
    for o in outcomes:
        escalate = o.small_confidence < threshold
        ok, c, lat, pen = _per_case(o, escalate, u)
        # ... accumulate correctness, cost, penalty, latency
        false_accepts += (not escalate) and (not o.small_correct)
        false_escalations += escalate and o.small_correct
    # ... returns PolicyMetrics with accuracy, cost, p95, rates, router accuracy, and utility

def sweep(outcomes: Sequence[CaseOutcome], u: UtilityModel,
          thresholds: Sequence[float] | None = None) -> list[PolicyMetrics]:
    ts = thresholds if thresholds is not None else sorted({0.0, 1.01, *(o.small_confidence for o in outcomes)})
    return [simulate(outcomes, t, u) for t in ts]
```

`baseline` (on disk) computes the always-small and always-large policies for comparison, and `calibration_table` and `expected_calibration_error` produce the reliability table shown earlier.

### Task, fakes, and tests

`tasks.py` turns each ticket in `shared-data/tickets.jsonl` into an `EvalCase` holding the exact `CompletionRequest` a candidate receives (system prompt with the twelve allowed categories, a JSON schema for `{"category", "confidence"}`, `max_tokens=64`, and `metadata={"task": "classify_ticket"}`), plus the expected label. `score_label` returns 1.0 for an exact category match and 0.0 for anything else, including malformed output. `label_confidence` reads the confidence field and returns 0.0 for malformed output, so unparseable answers always escalate.

`fakes.py` builds the models as `FakeLLM(handler=...)` instances. The small model is a keyword scorer whose confidence comes from the margin between its top two categories: right on tickets that use the obvious words, unsure or wrong on tickets that mix topics. The large model returns the gold label except on a deterministic hash-selected subset of roughly one in twenty (four of sixty here); the reasoning candidate is the same fake with a different subset and a longer latency. All three report fixed illustrative latencies. `tasks.py` and `fakes.py` are on disk in full.

The tests below demonstrate the chapter's central claims: a cascade pays for both calls, capability gaps beat fallback order, residency is never traded for availability, and the offline sweep predicts the live router. The full test file on disk also covers every rule, soft warning, misconfiguration, the hidden-fallback mismatch, and the utility-versus-router-accuracy disagreement.

```python
# path: book/projects/examples/ch07/test_ch07.py (excerpt; imports and fixtures omitted, full file on disk)
def make_router(catalog, gold, **overrides) -> Router:
    clients = {
        "nw-small": overrides.pop("small", make_small_model()),
        "nw-general": overrides.pop("general", make_large_model(gold)),
        "nw-reasoning": overrides.pop("reasoning", make_large_model(gold, model="reasoner-2026-01")),
        "nw-longctx": overrides.pop("longctx", make_large_model(gold, model="longctx-2025-12")),
    }
    overrides.setdefault("confidence", label_confidence)
    overrides.setdefault("validator", lambda c: parse_label(c) is not None)
    return northwind_router(catalog, clients, **overrides)

def test_low_confidence_escalates_and_costs_both_calls(catalog, gold):
    small = FakeLLM(responses=[{"category": "hardware", "confidence": 0.41}], model="small-instruct-2026-03")
    general = FakeLLM(responses=[{"category": "vpn_network", "confidence": 0.93}], model="general-2026-02")
    router = make_router(catalog, gold, small=small, general=general)
    r = router.complete(ticket_request("VPN drops", "VPN disconnects every 10 minutes since Monday"))
    assert r.escalated and r.served_by == "nw-general"
    assert [a.outcome for a in r.attempts] == ["low_confidence", "ok"]
    assert r.attempts[0].confidence == pytest.approx(0.41)
    assert r.cost_usd == pytest.approx(sum(a.cost_usd for a in r.attempts)) and r.attempts[0].cost_usd > 0
    assert parse_label(r.completion).category == "vpn_network"

def test_long_context_with_tools_never_falls_back_to_a_toolless_model(catalog, gold):
    tools = [ToolSpec(name="search_tickets", description="search")]
    req = CompletionRequest(messages=[Message.user(big_text(150_000))], max_tokens=2_000, tools=tools)
    d = make_router(catalog, gold).route(req)
    assert d.route == "long_context" and d.candidates == ["nw-reasoning"]
    huge = req.model_copy(update={"messages": [Message.user(big_text(400_000))]})
    with pytest.raises(NoCompatibleModelError):
        make_router(catalog, gold).route(huge)

def test_restricted_data_never_leaves_the_zone(catalog, gold):
    req = CompletionRequest(messages=[Message.user("summarize this payroll export")],
                            tools=[ToolSpec(name="lookup_employee", description="lookup")],
                            metadata={"data_zone": "onprem"})
    with pytest.raises(NoCompatibleModelError):   # only nw-small runs on-prem and it has no tools
        make_router(catalog, gold).route(req)
    ok = req.model_copy(update={"tools": None})
    assert make_router(catalog, gold).route(ok).candidates == ["nw-small"]

def test_online_router_matches_offline_simulation(catalog, gold, cases, pricing):
    """The offline sweep is only trustworthy if it predicts what the live router does."""
    outs = collect_outcomes(cases, make_small_model(), make_large_model(gold), score_label,
                            label_confidence, pricing)
    predicted = simulate(outs, 0.8, UtilityModel())
    router = make_router(catalog, gold, min_confidence=0.8, validator=None)
    results = [router.complete(c.request) for c in cases]
    accuracy = sum(score_label(c, r.completion) for c, r in zip(cases, results)) / len(cases)
    escalation = sum(r.escalated for r in results) / len(cases)
    cost = sum(r.cost_usd for r in results) / len(cases)
    assert accuracy == pytest.approx(predicted.accuracy)
    assert escalation == pytest.approx(predicted.escalation_rate)
    assert cost == pytest.approx(predicted.cost_per_request_usd)
```

## Code walkthrough

**The catalog is data with a validator.** A catalog loaded from JSON is checked at startup: a validator on `ModelProfile` (on disk) rejects an output limit larger than the window before any request runs. Only the ordering of tiers matters; the illustrative prices exist so `pricing()` (on disk) can feed `aie_core`'s `PricingTable`, keyed by the pinned `model_id` that completions report. `capability_gaps` is the single definition of "compatible", and it returns structured gaps rather than a boolean so the router can explain every skip. Data zone and effort come from metadata set by application code, never from the user's text.

**The harness scores failures instead of crashing on them.** `run_case` records an `LLMError` as a zero score with the error class, so a candidate rate-limited half the time shows a 50% error rate instead of aborting the run. Latency prefers the provider-reported figure so fakes are deterministic; a real bake-off should also log wall time, because client-side queueing is part of what users experience.

**The router validates its configuration at construction.** `_check_config` (on disk) confirms that every route, rule target, and alias exists and has a client. A typo in a fallback list is a startup error, not a `KeyError` during the first outage that exercises it.

**Route selection and capability filtering are separate steps.** `_pick_route` answers "which policy applies"; `route` answers "which models can execute it." Keeping them apart is what makes the long-context test readable: the request matches the long-context rule, the long-context model is skipped for lacking tools, and the reasoning model, which has both the window and tool calling, serves it. Make the prompt 400k tokens and nothing fits; the router raises instead of truncating.

**Escalation reads the attempt it is judging.** `_needs_escalation` runs the validator before the confidence function, because a parse failure makes the confidence meaningless, and it writes the outcome (`invalid` or `low_confidence`) and the confidence onto the attempt record. The escalation is skipped when the escalation target already served the request as a fallback, which `test_retryable_error_falls_back_without_double_escalation` (on disk) pins down: paying the strong model twice for one answer is a classic cascade bug.

**Degraded is a first-class outcome.** When escalation fails, the router returns the cheap answer with `degraded=True` and the caller decides: caveat, review queue, or retry. Raising would turn a quality problem into an availability problem.

**The router checks what actually served the request.** The `mismatch` line compares `completion.model` with the pinned `model_id` of the alias that answered and stores it as `model_mismatch` on the result. It costs one string comparison and catches a layering bug the capability check cannot see: a capability-changing fallback configured inside a `ModelGateway`, below the router, where no capability check runs. `test_model_mismatch_below_the_router_is_detected` (on disk) simulates exactly that. Alert on the attribute rather than raising, because the answer may be fine and the fix is configuration, not a failed request.

**A disabled cascade is still a cascade.** When a route's escalation target has a hard gap for this request (an image request on a route whose strong model is text-only), the router disables escalation and records why. It still runs the validator and confidence check on the answer, and marks a failing answer `degraded`, which `test_disabled_escalation_marks_low_confidence_answer_degraded` (on disk) pins down. Otherwise the route's quality bar would silently not apply to exactly the requests that can least afford it.

**The cascade evaluator separates collection from simulation.** `collect_outcomes` runs each model once per case. `simulate` applies a threshold to that table with no model calls, so `sweep` can evaluate every distinct confidence value as a threshold for the cost of two passes. `_per_case` is the only place utility is defined; the false-accept branch charges the share caught downstream for both model calls plus rework, which is how the warning about misroutes becomes a number. The last test in the excerpt, `test_online_router_matches_offline_simulation`, runs the real router over all sixty tickets and checks that its accuracy, escalation rate, and cost equal the offline prediction.

## Production considerations

**Latency.** A cascade's escalated path costs the cheap call plus the strong call, serially. For every cascade in the sweep table, p95 is the sum, 2,050 ms illustrative, though its mean is far below the large model's. As a rule of thumb with steady latencies, once more than 5% of traffic escalates, the cascade's p95 exceeds the strong model's alone: p95 is the latency 95% of requests beat, so it lands on the two-stage path (2,050 ms against 1,800 ms). Mitigations: stream the strong answer, put a tight timeout on the cheap stage, or send predicted-hard requests straight to the strong model. Route selection itself must be cheap: rules cost microseconds, an embedding classifier one cached embedding call, a model-as-router a full call per request.

**Cost.** Track spend per route, per serving model, and per escalation outcome, not only per tenant. The two numbers that explain a cost regression are the route mix (what share of traffic each route receives) and the escalation rate on cascade routes. A one-point rise in escalation on a high-volume route can cost more than a price change. Chapter 30 builds the cost model these numbers feed. Cap reasoning effort with a token ceiling per route, so the most expensive knob in the catalog has a bound.

**Budget enforcement.** Routing is the natural place for it: when a tenant or feature approaches its spend ceiling, a policy rule can move its traffic to a cheaper route (or a shorter context) and mark responses degraded, which is a better failure than a hard stop at the end of the month.

**Security.** Policy rules read only metadata that application code sets after authentication: tenant, data zone, risk flag, task. A user cannot select a route by writing "this is high risk" in the prompt. Classifier and model-as-router strategies read user text, which makes them manipulable: a user who learns that complicated-sounding questions reach the reasoning model can drive up your cost, so cap spend per user and per route. The data-zone rule is first in the rule list and is a hard gap in the capability check, and a model marked `any` never satisfies an explicit zone, so even a misconfigured fallback list or a catalog-wide substitution cannot send restricted data to a cloud model. Two tests verify the router raises rather than leaking: `test_restricted_data_never_leaves_the_zone` and `test_oversized_onprem_request_is_never_substituted_to_a_cloud_model`. Every model a route can reach is part of the request's trust boundary: a fallback provider with different retention terms is a data-processing decision.

**Operations.** Log the route, stage, serving model and pinned version, attempts, escalation, degradation, and warnings on every request; these are span attributes in `aie_core` tracing (Chapter 31). Dashboard the route distribution and alert when it drifts, because a classifier whose input distribution shifts will silently move traffic between routes.

Exercise fallbacks continuously, not only during incidents: send a small fraction of traffic through each fallback path, or run the selection harness against fallbacks nightly, so you discover a broken fallback before you need it. Under overload, routing is also admission control: sending traffic to a smaller model or a reduced context can serve everyone acceptably where the largest model would time out for many (Chapter 29).

**What to measure and alert on.** Every signal below comes from the router's span attributes and result fields, aggregated per route and per serving alias. Baselines and thresholds are illustrative; set yours from a few weeks of your own traffic.

| Signal | Source | Alert when (illustrative) | Usually means |
|---|---|---|---|
| route mix | `router.route` share | any route's share moves more than 5 points week over week without a deploy | classifier drift or a traffic change |
| default-stage share | `router.stage == "default"` | doubles against its baseline | classifier confidence falling, input drift |
| escalation rate | `router.escalated` on cascade routes | more than 1.5x baseline for an hour | confidence shift after a prompt or model change |
| degraded rate | `router.degraded` | above 1% on any route | escalation target failing, or cascades disabled by capability gaps |
| model mismatch | `router.model_mismatch` | any occurrence | a fallback below the router, or an alias pointing at a floating name |
| substitutions | `router.substitutions` greater than 0 | sustained rise on one route | requests no longer fit the planned models (prompt growth, new tools) |
| soft-gap warnings | `router.warnings` | sustained rise | requests running under changed assumptions, such as no native schema mode |
| per-alias error attempts | attempts with outcome `error` | above 2% on one alias | provider incident, rate limits, or a retired pin |
| cost per request by route | `router.cost_usd` | more than 20% above the 7-day median | escalation or route-mix change, or a price change |
| p95 latency by route | sum of attempt latencies | above the route's SLO | escalation tail or a slow fallback |
| `NoCompatibleModelError` count | router exceptions | any sustained rate | requests outgrew the catalog; callers must shrink context |

Two of these deserve a dashboard of their own: escalation rate next to cost per request on each cascade route, and route mix next to per-route sampled quality. Together they explain most cost and quality regressions in a routed system without opening a single trace.

## Common mistakes

- **Choosing from a leaderboard.** Public benchmarks measure someone else's distribution. Run your evaluation set.
- **One model for everything.** The strongest model as a universal default overpays on easy traffic by an order of magnitude and often has worse latency than the task needs.
- **Tuning the router on router accuracy.** It weighs a wasted escalation the same as a shipped wrong answer. Optimize utility with real error prices.
- **Pricing self-hosting at peak throughput.** A GPU node costs the same idle or busy. Compare cost per token at your measured average utilization, plus the people who run it.
- **Comparing candidates on point estimates from small sets.** Sixty cases give intervals around fifteen points wide. Use lower bounds and paired counts.
- **Testing the cascade only offline.** If the live router does not reproduce the simulated numbers, the sweep's numbers are untested. Check them against each other.
- **Escalating twice.** When the fallback is the escalation target, a naive cascade pays the strong model again for the same answer.

## Failure modes

**Silent downgrade.** Hard requests are routed to the cheap model and answered plausibly but wrongly. Telemetry: aggregate quality flat, quality on the hard slice down, false-accept rate up on the sampled labeled set, complaint or escalation rate up for one route. Test: per-route and per-slice evaluation in CI, with a floor on the hard slice, not only the overall score.

**Escalation storm.** The cheap model's confidence distribution shifts (a prompt change, a new ticket type, a model update) and most traffic escalates. Telemetry: escalation rate on a cascade route jumps; cost per request and p95 latency rise together while quality is unchanged. Test: alert on escalation rate against its baseline; re-run the calibration table on every change to the cheap model or its prompt.

**Overconfident cheap model.** Self-reported confidence was trusted without a calibration table, and confident errors pass every threshold. Telemetry: errors concentrated in the top confidence bin of the calibration table; raising the threshold buys little quality for large cost. Remedy: a validator or a better confidence signal, not a higher threshold. The sweep shows this as a flat quality curve against escalation rate.

**Fallback truncation.** During a primary outage, requests go to a fallback with a smaller window and some layer truncates the prompt to fit. Telemetry: answers lose citations or contradict evidence only while the primary is down; prompt token counts on the fallback cluster at its window size. Test: a capability-aware router makes this impossible by construction; `test_oversized_prompt_skips_the_small_model` checks it.

**Tool-less fallback.** A tool-using request reaches a model without tool calling and returns prose that claims to have checked something. Telemetry: `finish_reason` is `stop` with no tool calls on a route whose requests always carry tools; tool-call rate drops during an incident. Test: `test_long_context_with_tools_never_falls_back_to_a_toolless_model`.

**Hidden fallback below the router.** A `ModelGateway` registered for one alias has its own fallback list pointing at a model with different capabilities. During an outage it serves tool-using or long-context requests from that model, and the router, which only sees a successful completion, records the planned alias. Telemetry: `router.model_mismatch` is true; the completion's model differs from the alias's pin; tool-call rate or citation rate drops only during the provider incident. Test: `test_model_mismatch_below_the_router_is_detected`; the structural fix is to keep only same-capability retries in the gateway and put every capability-changing fallback in a route.

**Retired pin.** A provider retires a pinned version on its announced date. Telemetry: error attempts on one alias jump to 100% at a point in time; if the adapter treats the error as retryable, served-by shifts to the fallback and cost per request rises with no quality change; if not, the route fails outright. Test: a startup check on deprecation dates (exercise P3) and an adapter test that "model not found" maps to a non-retryable error.

**Router drift.** The classifier was fit on last quarter's traffic. Telemetry: the route distribution shifts without a deploy; classifier confidence drifts down and more traffic falls to the default route. Test: track the share of default-route decisions and the classifier's confidence histogram; re-label a monthly sample.

**Alias drift.** An alias was repointed to a new version without re-evaluation, or it points at a floating provider name that changed underneath it, so behavior changes without a deploy. Telemetry: quality or format-error rate changes at a point in time that matches no deploy; the pinned version attribute changed. Test: a release gate that re-runs the selection harness when any pin changes.

**Cascade latency tail.** The mean improves, the p95 gets worse. Telemetry: latency histogram becomes bimodal; p95 equals cheap plus strong latency. Test: include p95 in the sweep, as `cascade_eval` does, and set a ceiling on it in the policy choice.

## Tradeoffs

**Rules versus learned routing.** Rules are explainable, cheap, and testable, but they only separate what code can already see. Learned routing separates traffic by content but adds a component that drifts and must be evaluated on its own. Start with rules; add a classifier when you can measure its routing errors in money.

**Predictive routing versus cascades.** Predicting difficulty from the input avoids paying for a cheap attempt on hard requests and keeps the latency tail tight, but input features are a weak signal of difficulty. Cascades see the attempt, which is a stronger signal, but every escalation pays twice. Many systems use both: rules and a classifier send the obviously hard traffic straight to the strong model, and a cascade handles the uncertain middle.

**Cost versus quality on the hard slice.** Every routing configuration is a point on a curve of cost against hard-slice quality, and spending more buys quality where errors are expensive. The utility function is where product owners, not engineers, set the exchange rate between a cent of spend and a wrong answer.

**Provider diversity versus behavioral consistency.** A second provider improves availability but changes refusal patterns, prompt sensitivity, and structured-output reliability. Each fallback needs its own evaluation, and for regulated workloads some teams accept lower availability rather than inconsistent answers.

**Pinning versus freshness.** Pins give stability but leave improvements and price cuts unused until someone re-runs the evaluation. Make re-evaluation one command, so pinning does not become stagnation.

**A specialist versus routing.** A fine-tuned small model for one task (Chapter 33) can remove the need for a cascade on that task entirely. Routing is cheaper to build; fine-tuning can be cheaper to run at volume.

## Evaluation and testing

Test the router at four levels.

**Unit tests for policy.** Every rule, capability gap, soft warning, and misconfiguration, with nothing beyond a scripted `FakeLLM`. These tests are deterministic, run in milliseconds, form the router's contract, and belong in every CI run.

**Selection evaluation per candidate.** The harness on the evaluation set, reporting intervals and paired counts, re-run when a candidate, prompt, or pin changes. Report per slice (category, tenant, priority) as well as overall: a candidate that wins on average and loses on P1 tickets is not a win (Chapter 25 turns this into a release gate).

**Cascade evaluation.** Collect outcomes once, sweep thresholds, choose by utility with the product's error prices, and keep the calibration table. Re-run when either model, the prompt, or the confidence signal changes. Report the false-accept and false-escalation rates next to utility so a reviewer can see why a threshold was chosen.

**Online consistency and monitoring.** Check that the live router reproduces the offline prediction on the same set, as `test_online_router_matches_offline_simulation` does. In production, sample routed requests for labeling per route, compare each route's live quality with its offline number, and evaluate each route separately: a global quality metric averages a healthy general route with a failing cascade. Shadow evaluation (see the migration procedure) is the safest way to measure a new threshold or a new pin on real traffic before switching.

## Before you ship

- [ ] Every model reference in application code is a catalog alias, and every alias resolves to a pinned, versioned identifier (no floating names).
- [ ] Each routed alias has a recorded selection run (evaluation set version, prompt version, pin, results) that meets the quality floor on its interval lower bound and the latency ceiling at p95.
- [ ] Per-slice results exist for the hard and high-stakes slices, with a floor on each, not only an overall score.
- [ ] The data-zone rule is first in the rule list, and a test proves a restricted request raises rather than reaching a model outside its zone.
- [ ] Tests prove that tool-carrying, image-carrying, and oversized requests never reach a model without the tools, vision, or window they need.
- [ ] Each cascade threshold was chosen by utility with error prices agreed with the product owner, and the calibration table for its confidence signal is stored with the decision.
- [ ] A test checks that the live router reproduces the offline simulation's accuracy, escalation rate, and cost on the evaluation set.
- [ ] Every reasoning-effort route has a token ceiling.
- [ ] Gateways below the router contain only same-capability retries; every capability-changing fallback lives in a route.
- [ ] Deprecation dates are tracked for every pin, with a warning weeks before retirement, and "model not found" stays non-retryable through every proxy layer.
- [ ] Alerts are live for route mix, escalation rate, degraded rate, model mismatch, per-alias error attempts, and cost per request by route.
- [ ] For self-hosted aliases, the evaluated configuration matches the deployed one (quantization, inference server version), and cost was computed at measured utilization.

## Exercises

**Start here:** K2, K4, E1, P2, D1 (about 4 hours). The rest go deeper.

### Knowledge questions

**K1.** Name the ten selection axes and classify each as a hard constraint or a trade-off axis. Which one can be either, and what decides it?

**K2.** Define false accept and false escalation in a two-stage cascade. Which one costs quality, which one costs money, and why does router accuracy fail to capture the difference?

**K3.** Why does the selection procedure apply the quality floor to the lower bound of a confidence interval rather than to the point estimate? What does it cost to do so?

**K4.** What is expected calibration error, and why does an overconfident top bin limit what any escalation threshold can achieve?

**K5.** When is a small model with retrieval a better choice than a reasoning model? Give the conditions, not an example.

**K6.** What does pinning a model version protect against, and what new obligation does it create?

### Engineering questions

**E1.** Northwind's incident researcher uses tools and sometimes needs 150k tokens of logs. Its primary model has a 200k window and tool calling. Design its fallback chain from the illustrative catalog and state what the router should do when the primary is down and the request is 250k tokens.

**E2.** Product says a misrouted P1 ticket costs a human 20 minutes, while a misrouted P4 ticket costs nothing measurable. How would you change the cascade evaluation and the router to reflect this, and what new data would you need?

**E3.** A team proposes a model-as-router that reads every request and picks among four routes. Write the arguments for and against, the test you would require before accepting it, and the safeguards it needs in production.

**E4.** Your cascade's mean latency is 520 ms and its p95 is 2,050 ms. The SLO is p95 under 1,500 ms. List three design changes that could meet the SLO and the cost or quality price of each.

**E5.** Northwind classifies about 3 million tickets a month, each about 600 input and 40 output tokens, mostly during business hours. A team proposes moving `nw-general`'s share of that traffic to a self-hosted open-weight model. List the numbers you need to compare the two options, explain how traffic shape changes the answer, and name the evaluation you would run before any cost comparison matters.

### Practical exercises

**P1.** (about 2 hours) Add a third cascade signal to the router: agreement. Call the small model twice (the second time with a different temperature, or a second small model) and escalate when the labels differ. Extend `cascade_eval` to simulate it from collected outcomes and compare its utility with the self-reported confidence signal at two error prices.

**P2.** (about 2 hours) Make the cascade evaluation slice-aware: compute utility per priority (P1 to P4) with a different silent-error cost per priority, choose a threshold per slice, and extend the router so a route's `min_confidence` can depend on request metadata. Show that per-slice thresholds beat a single threshold on the ticket set.

**P3.** (about 60 min) Add a `deprecation_date` field to `ModelProfile` and a startup check that warns when any routed alias's pin expires within a configurable window and fails when it has passed. Add tests for both.

**P4.** (about 90 min) Extend the selection harness to evaluate effort levels as separate candidates: given a client and a list of effort levels, produce one row per level with its own quality, latency, and cost, and include them in the Pareto front.

### Debugging exercises

**D1.** After a routine prompt update to the ticket classifier, the monthly model bill for Northwind Assist rises 40% while accuracy on the evaluation set is unchanged. Traces show the `small_first` route's share of traffic is unchanged, `router.escalated` is true on 61% of its requests (it was 15%), and the attempts on escalated requests show outcome `low_confidence` with confidences clustered between 0.70 and 0.79. Diagnose.

**D2.** During a two-hour outage of the cloud provider, incident-research answers stopped citing log lines and several claimed that "no errors were found in the period." Traces from the window show `router.model` as `nw-reasoning`, but the completion's `model` field reports `longctx-2025-12`; `finish_reason` is `stop`, there are zero tool calls, and the decision's `substitutions` list is empty. The same requests before the outage were served by `nw-reasoning` with three to six tool calls each. What is wrong with the system, and which code path allowed it?

**D3.** A new embedding-based route classifier passes its offline evaluation with 92% route accuracy. Two weeks after launch, the share of requests on the `reasoning` route has fallen from 18% to 6%, the share decided at stage `default` has risen from 5% to 19%, and user ratings for analytical questions have dropped. No code or catalog change was deployed in that period. What happened, what telemetry confirms it, and what would have caught it earlier?

**D4.** On a Monday morning, cost per request on Northwind's `general` route rises about fivefold and its p95 latency moves from under 2 s to about 6.5 s. Quality ratings are unchanged and nothing was deployed. Calls to cloud models go through Northwind's internal inference proxy. Every trace on the route shows two attempts: the first on `nw-general` with outcome `error` and detail `ProviderUnavailableError`, after a latency of a few milliseconds, and the second `ok` on `nw-reasoning`. Other customers of the same provider report no incident. The catalog entry for `nw-general` was last changed eight months ago. Diagnose the root cause, explain why the router behaved as it did, and name the two changes that would have turned this into a loud, early failure.

## Key takeaways

- Choose models from a task-level evaluation on your own workload, in the configuration and deployment you will ship; benchmarks and vendor claims are hypotheses to test, and self-hosting pays only at sustained utilization or when data must stay inside.
- Separate hard constraints (context, tools, vision, data zone, output length) from trade-off axes (quality, latency, cost, reliability); filter by the first, optimize over the second.
- Read candidates as a Pareto front, choose the cheapest that meets the quality floor on its lower bound, and use paired counts to see whether two models fail on different cases.
- Treat reasoning effort as a per-route parameter with a budget cap, evaluated by outcome, never by the length of the reasoning.
- Start routing with static mapping and policy rules; add classifiers, embedding similarity, or cascades only when you can measure their routing errors.
- A cascade is only as good as its escalation signal. Measure calibration, prefer validators where they exist, and count the confident errors no threshold can catch.
- Optimize thresholds on end-to-end utility with real error prices; the same data gives different winners as the price of an error changes, while router accuracy picks the same threshold every time.
- Fallbacks must be capability-aware: never fall back to a model that lacks the window, tools, modality, or data zone the request needs, and fail loudly when nothing fits.
- Reference models by catalog aliases that resolve to pinned versions, record the pinned version on every trace, alert when the model that answered is not the pin the router planned for, and start every migration well before a pin's retirement date.
- Verify that the live router reproduces the offline simulation, then monitor route mix, escalation rate, and per-route quality in production.

## Further reading

- *On Calibration of Modern Neural Networks* (Guo et al., 2017): the reliability diagram and expected calibration error behind every cascade threshold in this chapter.
- *Self-Consistency Improves Chain of Thought Reasoning in Language Models* (Wang et al., 2023): the origin of agreement across samples as a signal, the basis of exercise P1.
- *Holistic Evaluation of Language Models (HELM)* (Liang et al., 2022): why a single benchmark number is a poor basis for choosing a model, and what multi-metric, scenario-based comparison looks like.
- *Trustworthy Online Controlled Experiments* (Kohavi, Tang, and Xu, 2020): the discipline behind shadowing, canaries, and reading online results when you repoint an alias or change a threshold.
- *Efficiently Scaling Transformer Inference* (Pope et al., 2023): the serving cost model behind the self-hosting arithmetic, developed further in Chapter 34.
