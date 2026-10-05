# Chapter 26 — Solutions

Solutions to the exercises in Chapter 26. Practical solutions describe the expected implementation and
acceptance criteria; debugging solutions name the root cause and the telemetry that reveals it.

## Knowledge questions

**K1.** Prompt injection is an authorization problem because the damage comes from an action being carried
out, not from the model being convinced. A language model reads its entire context as one token stream and
cannot reliably separate "instruction" from "data," so an attacker who places text anywhere in that
stream competes with the application's instructions. The wrong mental model is "find a perfect system
prompt the model can never disobey." The replacement is "assume the model can be influenced, and ensure
it lacks the authority to cause an unacceptable effect." Defense moves from wording to authorization
enforced in deterministic code.

**K2.** Direct injection is supplied by the user themselves, in their own message to the application; a
Northwind entry point is the chat message to the support agent ("waive my fee"). Indirect injection is
supplied by a content author or ticket submitter and embedded in content the system later reads on a
victim's behalf; Northwind entry points include a retrieved knowledge-base chunk or a retrieved support
ticket whose body contains instructions addressed to the assistant.

**K3.** The system prompt is enforced only by the model's tendency to follow it, and that tendency can be
shifted by adversarial input; it is a strong default, not a hard boundary. An effect that must be enforced
elsewhere: sending data to an external recipient. It is enforced in the tool gateway by a recipient
allowlist and a human-approval gate bound to the concrete arguments, so the send cannot occur even if the
model is persuaded to request it.

**K4.** (1) Outbound tools such as email or HTTP, controlled by separating sensitive-data access from
outbound capability and by an egress allowlist. (2) URLs in rendered markdown images, which need no tool;
controlled by stripping or sandboxing images, an egress allowlist on rendered hosts, and a Content
Security Policy. (3) URLs in rendered markdown links a user is lured to click; same controls. (4) Encoded
data smuggled into otherwise-legitimate tool arguments; controlled by argument validation and schema
constraints in the gateway. Canary tagging detects all four.

**K5.** A confused deputy is a privileged component tricked into using its authority for someone who lacks
it. An agent holds tools and credentials and takes instructions from untrusted text, so it is an ideal
deputy. The gateway must ask "is the requesting end user authorized for these specific arguments?" The
wrong question, which produces the vulnerability, is "is this agent allowed to call this tool?" The agent
is always allowed; the user may not be.

**K6.** Tool descriptions, skills, and MCP server metadata influence the model before and during a call,
so a change to them can alter agent behavior with no change to application code and no diff a code review
would catch. A poisoned description can bias tool selection or arguments; a poisoned skill can execute.
Two controls: pin and verify versions of tools, skills, and servers from trusted sources; and put their
descriptions under change review as dependencies, giving the agent only the minimum tool set per task.

**K7.** Both patterns combine read access to sensitive data with an outbound channel the attacker can
steer: for rendered-image exfiltration the channel is the user's browser fetching a model-chosen URL, for
the inbox agent it is the send tool. Either capability alone is harmless; the combination, reached by
injected text, is an exfiltration path. Moderation classifies content into harm categories (violence,
harassment, self-harm). For harmful content the harm is in the text itself, so classifying the text
addresses the actual risk. An exfiltration payload is ordinary-looking text and URLs with no harmful
category at all, and the damage is done by the effect (a fetch, a send), so only effect controls (egress
allowlist, separation of read and send, approval bound to arguments) address it.

## Engineering questions

**E1.** Adding an email-backed tool converts a read-only RAG system into a tool-using agent, which is the
jump from worked model 1 to worked model 2. New threats appear: indirect injection driving an outbound
send (A1), confused-deputy misuse of the send tool, replayed or duplicated sends (A4), and the agent
becoming an exfiltration path because it now has both sensitive-data read access and an outbound channel.
Existing disclosure risks escalate because the output can now leave through a tool, not only through the
rendered page. Required controls before shipping: human approval bound to the concrete recipient and body;
a recipient allowlist enforced in the gateway; argument authorization against the requesting user;
least-privilege, short-lived credentials for the send tool; idempotency keys; step and spend budgets; and
an effect-based red-team test that confirms an injected "email this data out" instruction produces no
send.

**E2.** The cache key must include every input that could change what a user is allowed to see, not only
what was asked. Required inputs: the normalized question text (what was asked); the tenant identifier (a
retail answer must never serve a logistics user); the user's authorization context, that is the set of
ACL groups or a stable hash of the user's entitlements (two users in the same tenant with different group
memberships must not share an entry); and the corpus or index version (so a re-indexed or redacted corpus
invalidates stale answers). Omitting tenant or authorization context is exactly threat R4. Caching the
rendered answer rather than raw documents further limits exposure if a key collision ever occurred.

**E3.** Argue against relying on it as a primary control. It is a weak signal with trivial bypasses and it
cannot be a boundary. Two corpus carriers defeat it directly: the base64 variant encodes the instruction,
so a regex for "ignore previous instructions" matches nothing while a capable model still decodes and
acts on it; and the HTML-comment variant hides the instruction inside `<!-- -->`, which may be stripped
from what a reviewer reads but preserved in what the model receives, and the phrasing need not contain the
blocked keywords at all. The fake-tool-output variant also evades it by using no imperative phrasing. A
regex detector is acceptable only as an early-warning layer that raises a signal; the actual defense must
be authorization, least privilege, and egress control that block the effect regardless of wording.

**E4.** Contract for `send_reply`:
- **Schema:** `{to: string (email), subject: string, body: string, reply_to_ticket: string}`, all
  required, with length caps on subject and body.
- **Authorization:** the gateway checks that the requesting end user is permitted to correspond on
  `reply_to_ticket` and that `to` is on the recipient allowlist for that user and tenant. The check uses
  the user's identity, never the agent's service identity.
- **Approval:** required. The approval request shows the exact `to`, `subject`, and `body`, and the
  approval binds to those concrete arguments; changing any argument voids it.
- **Idempotency:** an idempotency key derived from the approved arguments; the gateway records completed
  sends so a retry after a timeout does not send twice.
- **Timeout:** a bounded deadline on the external call, with the failure classified as retryable or not.
- **Model decides:** the proposed recipient, subject, and body text. **Code decides:** whether the
  recipient is allowed, whether the user is authorized, whether approval was granted for these exact
  arguments, and whether the action already executed.

**E5.** First hour, in order:

1. **Contain by capability (minutes 0 to 5).** Flip the `send_reply` kill switch for all tenants and put
   the support agent in read-only mode. The gateway blocked this call, but a blocked attempt proves an
   injection reached the planner with HR data in context, and other outbound paths (other tools, rendered
   links) may not be as well guarded. Disable link and image rendering in the answer pane if the same
   context could reach a UI.
2. **Find the carrier (5 to 25).** From the blocked span, follow the trace id to the request: principal,
   tenant, the context items with document ids and versions, and the tool results. Identify the item that
   carried the instruction (usually a retrieved ticket or document). Search the index for chunks from the
   same source and for other documents with the same payload signature, and search traces from the last
   retention window for other requests that retrieved those chunks, especially ones whose egress was
   allowed.
3. **Remove the poison (25 to 40).** Quarantine the source, delete its chunks and embeddings, invalidate
   answer caches that cite it, and review memory entries written by sessions that retrieved it.
4. **Rotate (40 to 50).** Rotate the canaries for the HR record class (they are now known to whoever
   reads the egress destination's logs) and any credential the agent's tools used in affected sessions.
   Check whether the HR record should have been in the agent's context at all: if the requesting user
   lacked scope, this is also a confused-deputy finding (A2).
5. **Decide on re-enable (50 to 60).** Re-enable only when the carrier is removed everywhere, the trace
   search shows no allowed egress of canaries in the window, the attack is added to the red-team corpus
   and passes in CI against the current controls, and the incident owner signs off. If any allowed egress
   was found, the incident escalates to a data-breach process, and the tool stays off.

## Practical exercises

**P1.** Expected implementation: add a `Variant` member (for example `ZERO_WIDTH` or `RTL_OVERRIDE`) and a
renderer that embeds the exfiltration instruction using zero-width spaces between characters or a
right-to-left override to visually reorder text, then register it in the renderer, effect, and
detection-hint maps. Acceptance: a test shows a keyword scan for the plain instruction fails to match the
rendered body, while an effect-based assertion (canary leak or off-allowlist URL after a simulated
compliance) still detects the attempt. The point the solution must demonstrate is that obfuscation
defeats detection but not effect-based control.

**P2.** Expected implementation: a `northwind_ingestion_model()` returning a `ThreatModel` with assets
(corpus integrity, hr-documents, tenant-isolation), principals (employee uploader as semi-trusted, the
uploaded file content as untrusted, the ingestion service as trusted), boundaries (upload to pipeline,
pipeline to index), and entry points (uploaded-file, extracted-text, embedding-write). At least five
threats: index poisoning via a malicious upload; a malformed file causing a parser crash or resource
exhaustion; mis-tagged ACL metadata causing later cross-tenant leakage; an oversized upload as denial of
wallet on embedding cost; and hidden-text injection surviving extraction. Acceptance: `validate()`
returns an empty list and `render_markdown()` produces a table; every threat names at least one control.

**P3.** Expected implementation: `egress_guard` extracts URLs and markdown image URLs, computes each
host, and for any host not on `allowed_hosts` replaces the URL (and the enclosing image markup) with a
neutral placeholder, returning the sanitized string. Acceptance: given the corpus markdown-image
exfiltration document, `off_allowlist_urls(egress_guard(doc.body, allowed), allowed) == []`, and a clean
answer with an allowlisted link passes through unchanged. Tests must assert the effect (no surviving
off-allowlist URL), not the presence of specific placeholder text.

**P4.** Expected implementation: a pytest module with one test per red-team item, each instantiating the
relevant corpus attack and asserting a blocked effect: no tool fires on injection, authorization denies a
scope-widening argument, no cross-tenant chunk is returned, no shared cache entry across ACLs, no
canary in the trace sink, a single execution under a repeated idempotency key, termination at the step
budget, schema rejection of malformed output, and no gated effect under a refusal bypass. Acceptance: at
least one test is first shown failing against a deliberately weakened control (for example an egress
allowlist that is empty-checks only), then passing after the control is restored, demonstrating the test
has real discriminating power.

## Debugging exercises

**D1.** Root cause: a confused-deputy failure, which is threat A2 escalating into A1. Two signals in the
trace show it. First, steps 2 and 3 carry `authz=ok(agent=svc-support)`: authorization was evaluated
against the agent's service identity, not against the requesting user `support_42`, so scope was never
checked for employee 4021. Second, step 3 has `approval=none` for a `send_reply` to an external address,
so the outbound send had no approval gate. The retrieved ticket body is the injection source. The
controls that would have blocked the effect: authorize `lookup_employee` and `send_reply` arguments
against the requesting user (A2), a recipient allowlist plus human approval bound to the concrete
arguments on `send_reply` (A1). The revealing fields are the `authz` principal (agent instead of user) and
`approval=none` on a consequential external action.

**D2.** This is cross-ACL cache leakage, threat R4, not a retrieval failure. Because both retrieval spans
show correct ACL-filtered chunks, the leak is downstream of retrieval: look at the answer cache. The
confirming field is the cache key. If the key is derived from the question text alone (or omits tenant and
the user's authorization context), a cached answer built from one user's documents is served to another
user asking a similar question. Confirm by logging the cache-key inputs and checking whether tenant and
the user's entitlement hash are present; their absence is the root cause. Fix per E2.

**D3.** This is memory poisoning (the memory-quality threat from Module 10's advanced note). The false
claim most plausibly entered as follows: an injected ticket or document asserted that logistics refunds
are pre-approved; the agent, lacking a write policy, summarized that assertion into long-term memory as a
fact; later tasks retrieved the memory and treated it as authoritative, so behavior changed with no code
or prompt change. The absent write-policy fields that allowed it: provenance (the entry does not record
that it came from untrusted document text rather than a system-of-record), confidence, owner, and expiry,
plus a write criterion preventing untrusted text from becoming durable state automatically. With
provenance recorded, the entry would be marked model-generated from untrusted input and would not be
trusted as a policy fact; with expiry, it would not persist indefinitely.
