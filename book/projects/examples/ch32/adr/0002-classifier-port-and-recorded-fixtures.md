<!-- path: book/projects/examples/ch32/adr/0002-classifier-port-and-recorded-fixtures.md -->
# ADR 0002: Put ticket classification behind a ClassifierPort and test adapters with recorded fixtures

- Status: accepted
- Date: 2026-03-02
- Deciders: triage team lead, platform engineer
- Components affected: models, prompts, evaluators

## Context

Triage started as one function that built a provider request and parsed the reply inline.
Two problems followed. A provider SDK upgrade renamed a response field, and the break
surfaced in production because tests used a hand-written mock of the SDK. Separately, the
team wants to evaluate a smaller fine-tuned classifier served over HTTP, which would have
required editing the use-case code.

## Decision

The use case depends on `ClassifierPort` (application types only). Provider access lives in
`LLMClassifier`, which translates `aie_core` types and errors into `ClassifierOutput` and
`ClassifierUnavailable`. Adapter tests replay HTTP exchanges recorded with
`RecordReplayTransport`; CI runs with `CASSETTE_MODE=replay`, so any unrecorded request fails.

## Alternatives considered

| Option | Why not |
|---|---|
| Mock the provider SDK in tests | Mocks encode our belief about the SDK, which is exactly what broke |
| Call the real provider in CI | Slow, costly, nondeterministic, and needs secrets on every branch |
| Adopt a framework's model abstraction | Ties use-case code to the framework's types and release cycle |

## Evidence

- Incident: field rename after SDK upgrade, caught by users, not tests.
- Spike: swapping to an HTTP classifier touched two files (adapter, composition root).

## Consequences

Positive: provider swaps are local; adapter decoding and error mapping are tested offline.
Negative: cassettes must be re-recorded when prompts or parameters change, and reviewed for
PII before commit. One more layer of indirection for new contributors.

## Revisit when

A second use case needs the same port (consider promoting it to a shared package), or
cassette churn exceeds one re-record per week.

## Rollback

Not applicable at runtime; the port is a code structure. Reverting is a normal code change.
