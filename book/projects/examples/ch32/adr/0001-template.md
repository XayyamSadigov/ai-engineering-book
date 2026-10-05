<!-- path: book/projects/examples/ch32/adr/0001-template.md -->
# ADR NNNN: <decision in one line, imperative>

- Status: proposed | accepted | superseded by ADR-XXXX | deprecated
- Date: YYYY-MM-DD
- Deciders: <names or roles>
- Components affected: prompts | models | embeddings/index | tools | policies | evaluators | infra

## Context

What forces are at play: the user job, the constraint (latency, cost, data residency,
quality bar), and what we observed. Link evidence: eval report ids, trace queries, incident ids.

## Decision

What we will do, stated so that a reviewer can check compliance in a pull request.

## Alternatives considered

| Option | Why not |
|---|---|
| <option A> | <reason with evidence> |
| <option B> | <reason with evidence> |

## Evidence

- Offline eval: dataset `<name@version#hash>`, evaluator `<name@version#hash>`, results per slice.
- Online: experiment or canary id, primary metric, guardrails, sample size, duration.
- Version manifest fingerprint(s) of what was compared.

## Consequences

Positive, negative, and the new risks. What becomes harder. Cost and latency impact
(illustrative numbers labeled as such).

## Revisit when

Concrete triggers that reopen this decision: a metric threshold, a provider change, a new
requirement, a date. AI decisions decay faster than most; every ADR here names its trigger.

## Rollback

How to undo it, how long that takes, and which flag or version pin performs it.
