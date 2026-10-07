# Chapter 20 — Agent Architectures

Once one agent loop works, every request for more (parallel checks, specialist agents, self-review) is a change of architecture that moves success rate, cost, latency, and debuggability at once. This chapter covers nine patterns (ReAct, router, planner-executor, supervisor and workers, hierarchical agents, reflection, evaluator-optimizer, parallel agents, and sequential workflows with agents inside), shows how to choose among them from evidence rather than fashion, and builds Project 5, the Northwind incident-research agent.

**You will be able to:**
- Choose an architecture with a decision procedure that starts from the simplest design and adds structure only in response to a named failure.
- Count best-case, typical, and worst-case model calls for a design before building it, and compare its token growth against a ReAct baseline.
- Implement each pattern as a composition of bounded `AgentRuntime` runs, single calls, and code, without writing a second agent loop.
- Build a planner-executor that validates typed plans and replans only when a deterministic deviation rule fires, within a replan budget.
- Build an evaluator-optimizer with deterministic checks before a judge, best-so-far protection, and plateau stops, and gate an irreversible action on human approval.
- Evaluate an architecture at three levels (outcome against a baseline, trajectory from events, each decision-maker as what it is) and replay recorded runs in CI.

**Prerequisites:** Chapters 17 (workflows, fan-out, the ladder from chain to agent) and 19 (`AgentRuntime`, budgets, Definition of Done, approval, replay). | **Code:** `book/projects/examples/ch20/` and `book/projects/p5-incident-agent/` (run: `cd book/projects/examples/ch20 && pytest -q`, then `cd ../../p5-incident-agent && pytest -q`) | **Builds:** Project 5, a planner-executor that searches runbooks and past incidents, reads metrics and deploy history, writes a cited report in an evaluator-optimizer loop, and posts it only after a human approves.

## Why this matters

Chapter 19 gave you a single, well-behaved loop. Production rarely stops there. The on-call team wants the incident agent to check three systems at once, a product manager wants a "triage agent" that hands work to specialists, someone reads about self-critique and wants the agent to review its own drafts. Each request is a change of architecture, and each architecture changes four numbers at once: how often the task succeeds, how many model calls it takes, how long the user waits, and how hard the next incident review will be.

Teams tend to get this wrong in one of two ways. The first is to treat architecture as a capability upgrade: a supervisor with five specialists must be smarter than one agent, reflection must improve quality, a plan must beat improvisation. None of these is true in general. A supervisor adds a routing decision that can be wrong and a hand-off that loses context; reflection without new evidence reinforces the original mistake; a plan written before the first observation can go stale by step two. The second mistake is never measuring the chosen design against the simpler alternative, so a design that costs four times as much as a plain ReAct loop survives unexamined.

The principle this chapter follows: choose a pattern by the task's dependencies and by how its output can be verified, and add structure only when failure analysis justifies it. This chapter turns that principle into a working method. You will see what each pattern buys, what it costs, what breaks, and how to tell from a trace which of those is happening.

## Mental model

> **Mental model:** An architecture decides who owns each decision, the model or the code, and how many model calls sit on the critical path. Every pattern is a different answer to those two questions.

Draw any agent system as a set of decisions: which tool next, which specialist, what the plan is, whether the answer is good enough, whether to try again, whether to stop. For each decision, ask who makes it. In a ReAct loop the model makes almost all of them inside one context. A router moves one decision (which specialist) to the front and makes it once. A planner-executor moves "what are the steps" out of the loop into a reviewable artifact and gives "when to change the plan" to deterministic rules. A supervisor turns delegation into a tool call the harness can count and limit. An evaluator-optimizer gives the "is it good enough" decision to an independent evaluator and the "try again or stop" decision to a controller in code.

The second question is arithmetic. Every model call on the critical path adds latency and an independent chance of error, and every call adds cost whether or not it is on the critical path. Chapter 17 showed that chains multiply success probabilities; agent architectures do the same, with the extra twist that the number of calls is itself a random variable. When you look at a proposed architecture, count the calls in the best case, the typical case, and the worst case allowed by its budgets. If you cannot state the worst case, the architecture has an unbounded loop somewhere, and production traffic will eventually reach it.

One more framing keeps the patterns from blurring together: none of them requires a new loop. Each is a composition of bounded `AgentRuntime` runs, single model calls, and ordinary code. If your implementation of a pattern contains its own "call the model, run the tools, append the result" loop, it has reimplemented the harness without the budgets, policy, events, and replay that make the harness safe. Every pattern in this chapter builds its agents on `AgentRuntime`, through the shared `make_agent` factory, and none of them loops over tool calls itself.

## Core concepts

This section starts with the decision: the nine patterns in one line each, a table of what each costs and how it fails, and a procedure for choosing. The pattern sections after it are reference material, one per pattern, each with its structure, when to use it, failure modes, cost profile, and evaluation.

### The patterns at a glance

- **ReAct:** one agent decides every next tool call from the whole transcript.
- **Router:** rules or one classification call pick exactly one specialist agent.
- **Planner-executor:** a typed plan is written first; each step runs as a small agent; rules decide when to replan.
- **Supervisor and workers:** an agent whose only tools delegate self-contained sub-tasks to worker agents.
- **Hierarchical agents:** supervisors whose workers are other supervisors.
- **Reflection:** a critic inside the Definition of Done rejects drafts, and the same agent revises in context.
- **Evaluator-optimizer:** controller code loops a generator against an independent evaluator and keeps the best candidate.
- **Parallel agents:** independent branch agents run concurrently and a fan-in policy merges them.
- **Sequential workflow with agents inside:** a fixed chain of steps where only the open-ended step is an agent.

The scripted comparison in `compare.py` runs all nine patterns on one Northwind task ("Trackline lookups are slow; find the likely cause and the documented fix") with the same fixture tools and a scripted model, and counts calls, runs, tool calls, and tokens. The counts show each structure's shape, not real costs or quality; on a real provider the ratios move, but the ordering rarely does.

| Pattern | Who decides the next step | Model calls (scripted, illustrative) | Agent runs | Tool calls | Critical-path latency | Main failure mode | Primary evaluation |
|---|---|---|---|---|---|---|---|
| ReAct | model, every step | 5 | 1 | 4 | sum of steps | wandering, context growth | trajectory metrics, termination reasons |
| Router | rules, then model once; specialist after | 6 | 1 | 4 | one short call plus specialist | misroute, mixed requests | route confusion matrix, per-route success |
| Planner-executor | planner up front; rules decide replans | 8 | 3 | 3 | plan + steps + synthesis | stale plan, replan thrash | plan validity, replan benefit, per-step success |
| Supervisor/worker | supervisor model via delegate tools | 9 | 3 | 6 | supervisor steps plus workers | redundant delegation, hand-off loss | delegation accuracy, single-agent baseline |
| Hierarchical | supervisors at every level | 13 | 5 | 8 | grows with depth | telephone effect, spawn explosion | per-level metrics, flat-supervisor baseline |
| Reflection | model, critic in the DoD | 8 | 1 | 4 | plus one critic and one revision per round | self-agreement, token burn | first draft vs accepted, critic calibration |
| Evaluator-optimizer | controller in code, independent evaluator | 12 | 2 | 8 | rounds times generator | Goodhart, plateau | pass rate by round, evaluator false-pass rate |
| Parallel | code fans out; aggregator merges | 7 | 3 | 3 | slowest branch plus merge | contradictions, silent partials | partial-answer rate, p95 vs sequential |
| Sequential | code, fixed order; agent inside one step | 7 | 1 | 4 | sum of steps | error propagation, rigidity | per-step metrics, path distribution |

Each row is a trade: each extra call must buy something specific, such as least privilege (router), a reviewable plan and linear token growth (planner-executor), context isolation (supervisor), or quality, which the two loops deliver only when their feedback carries information the generator lacked.

### Choosing and composing

A decision procedure that works in practice starts from the simplest design and adds structure only in response to a named failure, which mirrors Chapter 17's ladder:

1. If one call with retrieval and validation solves it, stop (Chapters 6 and 13; Chapter 1 covers when not to build an agent at all).
2. If the stages are known, build a sequential workflow. Make one stage an agent only if its path cannot be enumerated.
3. For that agent, start with ReAct, good tool descriptions, and a real Definition of Done.
4. If requests split into categories needing different tools or permissions, put a router in front.
5. If runs lose the objective on long horizons, or a plan must be reviewed before acting, move to planner-executor and write the deviation rules.
6. If outputs miss checkable criteria, add an evaluator, deterministic checks first; prefer an evaluator-optimizer over in-loop reflection when you need plateau stops or independent judging.
7. If independent sub-tasks dominate latency, fan them out.
8. Only if sub-tasks need isolated contexts or separate permission domains, and a single-agent baseline loses, use a supervisor. Use a hierarchy only when a flat supervisor demonstrably cannot choose well.

Real systems compose patterns, which is fine when each layer is bounded and measured separately. Project 5 is a sequential workflow of a planner-executor, an evaluator-optimizer, and an approval-gated action, in which every agent is a bounded `AgentRuntime` with one tool.

### The shared pieces

All nine patterns live in one small package. A few helpers keep them uniform: every agent is built by one factory, every system prompt carries a role tag so traces and test models can tell agents apart, and a metering wrapper counts calls and tokens per role without touching pattern code. Every pattern returns a `PatternResult` holding the answer, every agent run it made, and pattern-specific data such as a plan history or a delegation ledger.

```python
# path: book/projects/examples/ch20/patterns/common.py  (excerpt; full file on disk)
@dataclass
class PatternResult:
    """What every pattern returns: the answer, every agent run it made, and pattern-specific data."""

    pattern: str
    ok: bool
    answer: str | None
    runs: list[RunResult] = field(default_factory=list)
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    # ... tool_calls, agent_steps, trajectory() aggregate over runs


def make_agent(
    llm: LLMClient,
    # ... tools, role, instructions, budget, dod, store, config, principal
) -> AgentRuntime:
    """One bounded agent. Every pattern builds its agents through this function."""
    return AgentRuntime(llm, list(tools), system_prompt=role_prompt(role, instructions),
                        budget=budget or Budget(max_steps=5, max_tool_calls=5), dod=dod, store=store,
                        config=config, principal=principal)
```

The tools in the pattern examples are a tiny in-memory Northwind world (`fixtures.py`): `search_runbooks`, `search_policies`, `search_incidents`, `get_service_status`, and `query_metrics`. Each returns text that starts every fact with a citation key such as `[it-vpn-access-runbook]` or `[metric:pg-logi-prod.seq_scans_per_s]`, which is what `citations_grounded` from Chapter 19 checks against. Project 5 replaces the fixtures with real retrieval; the patterns do not change. Tests use a `Script` helper that turns a dictionary of per-role functions into a `FakeLLM` handler, so a test of a supervisor with two workers reads like three small functions rather than one tangled script.

### ReAct

```mermaid
flowchart LR
    G[goal] --> M[model decides]
    M -->|tool call| H[harness: validate, authorize, execute]
    H --> O[observation appended]
    O --> M
    M -->|final answer| V{Definition of Done}
    V -->|rejected, feedback| M
    V -->|accepted| A[answer]
```

**Structure.** ReAct, short for reasoning and acting, interleaves a decision, an action, and an observation until the objective is met. One model sees the whole transcript at every step and picks the next tool call or the final answer. `AgentRuntime` is a ReAct loop by construction, with the safety parts built in, so the pattern-level code is a configuration: which tools, which instructions, which budget, which Definition of Done.

```python
# path: book/projects/examples/ch20/patterns/react.py  (excerpt; full file on disk)
REACT_INSTRUCTIONS = (
    "Work in short cycles: decide what you still need to know, call one tool, read the result. "
    "Keep the objective, what you have learned, and what is still unknown in mind at every step. "
    "Stop as soon as the evidence answers the objective; cite every fact as [source-id]."
)

# ... def react(llm, goal, tools, *, budget, dod, store, config, principal) -> PatternResult:
    runtime = make_agent(llm, tools, role="react", instructions=REACT_INSTRUCTIONS,
                         budget=budget or Budget(max_steps=6, max_tool_calls=6), dod=dod, store=store,
                         config=config, principal=principal)
    run = runtime.run(goal)
    return PatternResult("react", run.ok, run.final_answer, [run],
                         detail=run.stop_reason.value if run.stop_reason else "")
```

**When to use it.** ReAct is the default agent: the next action depends on the last observation, the horizon is a handful of tool calls, the tool set fits one prompt, and one context window holds the whole investigation. Most features that justify an agent at all are best served by a ReAct loop with good tools and a real Definition of Done.

**Advantages.** The fewest moving parts and model calls for adaptive work; nothing is lost in hand-offs; the trace reads top to bottom.

**Failure modes.** *Wandering*: without a clear completion criterion the model keeps gathering evidence; the harness catches it as `REPEATED_ACTION`, `NO_PROGRESS`, or a budget stop, and the termination-reason distribution shows it. *Local myopia*: on long tasks the model optimizes the next step and loses the global objective, visible as a long run that ends with an answer to a narrower question than the one asked. *Context growth*: the transcript grows with every observation, so late steps are slow, expensive, and prone to ignoring early evidence. *Premature completion*: answering after one search, which the Definition of Done exists to reject.

**Cost profile.** Calls equal steps, typically the number of tool calls plus one. Input tokens grow roughly quadratically with steps because each call resends the transcript; Chapter 19 works that arithmetic in full. The reference numbers this chapter compares against (illustrative: a 1,500-token prompt, 600-token observations) are about 18,000 input tokens over six steps and about 57,600 over twelve. Latency is the sum of step latencies, all on the critical path.

**Evaluation.** Final-answer quality with the task's own metric, plus trajectory metrics from the event log: tool-selection accuracy against a labeled set, redundant calls per run, steps to completion, and the distribution of termination reasons. A rising share of `NO_PROGRESS` stops after a prompt change is a regression even when answer quality on completed runs looks flat.

### Router

```mermaid
flowchart LR
    R[request] --> K{keyword rules}
    K -->|match| S1
    K -->|no match| C[model classifies into closed label set]
    C -->|confident, known label| S1[specialist agent A]
    C -->|confident, known label| S2[specialist agent B]
    C -->|low confidence or invalid| F[fallback: human queue]
```

**Structure.** A router classifies the request once and hands it to exactly one specialist agent with its own tools, prompt, and budget. This is the agent-level cousin of Chapter 7's model router, which picks a model; here the route picks a whole configured agent. The implementation runs deterministic rules first, asks the model only when no rule matches, constrains the model's answer to a closed set of labels through the response schema, and sends low-confidence or malformed decisions to a fallback instead of to the nearest-sounding specialist.

```python
# path: book/projects/examples/ch20/patterns/router.py  (excerpt; full file on disk)
class AgentRouter:
    # ... __init__ stores specialists, rules, min_confidence, fallback
        names = tuple(self.specialists) + (fallback,)
        # The label space is closed: the schema itself rejects a route that does not exist.
        self._schema: type[BaseModel] = create_model(
            "RouteDecision",
            route=(Literal[names], ...),  # type: ignore[valid-type]
            confidence=(float, Field(ge=0.0, le=1.0)),
            reason=(str, ""),
        )

    def classify(self, request: str) -> tuple[str, str, float]:
        """Return (route, decided_by, confidence)."""
        for rule in self.rules:
            if rule.matches(request):
                return rule.route, "rule", 1.0
        # ... build a system prompt listing each specialist's name and description
        try:
            decision: Any = ask_structured(self.llm, "router", system, request, self._schema)
        except MalformedResponseError:
            return self.fallback, "fallback:malformed", 0.0
        if decision.route != self.fallback and decision.confidence < self.min_confidence:
            return self.fallback, "fallback:low_confidence", decision.confidence
        return decision.route, "model", decision.confidence

    # ... run(): classify, then one make_agent(...) run of the chosen specialist, or a fallback result
```

The self-reported confidence is a weak signal; models are often confidently wrong. Treat the threshold as a tunable that you calibrate on a labeled set, and prefer rules where the category is obvious. The same rule applies to supervisors later in the chapter: route obvious cases with deterministic rules, and use the model only where the decision needs semantic judgment.

**When to use it.** Requests fall into a few categories that need genuinely different tools, permissions, or instructions: HR questions, IT access problems, and incident investigations at Northwind. The specialists can then be smaller, cheaper, and more tightly permissioned than one generalist with every tool.

**Advantages.** Least privilege by construction, since the HR specialist cannot see incident tools. Shorter prompts per specialist. Independent evaluation and ownership per route. One extra call at most, and zero for rule-matched requests.

**Failure modes.** *Misroute*: the request lands with a specialist who lacks the tools, which shows as a specialist run ending in `NO_PROGRESS` or `VERIFICATION_FAILED` with a route whose confidence was marginal. *Mixed requests*: "my VPN is broken and I need to know my PTO balance" belongs to two routes; a single-label router answers half. *Label drift*: a new category appears in traffic and the router forces it into the closest old label. *Fallback flood*: a threshold set too high sends most traffic to humans.

**Cost profile.** One classification call (often a small model), plus the specialist's run. Latency adds one short call before the first token of real work.

**Evaluation.** Treat the router as a classifier (Chapter 24): a confusion matrix over routes on a labeled set, per-route precision and recall, the fallback rate, and accuracy as a function of the confidence threshold. Then measure end-to-end success per route, because a correct route to a weak specialist still fails.

### Planner-executor and replanning

```mermaid
flowchart TD
    G[goal] --> P[planner: typed plan]
    P --> V{validate plan}
    V -->|invalid, one repair| P
    V -->|valid| Q[step queue]
    Q --> E[execute step: bounded agent with only that step's tools]
    E --> D{deviation rules}
    D -->|none| Q
    D -->|deviation and replans left| P
    D -->|deviation, no replans left| X[stop: replan budget exhausted]
    Q -->|empty| S[synthesize answer from ledger]
```

**Structure.** A planner writes a bounded, typed plan before any tool runs. Each step then executes as its own small agent that sees only the tools that step names and only the findings it depends on. After each step, deterministic deviation rules decide whether the plan still holds; when a rule names a concrete deviation, the planner is consulted again with the completed steps, the remaining queue, and the reason, up to a replan budget. A final synthesis call writes the answer from the ledger of findings.

Holding the plan as data is what makes it useful. It can be validated before anything runs (unknown tools, duplicate ids, dependencies on steps that do not run earlier, which also rules out cycles), shown to a human, logged, and diffed across runs. A useful plan has two properties: every step is verifiable, and the plan is bounded. "Research everything" is not a plan; "find three current policy documents, compare eligibility clauses, then produce a cited summary" is, because each step has an observable completion.

```python
# path: book/projects/examples/ch20/patterns/planner_executor.py  (excerpt; full file on disk)
class PlanStep(BaseModel):
    id: str = Field(pattern=r"^s[0-9]+$")
    objective: str = Field(min_length=8)
    tools: list[str] = Field(min_length=1, max_length=2)
    target: str | None = None                 # the entity the step is about, e.g. a service name
    depends_on: list[str] = Field(default_factory=list)


# ... validate_plan(): unique ids, known tools, and every dependency completed or earlier in the list


    # ... PlannerExecutor.run(), after the first plan() call:
        queue = list(plan.steps)
        while queue:
            if len(done) >= self.max_steps:
                return self._result(False, None, done, plans, f"executed {len(done)} steps; step cap reached")
            step = queue.pop(0)
            outcome = self.execute(goal, step, done, f"{run_id}{SEP}{step.id}-{len(done) + 1}")
            done.append(outcome)
            reason = next((r for rule in self.deviation_rules if (r := rule(outcome, queue))), None)
            if reason is None:
                continue
            if replans >= self.max_replans:
                return self._result(False, None, done, plans, f"replan budget exhausted after: {reason}")
            replans += 1
            try:
                plan = self.plan(goal, done, reason)
            except MalformedResponseError as exc:
                return self._result(False, None, done, plans, f"replanning failed: {exc}")
            plans.append(plan)
            finished = {o.step.id for o in done}
            queue = [s for s in plan.steps if s.id not in finished]
        # ... one synthesis call writes the answer from the findings ledger
```

Two methods stay on disk. `plan` asks for a `Plan` through the response schema, validates it, and allows one repair round that quotes the validation errors back before giving up. `execute` builds the step's agent with only the tools the step names, a Definition of Done that requires the first of them to have been called, and only the findings listed in `depends_on`. The loop above is the part to study: a deviation rule returns a reason or nothing, the reason goes to the planner, completed steps are never rerun, and every exit names why it stopped.

**Replanning on deviation, not on every step.** A replan is a model call that can make things worse: it can drop a step that was about to succeed, reorder dependencies, or chase the latest observation at the expense of the goal. So the default is to keep executing the plan and consult the planner only when a rule names a meaningful deviation.

Useful rules are concrete and cheap: a step failed; a search returned nothing, so the plan's assumption about where to look was wrong; a step revealed an entity the plan never mentions (Project 5's rule: an anomalous dependency that no planned step examines); a precondition of a later step is now false. Each rule returns a reason string, and that reason is what the planner sees, which makes replans explainable in the trace. Treat a plan as a hypothesis about how the work will go; the deviation rules are the written-down conditions for revising it.

**When to use it.** Longer tasks where a reactive loop loses the global objective; tasks where a human or a policy should review the steps before they run; tasks whose steps need different tools or permissions; and tasks where you want per-step budgets and per-step evaluation.

**Advantages.** A reviewable artifact before any side effect, least privilege per step, roughly linear token growth, per-step success rates, and a natural place to insert approval.

**Failure modes.** *Stale plan*: the environment changed after planning and no rule noticed, visible as steps that succeed mechanically but answer the wrong question. *Over-planning*: twelve steps for a three-step task, each paying a model call. *Replan thrash*: a rule that fires too easily, visible as `replans` near the budget on most runs. *Lost context between steps*: a step needs a finding it was not given because `depends_on` was wrong. *Invalid plans*: unknown tools or impossible targets, caught by validation before execution.

**Cost profile.** One planning call, two or so calls per step (call the tool, report), one synthesis call, plus one call per replan. Using the earlier illustrative numbers with 700-token step prompts, two calls per step (the second also carrying the step's 600-token observation), and 1,500-token planning and synthesis calls, six steps cost about 15,000 input tokens and twelve steps about 27,000, versus 18,000 and 57,600 for ReAct. Latency is higher than ReAct for short tasks because of the planning and synthesis calls, and lower for long tasks because each step's call is small.

**Evaluation.** Plan validity rate; plan quality judged against gold step sets or with a rubric; replan rate and, more importantly, whether replans helped (success on runs with a replan versus similar runs without); per-step success; and end-to-end success against a ReAct baseline on the same cases.

### Supervisor and workers

```mermaid
flowchart TD
    U[request] --> SV[supervisor agent]
    SV -->|delegate_incident_analyst task| T1[delegate tool]
    SV -->|delegate_runbook_finder task| T2[delegate tool]
    T1 --> SB{spawn budget and per-worker cap}
    T2 --> SB
    SB -->|allowed| W1[worker run: own tools, own budget]
    SB -->|allowed| W2[worker run: own tools, own budget]
    SB -->|refused| ERR[permission error observation]
    W1 -->|final answer only| SV
    W2 -->|final answer only| SV
    SV --> L[(task ledger)]
```

**Structure.** A supervisor is an agent whose only tools are "delegate to worker X." Each delegation starts a bounded child `AgentRuntime` with its own tools, prompt, budget, and Definition of Done. The supervisor sees only the worker's final answer, never its intermediate tokens, which is the context isolation that makes the pattern worth having. A ledger records every delegation. Child run ids are derived from the parent's (`supervisor.incident_analyst-1`), and the child's `GoalSet` metadata carries the parent run id and request id, so a trace can follow a request from parent to child.

The separator rule is book-wide: `.` (the `SEP` constant) separates levels and nothing else, `-` joins words inside a segment, and `/` is never used because the JSONL event store turns run ids into file names. A tool can then read a run's depth by counting dots, as `run_hierarchy` does, and Chapter 22's research team derives its ids the same way.

```python
# path: book/projects/examples/ch20/patterns/supervisor.py  (excerpt; full file on disk)
class Supervisor:
    # ... constructor and run() on disk; SpawnBudget.acquire(depth) returns a refusal reason or None
    def _delegate_tool(self, member: Member) -> FunctionTool:
        def delegate(ctx: ToolContext, task: str) -> ToolOutput:
            used = sum(1 for e in self.ledger if e.member == member.name and e.parent_run_id == ctx.run_id)
            if used >= self.max_delegations:
                return ToolOutput.failure(f"{member.name} already received {used} tasks in this run",
                                          ErrorClass.PERMISSION)
            depth = ctx.run_id.count(SEP) + 1
            refused = self.spawn.acquire(depth)
            if refused:
                return ToolOutput.failure(refused, ErrorClass.PERMISSION)
            child_id = f"{ctx.run_id}{SEP}{member.name}-{used + 1}"
            run = self._run_member(member, task, child_id, ctx)
            # ... append a LedgerEntry for this delegation
            if not run.ok:
                return ToolOutput.failure(f"{member.name} stopped with {run.stop_reason.value if run.stop_reason else '?'}:"
                                          f" {run.detail}", ErrorClass.SEMANTIC)
            return ToolOutput(content=f"[{member.name} result] {run.final_answer}", data={"child_run_id": child_id})
        # ... wrapped as FunctionTool(f"delegate_{member.name}", ...) with a required "task" string
```

Delegation as a tool call is the design choice everything else follows from: argument validation, the identical-call detector, the tool-call budget, and the event log all apply to hand-offs without extra code. The pattern adds only a spawn budget shared by the whole tree and a per-worker cap. A worker that fails returns a `semantic` failure observation, so the supervisor learns that the sub-task did not complete and why, rather than receiving an empty answer it might paraphrase as a finding.

**When to use it.** Sub-tasks need different tools or permission domains (the analyst may read metrics; the runbook finder may read the incident archive); their intermediate work is large and the parent only needs a summary; or they can be evaluated and owned separately.

The pattern here is the minimal, bounded form. Chapter 22 adds what it leaves out once workers become a team: typed task and result envelopes instead of a result string (section "Message contracts"), a shared token and cost pool split by reserve-then-settle instead of a fixed `Budget` per worker ("Budgets at parent and child"), trace ids that survive thread and process boundaries ("Trace propagation"), a supervisor whose dispatch logic is code rather than a model, and a benchmark that asks whether the extra agents were justified at all.

**Advantages.** Context isolation keeps the supervisor's transcript small. Each worker has least privilege and its own budget. Worker quality can be measured and improved independently.

**Failure modes.** *Redundant delegation*: the supervisor asks two workers the same question, or one worker twice with reworded tasks that slip past the identical-call detector; the per-worker cap and the ledger make it visible. *Context loss at the hand-off*: the worker sees only the task string, so a task like "check the thing we discussed" fails; tasks must be self-contained, and the instructions say so. *Supervisor as single point of failure*: if the supervisor misreads a worker's answer, every worker's quality is wasted. *Runaway spawning*: a confused supervisor keeps delegating; the spawn budget refuses with a permission error the supervisor can read. *Paraphrase drift*: the supervisor rewrites worker findings and drops their citations; requiring `citations_grounded` on the supervisor's answer catches it, because the worker answers, citations included, are the supervisor's observations.

**Cost profile.** Supervisor steps (delegations plus one) plus the sum of worker runs. Workers' tokens are often lower than a single agent's would be, since each sees a narrow task; the supervisor's are low because it sees summaries. The total is usually higher than ReAct on the same task, by the cost of the supervisor's own calls. Latency is the sum of sequential delegations unless the supervisor issues several delegate calls in one step and the workers run concurrently.

**Evaluation.** Delegation accuracy (was the right worker chosen, judged against labels), redundant delegations per run, child failure rate and how the supervisor handled it, and cost and success against a single-agent baseline with the union of tools. The baseline comparison is the one that justifies the pattern; skip it and you will not know whether the supervisor paid for itself.

### Hierarchical agents

```mermaid
flowchart TD
    ROOT[ops supervisor] --> D[diagnostics supervisor]
    ROOT --> K[knowledge worker]
    D --> S[status_checker worker]
    D --> M[metrics_reader worker]
    subgraph Shared limits
        SB[(spawn budget: agents and depth)]
        ES[(one event store)]
    end
```

**Structure.** Hierarchical agents are supervisors whose members are other supervisors. No new mechanism is needed: a `Supervisor` is a valid member of another `Supervisor`. What the hierarchy adds is shared limits (one spawn budget and one event store for the whole tree), run ids that encode the path from the root (`ops.diagnostics-1.status_checker-1`), and a ledger that rolls up from every level.

```python
# path: book/projects/examples/ch20/patterns/hierarchical.py  (excerpt; full file on disk)
def build_tree(llm: LLMClient, team: Team, *, spawn: SpawnBudget | None = None,
               store: EventStore | None = None) -> Supervisor:
    members = [build_tree(llm, m, spawn=spawn, store=store) if isinstance(m, Team) else m for m in team.members]
    return Supervisor(llm, members, name=team.name, description=team.description, budget=team.budget,
                      spawn=spawn, store=store)
```

`run_hierarchy` (on disk) builds the tree with one shared `SpawnBudget` (ten agents, depth three by default), runs the root, and records the deepest run id it reached.

**When to use it.** Rarely, and only when the organization of the work is itself hierarchical: a task large enough that one supervisor's menu of workers would be too long to choose from well, with sub-teams whose members genuinely share context that the root does not need. A research system that splits "diagnose the platform" into "diagnose the database tier" and "diagnose the delivery tier," each with several specialists, is the shape that fits.

**Advantages.** Each supervisor chooses among a short menu. Contexts stay small at every level. Teams can be developed and tested as units.

**Failure modes.** Everything a supervisor can get wrong, multiplied by depth. *Telephone effect*: each level summarizes, so details and citations erode on the way up. *Spawn explosion*: the number of agents grows with branching factor to the power of depth; the shared spawn budget is not optional. *Diffuse responsibility*: when the answer is wrong, it is hard to tell which level failed without the derived run ids. *Latency stacking*: every level adds at least two sequential calls.

**Cost profile.** The most expensive pattern in this chapter: in the scripted comparison below, the three-level tree made thirteen model calls where ReAct made five on the same task. Worst case grows multiplicatively with depth, so set depth and agent limits explicitly.

**Evaluation.** As for a supervisor, at each level, plus depth and spawn distributions, citation survival from leaf to root (count citations in leaf answers that appear in the root answer), and end-to-end comparison against a flat supervisor with the same leaves. If the flat version is as good, flatten.

### Reflection

```mermaid
flowchart LR
    A[agent proposes final answer] --> O{objective checks}
    O -->|fail| N[harness note: problems]
    O -->|pass| C[model critic with evidence]
    C -->|score below threshold| N
    C -->|pass| ACC[accepted]
    N --> A
```

**Structure.** Reflection adds a critique and revision stage: generate a candidate, evaluate it against criteria, improve it. In `agentkit` the natural place for the critic is the Definition of Done. A rejected final answer becomes a harness note that the model reads on its next step, with every observation still in its context, so a revision costs one more model call rather than a new run. The implementation runs objective checks first and calls the model critic only when they pass, and the critic sees the evidence the agent actually read.

```python
# path: book/projects/examples/ch20/patterns/reflection.py  (excerpt; full file on disk)
    # ... CriticCheck: a Definition-of-Done verifier that runs objective checks, then a model critic
    def __call__(self, answer: str, state: AgentState) -> Verdict:
        problems = [p for check in self.objective for p in check(answer, state)]
        if problems:   # cheap, trustworthy signals first; no model call spent on a known-bad answer
            self.history.append({"kind": "objective", "problems": problems})
            return Verdict(self.name, False, "objective checks failed: " + "; ".join(problems))
        evidence = state.observation_text()[-6000:]
        try:
            c: Any = ask_structured(self.llm, "critic", CRITIC_SYSTEM,
                                    f"Criteria: {self.criteria}\n<evidence>\n{evidence}\n</evidence>\n"
                                    f"<answer>\n{answer}\n</answer>", Critique)
        except MalformedResponseError:
            return Verdict(self.name, False, "critic unavailable; answer not accepted")
        self.history.append({"kind": "critic", "score": c.score, "problems": c.problems})
        if c.score >= self.pass_score:
            return Verdict(self.name, True)
        return Verdict(self.name, False, f"score {c.score}/5; problems: {c.problems}; fix: {c.fix}")
```

This sets the design constraint: repeated self-critique without new evidence mostly consumes tokens, and a model critiquing itself in the same context tends to reinforce its own misconception. Reflection helps when the critique has access to an objective signal: tests, a schema, retrieved evidence, a compiler, an independent model. That is why the critic here is a separate call with its own prompt, why it sees the observations rather than only the answer, and why objective checks run first. A critic that only rereads the answer in the same context is the weakest version and rarely worth its tokens.

**When to use it.** The output has checkable properties the generator tends to miss on the first pass: a required element, a citation per claim, consistency with the retrieved evidence. Coding agents are the clearest case, where the "critic" is the test suite.

**Advantages.** Revision happens with full context and costs one step. Objective feedback is specific, which makes the revision targeted. The bound (`max_dod_rejections`) is enforced by the runtime.

**Failure modes.** *Sycophantic critic*: the critic approves everything, visible as a near-100 percent first-pass acceptance with no quality change. *Contrarian critic*: it always finds something, visible as runs ending in `VERIFICATION_FAILED` with answers that a human rates as fine. *Self-agreement*: the same model in the same frame of mind agrees with itself; use a different prompt, ideally a different model. *Oscillation*: the revision fixes one criterion and breaks another. *Token burn*: each rejected round resends the full transcript.

**Cost profile.** One critic call per final answer that passes the objective checks, plus one generator step per revision. The revision step is expensive because it carries the whole transcript; the critic call is moderate because it carries evidence and the answer.

**Evaluation.** Measure the paired difference between first drafts and accepted answers on the same cases with an independent evaluator (Chapter 24's paired bootstrap). Track revisions per run, critic agreement with human labels, and the false-rejection rate. If accepted answers are not better than first drafts, the critic adds cost without adding quality.

### Evaluator-optimizer

```mermaid
flowchart TD
    T[task] --> GEN[generator: bounded agent run]
    GEN --> EV[evaluator: deterministic checks, then judge]
    EV -->|pass| OK[accept]
    EV -->|fail| CTL{controller}
    CTL -->|improved and rounds left| GEN
    CTL -->|no improvement: plateau| BEST[return best so far, not accepted]
    CTL -->|round budget spent| BEST
```

**Structure.** A generator produces a candidate, a separate evaluator scores it, and a controller in code decides: accept, revise with the evaluator's feedback, or stop. It differs from reflection in three ways. The evaluator is independent of the generator (different prompt, often a different model, ideally deterministic checks first). The controller owns the loop, so it can implement stop rules the runtime cannot: pass, round budget, and plateau, meaning no meaningful improvement for a set number of rounds. And the best candidate so far is always kept, so a late regression never replaces an earlier good draft.

```python
# path: book/projects/examples/ch20/patterns/evaluator_optimizer.py  (excerpt; full file on disk)
    # ... checks_then_judge(): deterministic checks gate; the judge scores only candidates that pass them
    def evaluate(candidate: str) -> Evaluation:
        problems = [p for c in checks if (p := c(candidate))]
        if problems:
            return Evaluation(passed=False, score=0.0, feedback=problems)
        # ... otherwise one structured judge call; a broken judge fails the round, the best draft survives


    # ... EvaluatorOptimizer.run(): the controller owns the loop
        for round_no in range(1, self.max_rounds + 1):
            candidate, run = self.generate(previous, feedback, round_no)
            # ... record the run; a generator that produces nothing ends the loop
            ev = self.evaluate(candidate)
            self.rounds.append({"round": round_no, "score": ev.score, "passed": ev.passed, "feedback": ev.feedback})
            improved = best is None or ev.score >= best[0] + self.min_improvement
            if best is None or ev.score > best[0]:
                best = (ev.score, candidate)
            if ev.passed:
                return self._result(True, (ev.score, candidate), runs, f"accepted in round {round_no}")
            stale = 0 if improved else stale + 1
            if stale >= self.patience and round_no > 1:
                return self._result(False, best, runs, f"plateau after round {round_no}")
            previous, feedback = candidate, ev.feedback
        return self._result(False, best, runs, f"round budget of {self.max_rounds} exhausted")
```

Ordering the evaluator matters for cost and for signal. Deterministic checks are free, exact, and unambiguous, so they go first and their failures become the most useful feedback. The judge runs only on candidates that pass them, which saves its cost on known-bad drafts and keeps it focused on what rules cannot check. The judge itself is a measurement instrument that must be calibrated against humans (Chapter 24) before its threshold means anything.

One subtlety separates this pattern from reflection in practice. Each round of `agent_generator` is a fresh agent run, so tools execute again unless you memoize read-only results or hand the generator the previous evidence. The scripted comparison shows it: the evaluator-optimizer made eight tool calls where ReAct made four, because round two re-gathered everything. Project 5 avoids this by separating evidence gathering (planner-executor, done once) from writing (the only thing the loop repeats).

**When to use it.** Quality criteria are explicit and checkable, first attempts often miss them, and feedback is actionable: reports with required structure, generated SQL that must parse and return the right columns, translations against a glossary, code against tests.

**Advantages.** Independent verification; explicit, testable stop rules; best-so-far protection; a natural record of rounds for evaluation.

**Failure modes.** *Goodhart*: the generator learns to satisfy the evaluator rather than the task, for example by adding citations that are technically grounded but irrelevant. *Uninformative feedback*: "make it better" produces random changes and a plateau. *Evaluator drift*: the judge's model or prompt changes and pass rates move without any change in quality. *Infinite polish*: without a plateau rule, the loop spends the round budget on marginal gains.

**Cost profile.** Rounds times generator cost, plus evaluator cost for candidates that reach the judge. With deterministic gating the judge often runs once. Latency is sequential across rounds, so the round budget is also a latency budget.

**Evaluation.** Pass rate by round (how often round one passes, how often the loop eventually passes), the score trajectory, plateau rate, evaluator calibration against humans (agreement and Cohen's kappa), and cost per accepted output. Watch the share of accepted outputs whose quality a human rejects; that is the evaluator's false-pass rate and the number that bounds how much you can trust the loop.

### Parallel agents: fan-out and fan-in

```mermaid
flowchart LR
    R[request] --> F[fan-out]
    F --> B1[branch agent: status]
    F --> B2[branch agent: metrics]
    F --> B3[branch agent: history]
    B1 --> J{fan-in policy: all, quorum, any}
    B2 --> J
    B3 --> J
    J -->|enough| AGG[aggregator: merge, report gaps and conflicts]
    J -->|not enough| FAIL[refuse to answer]
```

**Structure.** Independent sub-tasks run as bounded agents concurrently; a fan-in step merges their results under an explicit completeness policy. Chapter 17 introduced fan-out over plain steps; here each branch is an agent with its own tools and budget. `AgentRuntime` is synchronous, so branches run in a thread pool whose size is the concurrency limit you set to respect provider rate limits. If you pass a tracer, note that `aie_core` links each span to its parent through context variables, and a thread pool does not inherit them; submit each branch through `contextvars.copy_context().run`, as Chapter 22's research team does, or every branch starts an orphan trace.

```python
# path: book/projects/examples/ch20/patterns/parallel.py  (excerpt; full file on disk)
def _enough(ok: int, total: int, require: Require) -> bool:
    return total > 0 and {"all": ok == total, "quorum": ok * 2 > total, "any": ok >= 1}[require]


    # ... inside fan_out(): run_branch builds one make_agent(...) per branch with run id f"{run_id}{SEP}{b.name}"
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(run_branch, b) for b in branches]
    results: list[tuple[Branch, RunResult | None, str]] = []
    for b, f in zip(branches, futures):           # results in branch order, never completion order
        try:
            results.append((b, f.result(), ""))
        except Exception as exc:  # noqa: BLE001 - a crashed branch is a failed branch, recorded with its cause
            results.append((b, None, f"{type(exc).__name__}: {exc}"))
    # ... succeeded, missing, per-branch status
    if not _enough(len(succeeded), len(branches), require):
        return PatternResult("parallel", False, None, runs,
                             detail=f"{len(succeeded)}/{len(branches)} branches succeeded; require={require}", data=data)
    answer = merge(succeeded, missing) if merge else _synthesize(llm, request, succeeded, missing)
```

The design decision lives in the fan-in. `require="all"` refuses to answer if any branch failed, which is right when a partial answer is a wrong answer (a compliance check across three systems). `"quorum"` needs a strict majority, which suits redundant branches that check the same thing differently. `"any"` answers from whatever succeeded but must name the missing branches, and the aggregator prompt is told to report disagreements rather than pick a winner silently. Results are returned in branch order, not completion order, so the aggregator's input and the trace are deterministic.

**When to use it.** Sub-tasks are independent (no branch needs another's output) and latency matters: the incident agent checking service health, database metrics, and incident history at once. Also for redundancy: the same question to two differently configured agents, compared at fan-in.

**Advantages.** Latency drops from the sum of branches to the slowest branch plus the merge. Branches are isolated: one failing does not corrupt another's context.

**Failure modes.** *Hidden dependencies*: a branch needed another's result and guessed instead. *Contradictory outputs*: branches disagree and the aggregator picks one without saying so. *Rate-limit storms*: fan-out multiplies concurrent calls; without a pool limit you hit provider limits and every branch retries at once. *Silent partials*: an aggregator that writes a confident answer from two of three branches.

**Cost profile.** The sum of branch costs plus the aggregator, the same as running them sequentially plus one call. Only latency improves.

**Evaluation.** Per-branch success, partial-answer rate under each policy, contradiction handling (seed cases where branches must disagree and check the merge reports it), and p95 latency against the sequential version.

### Sequential workflows with agents in the steps

```mermaid
flowchart LR
    I[request] --> T[triage: one structured call]
    T --> G1{gate: is it an incident?}
    G1 -->|no| STOP[stop with reason]
    G1 -->|yes| A[investigate: bounded agent]
    A --> W[write: one model call]
    W --> OUT[answer]
```

**Structure.** A prompt chain whose order is fixed by code, where each step is a single model call, an agent, or plain code, with typed state passed between steps and an optional gate after each. The agent sits inside the one step where the work is open-ended; the chain around it stays deterministic and testable. This is Chapter 17's advice applied: graduate the smallest node that needs an agent, not the whole workflow.

```python
# path: book/projects/examples/ch20/patterns/sequential.py  (excerpt; full file on disk)
    # ... agent_step(): one bounded agent inside a chain step
    def fn(state: State) -> State:
        runtime = make_agent(llm, tools, role=name, instructions=instructions, budget=budget, dod=dod, store=store)
        run = runtime.run(goal(state))
        ctx.runs.append(run)
        if not run.ok:
            raise StepFailed(name, f"agent stopped with {run.stop_reason.value if run.stop_reason else '?'}: {run.detail}")
        return {**state, output: run.final_answer}


    # ... run_chain(): fixed order, a gate after each step, a step failure ends the chain
    for step in steps:
        try:
            state = step.fn(state)
        except StepFailed as exc:
            return PatternResult("sequential", False, None, ctx.runs, detail=str(exc), data={"path": path, "state": state})
        path.append(step.name)
        problem = step.gate(state) if step.gate else None
        if problem:
            return PatternResult("sequential", False, None, ctx.runs, detail=f"gate after {step.name}: {problem}",
                                 data={"path": path, "state": state})
```

**When to use it.** Most of the time when someone asks for "an agent." If you can name the stages (triage, investigate, write, review), make them a chain and let only the stage whose path cannot be predicted be an agent.

**Advantages.** Predictable order, per-step metrics, cheap gates that stop bad inputs before expensive steps, and a trace a reviewer can read. Steps that are plain calls are far cheaper and faster than agents.

**Failure modes.** *Error propagation*: a wrong triage label sends a request down the wrong chain; gates mitigate it. *Rigidity*: a case needs a stage the chain does not have. *Agent step failure*: the agent inside a step stops without completing, which the chain must treat as a step failure rather than passing `None` downstream.

**Cost profile.** The sum of steps; latency is the sum. Each non-agent step is one call.

**Evaluation.** Per-step metrics (the triage classifier's confusion matrix, the agent step's success and trajectory metrics), gate rejection rates, path distribution, and end-to-end success.

## How it works

Project 5 is the Northwind incident-research agent. An alert fires; the on-call engineer asks for an investigation; the system returns a cited report and waits for an incident commander to approve posting it to `#incidents`. One investigation of the Trackline latency alert from the sample data goes like this.

```mermaid
sequenceDiagram
    participant U as on-call (CLI or API)
    participant S as IncidentService
    participant P as planner call
    participant X as step agents
    participant W as writer call
    participant E as DoD + judge
    participant R as publish run
    participant H as incident commander
    U->>S: investigate ALR-2026-0914-01
    S->>P: alert, known services
    P-->>S: plan s1..s4
    S->>X: s1 query_service_metrics(trackline)
    X-->>S: finding + evidence, dependency pg-logi-prod anomalous
    S->>P: deviation: uncovered dependency
    P-->>S: s5, s6 for pg-logi-prod, then s2..s4
    S->>X: s5, s6, s2, s3, s4, one agent each
    S->>W: evidence ledger + findings
    W-->>E: draft 1
    E-->>W: DoD fails: invented runbook, uncited claim
    W-->>E: draft 2
    E-->>S: DoD passes, judge 5/5
    S->>R: post_report proposed
    R-->>S: paused: APPROVAL_REQUIRED
    H->>S: approve (possibly another process, hours later)
    S->>R: resume(approve=True)
    R-->>S: posted MSG-00001, once
```

The planner sees the alert and the list of known services and returns a typed plan: read Trackline's metrics, list its recent deploys, search past incidents, search runbooks. Plan validation rejects unknown services, repeating a step that already succeeded, missing runbook search, and plans larger than the remaining step budget, with one repair round that quotes the errors back.

Each step runs as its own `AgentRuntime` with exactly one tool, a three-step budget, and a Definition of Done that requires the tool to have been called and the finding to cite what it observed or state that the tool found no evidence. The first step's metrics tool flags Trackline's p95, error rate, and webhook retries, and reports that dependency `pg-logi-prod` is anomalous while `webhook-dispatcher` is normal. The deviation rule `uncovered_dependency` sees that no planned step examines `pg-logi-prod` and returns a reason. The planner, given the completed step, the remaining queue, and the reason, returns two new steps for the database (metrics, then deploys) followed by the steps it already had. That is the only replan; the remaining steps confirm the picture: sequential scans on the database rose about seventy-fold after a migration at 05:15, and a February postmortem describes the same pattern.

Evidence is collected from tool data, not parsed from text: every research tool returns a list of sources (id, kind, title, excerpt) that the orchestrator folds into an evidence ledger. The writer turns the ledger and step findings into six sections, citing a ledger id in every claim, and the deterministic Definition of Done checks exactly that (bullets, sentences, and table rows alike), plus that the recommended runbook exists and was retrieved. In the offline run the first draft recommends `it-db-index-rebuild-runbook`, which does not exist, and drops a citation; the checker returns four specific problems: one uncited claim and three caused by the invented runbook (`unknown_citation`, `runbook_not_in_catalog`, `runbook_missing`), the writer revises, the second draft passes, and only then does the rubric judge read it.

The report is saved, and a publish run starts whose only tool, `post_report`, is marked `EXTERNAL`. `agentkit`'s default policy requires approval for external side effects, so the run stops with `APPROVAL_REQUIRED` after recording the proposed call. Nothing has been posted. When the incident commander approves, from the CLI or the API and possibly from a different process, the service rebuilds the run from its JSONL event log and resumes it; the tool executes once with an idempotency key derived from the run and request ids; the record moves to `published`.

## Architecture

```mermaid
flowchart TB
    subgraph Clients
        CLI[p5 CLI]
        API[FastAPI: X-User header]
    end
    subgraph Service["IncidentService (trusted)"]
        DIR[user directory: groups, tenant]
        ORCH[IncidentResearchAgent]
        DOD[deterministic DoD]
        PUB[Publisher: AgentRuntime + approval]
        MET[MeteredLLM: call ceiling]
    end
    subgraph Models["Model calls (untrusted output)"]
        PL[planner]
        EX[step agents]
        WR[writer]
        JU[rubric judge]
    end
    subgraph Data["Read-only sources"]
        KB[KnowledgeBase: ragkit chunks + BM25 + ACL]
        TEL[Telemetry: metrics, deploys]
    end
    subgraph Untrusted["Untrusted content"]
        DOCS[runbook and incident text]
    end
    subgraph State["Durable state (P5_STATE_DIR)"]
        INV[(investigations JSON)]
        EVT[(agentkit JSONL event logs)]
        CH[(channel log)]
    end
    CLI --> ORCH
    API --> DIR --> ORCH
    ORCH --> MET --> PL & EX & WR & JU
    EX --> KB & TEL
    DOCS --> KB
    ORCH --> DOD
    ORCH --> PUB --> CH
    ORCH --> INV
    EX --> EVT
    PUB --> EVT
```

Three boundaries matter. Identity enters only through the server-side directory: the API maps the `X-User` header (standing in for your authentication proxy) to groups and a tenant, and the tools read the principal from `ToolContext`, never from model-supplied arguments, so the model cannot widen its own access. Retrieved documents are untrusted: they are escaped and wrapped in `<untrusted_data>` markers, both in tool output and in the writer's ledger (Chapter 26), and the instructions say they are data, but the real protection is structural, since the only tool with a side effect is not available to any agent that reads documents. And model output never reaches the channel unreviewed: the publish tool reads the report body from the store by investigation id, so what the human approved is byte for byte what gets posted.

An investigation moves through a small state machine. Only one state accepts a decision.

```mermaid
stateDiagram-v2
    [*] --> running
    running --> failed: planning failed, replan or call budget exhausted
    running --> needs_revision: DoD never passed within the revision budget
    running --> awaiting_approval: DoD passed, publish run paused
    awaiting_approval --> published: approved, posted once
    awaiting_approval --> rejected: rejected by reviewer
    needs_revision --> [*]
    failed --> [*]
    published --> [*]
    rejected --> [*]
```

The judge's verdict does not appear in the state machine on purpose. The deterministic Definition of Done is a hard gate: a report that fails it cannot be approved. The judge is advisory once the gate passes: it drives revisions inside the loop, and its final verdict is attached to the record so the approver sees "judge: 3/5, cause lacks an independent signal" next to the report. Making a calibrated-but-fallible judge a hard gate would turn its false negatives into missing reports during an incident, which is the wrong failure to optimize for.

## Implementation

### Layout and configuration

```
book/projects/p5-incident-agent/
  pyproject.toml  .env.example  Dockerfile  README.md
  data/alerts.json  data/telemetry.json
  incident_agent/
    config.py  tools.py  judge.py  agent.py  service.py  cli.py  api.py
    domain/   models.py  report.py  dod.py  deviation.py
    adapters/ corpus.py  telemetry.py  channel.py  store.py  metering.py  cassette.py  scripted.py
  tests/      test_tools_and_corpus.py  test_dod.py  test_trajectory.py  test_evaluator_optimizer.py
              test_approval.py  test_replay.py  test_api_cli.py
```

The project depends on `aie_core`, `agentkit`, `ragkit` (for loading and section-chunking the shared runbooks), `evalkit` (for the rubric judge), and `rank-bm25`, all as path dependencies in `pyproject.toml`. Model selection uses `aie_core`'s variables; with `LLM_PROVIDER=fake`, the default, a scripted stand-in model plays every role so the system runs offline.

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER`, `LLM_MODEL`, provider keys | `fake` | model for planner, step agents, writer, judge |
| `P5_STATE_DIR` | `.p5-state` | investigation records, event logs, channel log |
| `P5_MAX_PLAN_STEPS` | 8 | executed steps per investigation across replans |
| `P5_MAX_REPLANS` | 2 | replans allowed before the investigation fails |
| `P5_MAX_REVISIONS` | 3 | writer rounds in the evaluator-optimizer loop |
| `P5_MIN_IMPROVEMENT` | 0.05 | score gain below which a round counts as no progress |
| `P5_JUDGE_PASS_SCORE` | 4 | rubric score the judge requires, 1 to 5 |
| `P5_JUDGE_MODEL` | unset | a different model for the judge |
| `P5_STEP_MAX_STEPS`, `P5_STEP_MAX_TOOL_CALLS` | 3, 2 | budget of each step agent |
| `P5_MAX_LLM_CALLS` | 60 | hard ceiling on model calls per investigation, all roles |
| `P5_CHANNEL` | `#incidents` | where approved reports go |

The environment template is `.env.example`, and the README lists install, run, and Docker commands.

### Tools

Four read-only research tools are built per investigation, bound to the alert's time so the model cannot ask about a different window, plus one external tool for publishing. Searches go through the knowledge base, which loads the shared documents with `ragkit`, keeps runbooks (by tag) and incident reports (by id prefix), chunks them by Markdown section, ranks with BM25, filters by ACL before anything leaves the function, and returns the best chunk per document, since the document id is the citation unit everyone reasons about.

```python
# path: book/projects/p5-incident-agent/incident_agent/tools.py  (excerpt; full file on disk)
def untrusted(source: str, text: str) -> str:
    """Wrap retrieved text as data. Escaping keeps the text from closing the wrapper early."""
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f'<untrusted_data source="{source}">{safe}</untrusted_data>'


def research_tools(kb: KnowledgeBase, telemetry: Telemetry, alert: Alert, *, k: int = 3) -> list[FunctionTool]:
    as_of = alert.fired_at

    def search(kind: str) -> Callable[..., ToolOutput]:
        def run(ctx: ToolContext, query: str) -> ToolOutput:
            hits = kb.search(kind, query, ctx.principal, k)         # ACL from the trusted principal
            lines, items = [], []
            for h in hits:
                lines.append(f"[{h.doc_id}] {h.title} > {h.section}\n{untrusted(h.doc_id, h.text[:600])}")
                items.append(Evidence(id=h.doc_id, kind=kind, title=h.title, text=h.text[:600]))  # type: ignore[arg-type]
            return _evidence_output(lines, items)
        return run
    # ... query_service_metrics also reports each dependency as normal or ANOMALOUS;
    # ... get_recent_deploys lists deploys and migrations in a window before as_of


def publish_tool(channel: Channel, load: Callable[[str], Investigation | None]) -> FunctionTool:
    def post_report(ctx: ToolContext, investigation_id: str, channel_name: str) -> ToolOutput:
        inv = load(investigation_id)
        if inv is None or inv.report is None:
            return ToolOutput.failure(f"no report for investigation {investigation_id!r}", ErrorClass.IMPOSSIBLE)
        msg_id = channel.post(channel_name, f"Incident report {inv.alert.id}: {inv.alert.rule}", inv.report,
                              idempotency_key=ctx.idempotency_key)
        return ToolOutput(content=f"posted {msg_id} to {channel_name}", artifacts={"message_id": msg_id})
    # ... the FunctionTool declares the side-effect class and approval:
        post_report, side_effect=SideEffect.EXTERNAL, requires_approval=True, idempotent=True, pass_context=True)
```

If your organization uses Chapter 16's governed tool layer, wrap its executor with `agentkit.executor_tools(executor, ctx)` instead of `FunctionTool`; the executor's policy, idempotency store, and audit log stay in force, and the rest of the project is unchanged.

### The deterministic Definition of Done and the deviation rules

The report contract is small enough to state in a sentence: six named sections; every sentence or bullet in Summary, Impact, Timeline, and Likely cause cites at least one source; every citation is a source this investigation observed; the recommended runbook exists and was retrieved. The checker returns a list of named problems, each with a message written to be useful as revision feedback. These checks are the hard gate: they are cheap, exact, and cannot be talked out of a verdict, which is why they run before the rubric judge and why their output makes the best revision feedback.

```python
# path: book/projects/p5-incident-agent/incident_agent/domain/dod.py  (excerpt; full file on disk)
def check_report(report: str, evidence_ids: Iterable[str], runbook_catalog: Iterable[str]) -> list[Problem]:
    evidence, catalog = set(evidence_ids), set(runbook_catalog)
    sections = parse_sections(report)
    problems: list[Problem] = []

    # ... missing_section for each required section that is absent or empty

    for name in CLAIM_SECTIONS:
        for claim in claims(sections.get(normalize_heading(name), "")):
            if not cited(claim):
                problems.append(Problem(code="uncited_claim", message=f"{name}: claim has no [source-id]: {claim[:120]!r}"))

    for source in sorted(set(cited(report))):
        if source not in evidence:
            problems.append(Problem(code="unknown_citation",
                                    message=f"[{source}] is not a source this investigation observed"))

    body = sections.get(normalize_heading("Recommended runbook"), "")
    if body:
        named = cited(body)
        runbooks = [n for n in named if n in catalog]
        # ... runbook_not_in_catalog, runbook_missing, runbook_ambiguous, then:
        for n in runbooks:
            if n not in evidence:
                problems.append(Problem(code="runbook_not_retrieved",
                                        message=f"runbook [{n}] exists but was never retrieved in this investigation"))
    return problems
```

What counts as a claim is defined in `report.py`: each bullet is one claim, prose is split into sentences at a period followed by a capital letter (so version numbers survive), and table rows are skipped. The definition is crude on purpose. A citation per sentence is a check a reviewer can verify by eye and a writer can satisfy without guessing, and it is strict enough to catch the common failure, an unsupported inference written as fact.

The deviation rules are just as small. Each reads the finished step, the structured data its tools returned, and the steps done or queued, and returns a reason or nothing. A third rule on disk, `step_failed`, fires when a step agent did not complete.

```python
# path: book/projects/p5-incident-agent/incident_agent/domain/deviation.py  (excerpt; full file on disk)
def no_evidence(rec: StepRecord, data: list[dict[str, Any]], planned: list[PlanStep]) -> str | None:
    if rec.ok and rec.step.tool.startswith("search_") and not rec.evidence_ids:
        return f"step {rec.step.id} ({rec.step.tool} '{rec.step.target}') returned no evidence"
    return None


def uncovered_dependency(rec: StepRecord, data: list[dict[str, Any]], planned: list[PlanStep]) -> str | None:
    """A metrics step found an anomalous dependency that no planned step will look at."""
    covered = {s.target for s in planned if s.tool == "query_service_metrics"}
    for d in data:
        for dep in d.get("anomalous_dependencies", []):
            if dep not in covered:
                return (f"dependency {dep} of {rec.step.target} is anomalous and no step examines it; "
                        f"add query_service_metrics and get_recent_deploys for {dep}")
    return None
```

Note what `no_evidence` does not fire on: a deploy query that finds no deploys. "No change shipped to this service" is evidence, and replanning on it would be thrash.

### The orchestrator

`IncidentResearchAgent` composes the two patterns. The investigation loop is the planner-executor from earlier in the chapter, specialized with domain validation and the evidence ledger; the writing loop is the evaluator-optimizer, specialized so that evidence is gathered once and only writing repeats.

```python
# path: book/projects/p5-incident-agent/incident_agent/agent.py  (excerpt; full file on disk)
    def investigate(self, inv: Investigation) -> Investigation:
        with self.tracer.span("incident.investigate", investigation=inv.id, alert=inv.alert.id):
            tools = {t.name: t for t in research_tools(self.kb, self.telemetry, inv.alert)}
            # ... first plan; a planning failure ends the investigation as failed
            queue = list(plan.steps)
            while queue:
                if len(inv.steps) >= self.limits.max_plan_steps:
                    return self._fail(inv, f"step cap of {self.limits.max_plan_steps} reached")
                step = queue.pop(0)
                rec, data = self.execute(inv, step, tools[step.tool])
                inv.steps.append(rec)
                planned = [s.step for s in inv.steps] + queue
                reason = next((r for rule in self.rules if (r := rule(rec, data, planned))), None)
                if reason is None:
                    continue
                inv.deviations.append(reason)
                if len(inv.plans) - 1 >= self.limits.max_replans:
                    return self._fail(inv, f"replan budget exhausted: {reason}")
                # ... replan with the remaining queue and the reason; a replanning failure fails the investigation
                inv.plans.append(plan)
                queue = list(plan.steps)
            self.write_and_evaluate(inv)
        return inv

    def execute(self, inv: Investigation, step: PlanStep, tool: FunctionTool) -> tuple[StepRecord, list[dict[str, Any]]]:
        # ... one AgentRuntime per step, with exactly one tool, then fold the tool's sources into the ledger:
        data = [e.data for e in run.events_of(ToolResult) if e.ok and isinstance(e.data, dict)]
        ids: list[str] = []
        for d in data:
            for src in d.get("sources", []):
                ev = Evidence(**{**src, "step": step.id})
                inv.evidence.setdefault(ev.id, ev)
                ids.append(ev.id)
```

`execute` builds one `AgentRuntime` per step with exactly the tool the step names and a Definition of Done that requires the tool to have been called and the finding to be grounded in a citation or to say the tool found no evidence. Evidence comes from the tools' structured `sources`, never from parsing the model's text, which is why a citation the model invents cannot enter the ledger.

`write_and_evaluate` (on disk) is the evaluator-optimizer loop from earlier in the chapter with domain pieces plugged in: each round the writer drafts from the ledger, `check_report` runs, the judge runs only on a draft with no problems, the round is scored, the best report is kept, and the loop stops on acceptance, plateau, or the revision budget. A best report that still has Definition-of-Done problems ends as `needs_revision`; one without problems moves to `awaiting_approval`.

The scoring function (`round_score`) makes the controller's preferences explicit: any report that fails the Definition of Done scores below any report that passes it, fewer problems score higher, and among passing reports the judge's normalized score decides. The weights are illustrative; what matters is that "best so far" has a definition you can test (`test_round_score_orders_dod_failures_below_any_pass`).

The judge reuses `evalkit`'s `LLMJudge` (Chapter 24) with one rubric: is the cause supported by the cited evidence, is observation separated from inference, are the next steps specific and safe. It receives the evidence ledger, so it judges support, not style.

### Publishing behind an approval

The publish step has no decision to make, so its "model" is a small deterministic function that always proposes the same `post_report` call. `AgentRuntime` is used for what an irreversible action needs: a durable event log, a policy check, an approval pause that survives a restart, an idempotency key, and resume.

```python
# path: book/projects/p5-incident-agent/incident_agent/agent.py  (excerpt; full file on disk)
class Publisher:
    # ... __init__(channel, load, event_store, channel_name, tracer):
        self.runtime = AgentRuntime(
            PublishProposer(), [publish_tool(channel, load)],
            system_prompt=role("publisher", "Post the approved report."),
            budget=Budget(max_steps=3, max_tool_calls=1),
            dod=DefinitionOfDone(any_of(tool_was_called("post_report"), contains_all("rejected"))),
            store=event_store, tracer=tracer)

    def request(self, inv: Investigation) -> RunResult:
        goal = json.dumps({"investigation_id": inv.id, "channel": self.channel_name})
        return self.runtime.run(goal, run_id=f"{inv.id}.publish", metadata={"investigation": inv.id})

    def decide(self, run_id: str, *, approve: bool, reviewer: str, reason: str) -> RunResult:
        return self.runtime.resume(run_id, approve=approve, reason=f"{reviewer}: {reason}".strip())
```

`IncidentService.investigate` saves the record before requesting publication (the tool reads the report from the store) and checks that the publish run actually paused; if it ended any other way, the investigation is marked failed rather than trusted. `decide` accepts only `awaiting_approval`, requires the reviewer to be on-call staff in the alert's tenant, resumes the run, and records the message id from the run's artifacts. The CLI and the FastAPI app are thin shells over these two methods.

### Tests

The suite runs offline with the scripted model. Two kinds of test get a listing here because they are specific to agents: trajectory tests, which assert the path an investigation took, and replay tests, which assert that recorded runs reproduce.

```python
# path: book/projects/p5-incident-agent/tests/test_trajectory.py  (excerpt; full file on disk)
EXPECTED_MAIN = [
    ("query_service_metrics", "trackline"),
    ("query_service_metrics", "pg-logi-prod"),       # inserted by the replan
    ("get_recent_deploys", "pg-logi-prod"),          # inserted by the replan
    ("get_recent_deploys", "trackline"),
    ("search_incidents", "trackline latency degradation"),
    ("search_runbooks", "production incident response stabilise"),
]


def test_main_alert_replans_once_to_examine_the_anomalous_dependency(service):
    inv = service.investigate(MAIN, "oncall-logistics")
    assert inv.status is Status.AWAITING_APPROVAL
    assert inv.trajectory() == EXPECTED_MAIN
    assert len(inv.plans) == 2 and len(inv.deviations) == 1
    assert inv.deviations[0].startswith("dependency pg-logi-prod of trackline is anomalous")
    assert [s.ok for s in inv.steps] == [True] * 6
    assert "deploy:CHG-2026-0907" in inv.evidence and "[deploy:CHG-2026-0907]" in inv.report
```

A companion test on disk sets `max_replans=0` and asserts the investigation fails closed: status `failed`, no report, no publish run, one step executed.

```python
# path: book/projects/p5-incident-agent/tests/test_replay.py  (excerpt; full file on disk)
def test_every_step_replays_identically_without_executing_tools(service, telemetry):
    inv = service.investigate(MAIN, "oncall-logistics")
    before = telemetry.queries
    for step in inv.steps:
        report = replay(service.event_store.load(step.run_id))
        assert report.identical, (step.run_id, report.summary())
    assert telemetry.queries == before


def test_cassette_detects_a_prompt_change(make_service, tmp_path, monkeypatch):
    tape = tmp_path / "tape.jsonl"
    make_service(RecordingLLM(scripted_llm(), tape)).investigate(MAIN, "oncall-logistics")
    monkeypatch.setattr(agent_module, "WRITER_SYSTEM", agent_module.WRITER_SYSTEM + " Be brief.")
    replayer = ReplayLLM(tape)
    inv = make_service(replayer).investigate(MAIN, "oncall-logistics")
    assert inv.status is Status.FAILED and "CassetteMiss" in inv.detail
    assert len(replayer.misses) == 1                 # planning and execution replayed; the writer call is new
```

Step replay uses `agentkit.replay` on each step's event log. A whole investigation also makes model calls outside any agent loop (planner, writer, judge), so it needs a second mechanism: the cassette in `adapters/cassette.py` records every completion keyed by a hash of the request and serves them back. Recording a real run once and replaying it in CI turns "the report did not change" into a deterministic test, and a request the cassette has never seen is reported as a miss that pinpoints which prompt changed.

## Code walkthrough

Run it from the project directory:

```bash
cd book/projects/p5-incident-agent
python -m pytest -q                                    # all offline
p5 investigate ALR-2026-0914-01 --user oncall-logistics
```

```text
inv-alr-2026-0914-01-05113e  status=awaiting_approval
trajectory: query_service_metrics(trackline) -> query_service_metrics(pg-logi-prod) -> get_recent_deploys(pg-logi-prod) -> get_recent_deploys(trackline) -> search_incidents(trackline latency degradation) -> search_runbooks(production incident response stabilise)
replans: 1; deviations: ['dependency pg-logi-prod of trackline is anomalous and no step examines it; add query_service_metrics and get_recent_deploys for pg-logi-prod']
round 1: score=0.25 dod_problems=4
round 2: score=1.00 dod_problems=0 judge=5/5
usage: {"model_calls": 17, "by_role": {"planner": 2, "executor": 12, "writer": 2, "judge": 1}, "tokens": 12245}
```

Read the summary top to bottom the way you would read the trace. The trajectory shows the replan: two `pg-logi-prod` steps jumped the queue ahead of the planned deploy and search steps. The deviation line gives the exact reason, which is the first thing a reviewer asks for. The two rounds show the evaluator-optimizer doing its job: four Definition-of-Done problems in the first draft, none in the second, and a judge call only on the second. The usage line is the cost profile in miniature: twelve of seventeen model calls are step agents, because each step is a two-call run (call the tool, report).

The accepted report's Likely cause section reads, in part: "The migration CHG-2026-0907 preceded the degradation [deploy:CHG-2026-0907]. Database signals moved with it: pg-logi-prod.seq_scans_per_s baseline 12/s, at 06:10 UTC 880/s (x73.3) [metric:pg-logi-prod.seq_scans_per_s]. This matches a past incident in which a migration dropped a composite index and queries fell back to sequential scans [inc-2026-02-tracking-latency]. Confidence is moderate: the index loss is inferred, not yet confirmed [metric:pg-logi-prod.seq_scans_per_s]." Every sentence points at something a human can open. The recommendation is `it-incident-response-runbook`, which exists and was retrieved; the database failover runbook, which the search also surfaced, is not recommended, and the replication-lag metric that might have suggested it stayed below its absolute floor.

Approve it from a second shell and replay it:

```bash
p5 approve inv-alr-2026-0914-01-05113e --user ic-logistics --reason "matches dashboards"
p5 replay inv-alr-2026-0914-01-05113e
```

```text
inv-alr-2026-0914-01-05113e  status=published  message=MSG-00001
inv-alr-2026-0914-01-05113e.s1: identical: 2 decisions reproduced
inv-alr-2026-0914-01-05113e.s5: identical: 2 decisions reproduced
...
```

The approval ran in a new process with nothing in memory; the run was rebuilt from `.p5-state/events/<id>.publish.jsonl`. Approving again returns "refused: investigation ... is published", and the channel log still holds one message.

## Production considerations

**Latency.** Count the sequential model calls on the critical path, not the total. Project 5's offline run has seventeen calls, all sequential, which at an illustrative one to three seconds per call is well beyond an interactive budget. That is acceptable for an asynchronous investigation that pages a human, and unacceptable for a chat reply. Two changes cut it without changing behavior: run independent steps concurrently (metrics and deploys for the same service do not depend on each other, so adding `depends_on` to `PlanStep` lets the orchestrator fan them out), and let steps whose tool output is already the finding skip the second "report" call. In production, the API should return 202 with a status URL and run the investigation on a queue (Chapter 28 designs the job protocol, Chapter 29 the queue).

**Cost.** Meter every role, not only the agents. The `MeteredLLM` wrapper attributes calls to planner, executor, writer, and judge, and enforces a hard ceiling across all of them, so a pathological plan cannot spend more than the investigation is worth. A separately configured judge model gets its meter through `MeteredLLM.share`, so it counts against the same ceiling and appears in the same usage record; a second, independent meter would silently double the ceiling (`test_a_separate_judge_model_shares_the_investigation_call_ceiling`). Move the judge to a cheaper model only after calibrating it against human labels (Chapter 30 turns these counts into a cost model).

**Security.** The model never supplies identity: tools read groups and tenant from `ToolContext`, which the harness fills from the server-side directory. Retrieval applies ACLs before ranking results leave the knowledge base, so a document the caller cannot read never appears even as a title. The publish tool's isolation from document-reading agents, described under Architecture, is what keeps a prompt injection inside a runbook from reaching the channel (Chapter 26). The approver must be on-call staff in the same tenant; a four-eyes rule (approver differs from requester) is a one-line addition worth making for higher-impact actions.

**Operations.** Every agent run writes a JSONL event log under a derived run id (`<investigation>.<step>`), so one investigation's runs sort together. Alert on the distribution of statuses and step termination reasons, not only on errors: a rise in `needs_revision` or "replan budget exhausted" after a prompt change is a regression no exception will reveal. Replay a committed cassette of real investigations in CI on every prompt, model, or harness change. The single API worker is deliberate, since approvals resume runs from files; scaling out needs a shared event store and run-level locking (Chapter 38).

## Common mistakes

- **Choosing the architecture before the baseline.** Building a supervisor with five specialists without first measuring a single ReAct agent with the same tools. Without the baseline you cannot show the extra calls bought anything.
- **Writing a second agent loop.** A pattern implementation that calls the model and executes tools itself loses budgets, policy, events, and replay. Compose `AgentRuntime` runs instead.
- **Replanning on every step.** It doubles calls and makes plans chase the latest observation. Write deviation rules and replan only when one fires.
- **Reflection without new evidence.** A critic that rereads the answer in the same context mostly agrees with it. Give the critic observations, objective checks, or a different model.
- **Non-self-contained delegation.** "Check the thing above" means nothing to a worker that sees only its task string.
- **Unbounded fan-out.** Launching one branch per item without a concurrency limit, then discovering provider rate limits during an incident.
- **Letting the model retype the approved artifact.** If the publish call carries the report text in its arguments, what is posted can differ from what was approved. Pass a reference and read the artifact from the store.

## Failure modes

| Failure | How it shows in telemetry | How to test for it |
|---|---|---|
| Wandering ReAct loop | runs ending `NO_PROGRESS` or `REPEATED_ACTION`; tool calls per run creeping up | scripted model that repeats a call; assert the stop reason and the tool-call count |
| Misroute | specialist runs ending `VERIFICATION_FAILED` with marginal route confidence | labeled routing set; confusion matrix in CI |
| Stale plan | steps succeed but the final answer addresses a narrower question; no deviations logged | cases where step one reveals a new entity; assert a replan and the new step |
| Replan thrash | replans near the budget on most runs; investigations failing with "replan budget exhausted" | `max_replans=0` test that fails closed; replan-rate metric per release |
| Invented runbook or citation | DoD problems `runbook_not_in_catalog` or `unknown_citation` in round records | DoD unit tests with invented ids; trajectory test asserting the round sequence |
| Uncited inference | `uncited_claim` problems; judge flags "cause lacks an independent signal" | DoD test removing one citation; judge-calibration set with weak causes |
| Writer plateau | `needs_revision` with "plateau after round 2" | writer that ignores feedback; assert status and that approval is refused |
| Supervisor runaway | spawn-budget refusals in observations; agents spawned per run at the cap | supervisor that delegates every step; assert the ledger length |
| Silent partial merge | aggregated answers with fewer sources than branches; `missing` non-empty | `require="all"` and `"quorum"` tests with one failing branch |
| Duplicate side effect | two channel messages for one investigation | approve twice; replay the publish tool with the same key; assert one message |
| Cross-tenant leak | a retail document id in a logistics investigation's evidence | ACL tests on the knowledge base; tenant checks on investigate, read, decide |

Two of these are easy to misdiagnose. A stale plan produces no errors: every step completes and the report is internally consistent. The tell is an evidence ledger with no source about the entity the first step surfaced; the fix is a deviation rule, not a better prompt. A writer plateau looks like a model-quality problem, but it is usually uninformative feedback: if the round records show the same problem codes in every round, the feedback text for that code is what needs work.

## Tradeoffs

**Adaptivity versus predictability.** ReAct adapts at every step and is the hardest to predict; a sequential workflow is fully predictable and cannot adapt at all. Planner-executor sits between them, adaptive only at named deviations, which is why it suits regulated or reviewed work.

**Context sharing versus isolation.** One agent sees everything and loses nothing at hand-offs, but its context grows without bound. Supervisors and per-step agents keep contexts small and permissions tight, at the price of hand-off loss and more calls.

**Quality loops versus latency and cost.** Each evaluator round adds a full generation plus evaluation to the critical path. Deterministic gating keeps judge cost down; plateau rules keep the loop from polishing; but a loop that rarely improves the outcome should be removed, not tuned.

**Hard gates versus advisory signals.** Deterministic checks make good gates because their errors are rare and explainable. Model judges make good signals and poor gates, because their errors are frequent and hard to explain; a judge used as a hard gate turns its false negatives into missing reports during an incident. Gate on deterministic checks and show the judge's verdict to the human, as Project 5 does.

## Evaluation and testing

Evaluate an architecture at three levels, and keep them separate so a regression points at its layer.

**Outcome.** The task's own metric on a frozen set: for Project 5, whether the report names the correct cause and a valid runbook, judged against gold labels per alert, plus the DoD pass rate and the judge's score distribution. Compare every architecture against the simplest baseline on the same cases, with paired statistics (Chapter 24), and report cost per successful case and p95 latency next to success rate.

**Trajectory.** Assertions about the path, read from events: the expected tool and target sequence for each scripted scenario, the number of replans and their reasons, which tools each step agent could see, that the publish tool never executed before approval. Trajectory tests are cheap, deterministic, and catch regressions that outcome metrics miss, such as an agent that reaches the right answer through a step it should not have been allowed to take.

**Components.** Each decision-maker evaluated as what it is: the router as a classifier, the planner on plan validity and gold-step coverage, the deviation rules on precision (did a replan change the outcome), the judge on agreement with humans, the deterministic DoD with unit tests for every problem code.

Replay ties the levels together. Harness replay of recorded step runs checks that a new verifier, policy, or truncation limit still accepts what it should (`test_a_stricter_verifier_would_have_rejected_a_recorded_answer` shows the reverse: a stricter verifier rejecting an answer that was accepted when recorded). Counterfactual replay runs a new planner or prompt against recorded observations and reports where decisions diverge and which calls have no recording. The cassette replays whole investigations, model calls outside agent loops included, so a CI job can assert that a refactor changed nothing and can name the first prompt that did.

For Project 5 specifically, the minimum release gate is: all offline tests green, cassette replays of the recorded investigations identical, DoD pass rate and judge agreement on the frozen set at or above the previous release, and no increase in average model calls per investigation beyond an agreed tolerance.

## Before you ship

- [ ] A single-agent ReAct baseline with the same tools has been run on the same frozen case set, and the chosen architecture beats it on success rate by a margin you fixed in advance, with cost per successful case and p95 latency reported next to it.
- [ ] Best-case, typical, and worst-case model calls per request are written down, and the worst case is bounded by explicit settings (step, tool-call, replan, revision, spawn, and per-request call ceilings such as `P5_MAX_LLM_CALLS`).
- [ ] No pattern code calls the model and executes tools in its own loop; every agent is an `AgentRuntime` built through one factory.
- [ ] Plans are validated (known tools, unique ids, dependencies earlier in the list) before any step runs, with one repair round and a fail-closed path tested by `max_replans=0`.
- [ ] Each deviation rule has a unit test that shows it fires on its case and stays silent on a near miss (for example, a deploy query that finds nothing).
- [ ] Every agent, including the judge on a separate model, is metered against one shared call ceiling, and usage is recorded per role.
- [ ] The deterministic Definition of Done has a unit test for every problem code, and it is the only hard gate before approval; the judge's verdict is shown to the approver, not used to block.
- [ ] The judge has been calibrated against human labels (agreement and Cohen's kappa) before its pass score is trusted.
- [ ] The irreversible tool is unavailable to every agent that reads untrusted content, takes an artifact reference rather than its text, requires approval, carries an idempotency key, and has a test that approving twice posts once.
- [ ] Parallel branches run under a concurrency limit, results merge in branch order, and the fan-in policy (`all`, `quorum`, `any`) is chosen per use and tested with one failing branch.
- [ ] Trajectory tests for each scripted scenario and a cassette replay of recorded investigations run in CI on every prompt, model, or harness change.
- [ ] Dashboards alert on the distribution of statuses and termination reasons (`needs_revision`, "replan budget exhausted", `NO_PROGRESS`), not only on errors.

## Exercises

**Start here:** K1, K3, E2, P2, D1 (about 5 hours). The rest go deeper.

### Knowledge questions

**K1.** For each of the nine patterns, name who owns the "what happens next" decision and give the number of sequential model calls on the critical path in the best case.

**K2.** Why does ReAct's input-token cost grow roughly quadratically with the number of steps while planner-executor's grows roughly linearly? State the assumption under which the advantage disappears.

**K3.** Explain the difference between reflection and evaluator-optimizer in terms of who owns the loop, what the evaluator can see, and which stop rules each can implement.

**K4.** In Project 5, why is the deterministic Definition of Done a hard gate while the rubric judge is advisory? Describe the failure you would introduce by reversing them.

**K5.** What does each fan-in policy (`all`, `quorum`, `any`) assume about the relationship between branches? Give a Northwind example for each.

**K6.** Why does the publish tool take an investigation id rather than the report text as an argument?

### Engineering questions

**E1.** The Northwind HR team wants an assistant that answers policy questions, files leave requests, and escalates harassment reports to a human. Choose an architecture, draw its structure, list each agent's tools and side-effect classes, and state where approval is required.

**E2.** Design two more deviation rules for Project 5 and specify their inputs, the reason string, and a test that shows each fires exactly when it should. One must use the deploy data.

**E3.** Project 5's seventeen sequential calls are too slow for a team that wants a draft within thirty seconds. Beyond the two changes named under Production considerations, propose changes that reduce critical-path calls without removing per-step isolation or the Definition of Done, and estimate the new call count on the critical path.

**E4.** A product manager proposes a three-level hierarchy (platform lead, tier leads, specialists) for incident research. Write the evaluation plan that would justify or reject it against Project 5, including the baseline, metrics, and decision rule.

### Practical exercises

**P1.** (about 3 hours) Add plan-driven parallelism to `IncidentResearchAgent`: add a `depends_on` field to its `PlanStep`, then execute steps whose dependencies are satisfied concurrently, with a configurable worker limit, keeping the trajectory recorded in plan order. Add a test asserting the same evidence ledger as the sequential version and fewer sequential rounds.

**P2.** (about 90 min) Implement memoization for read-only tools in `patterns/evaluator_optimizer.py` so that round two of `agent_generator` reuses round one's observations for identical calls. Show with `Meter` and tool counters that tool calls drop while answers are unchanged.

**P3.** (about 60 min) Add a four-eyes rule to `IncidentService.decide` (the approver must differ from the requester) and an audit field recording both. Cover it in the service, CLI, and API tests.

**P4.** (about 2 hours, needs a provider key) Record a cassette from a real provider for both sample alerts, commit it, and add a CI test that replays it. Then change the planner prompt and capture the miss report.

### Debugging exercises

**D1.** After a release, 30 percent of investigations end `failed` with "replan budget exhausted: step s3 (search_incidents 'pg-logi-prod seq scans after CHG-2026-0907') returned no evidence." Investigations of the same alerts passed the week before. What changed, and what do you check in the plans and step records to confirm it?

**D2.** A supervisor-based variant of the incident agent produces reports whose citations point only at `[incident_analyst result]` and never at metric or deploy ids, and the Definition of Done rejects every draft. Diagnose the cause from the ledger and event logs and propose the fix.

**D3.** In a variant, the publish run is driven by a real model with the default `LoopConfig` and no tool-call limit, and an `approver` callback approves any `post_report` call for an investigation a human has already approved, "so reviewers are not asked twice." The channel shows two identical reports for one investigation, posted eleven minutes apart. The record shows `published` with the second message id, and the publish run's event log shows a `Resumed` event by "recovery" followed by a second `post_report` request with a new request id and a second `ToolResult`. Reconstruct what happened and name the defects.

## Key takeaways

- An architecture is a choice of who owns each decision, model or code, and how many model calls sit on the critical path. Count best, typical, and worst cases before building.
- Every pattern is a composition of bounded `AgentRuntime` runs, single calls, and code. A pattern with its own model-and-tools loop has thrown away budgets, policy, events, and replay.
- ReAct is the default agent. Add a router for categories that need different tools, planner-executor for long or reviewed tasks, and supervisors only when isolated contexts beat a single-agent baseline.
- Plans are hypotheses held as data. Validate them before execution and replan only when a deterministic rule names a concrete deviation, within a replan budget.
- Quality loops help only when feedback carries new information. Put deterministic checks first, an independent judge second, keep the best candidate, and stop on plateau.
- Fan-out buys latency, not cost; the design decision is the fan-in policy and how missing or conflicting branches are reported.
- Gate irreversible actions on deterministic checks and human approval, pass artifacts by reference, and make the side effect idempotent so resume and retries cannot duplicate it.
- Evaluate at three levels: outcome against a baseline, trajectory from events, and each decision-maker as what it is. Replay recorded runs and cassettes to test harness, prompt, and model changes without touching live systems.

## Further reading

- *ReAct: Synergizing Reasoning and Acting in Language Models* (Yao et al., 2023). The original interleaving of reasoning and tool actions that `AgentRuntime` implements with budgets and a Definition of Done.
- *Reflexion: Language Agents with Verbal Reinforcement Learning* (Shinn et al., 2023). Reflection with an external signal; read it for when self-critique works and why the signal matters.
- *Self-Refine: Iterative Refinement with Self-Feedback* (Madaan et al., 2023). The generate, critique, revise loop; compare its gains with this chapter's caution about critique without new evidence.
- *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* (Zheng et al., 2023). The biases to calibrate out of the judge in an evaluator-optimizer loop.
- *Improving Factuality and Reasoning in Language Models through Multiagent Debate* (Du et al., 2023). A multi-agent design whose gains come with many more calls; useful context before Chapter 22.
