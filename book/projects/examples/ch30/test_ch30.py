# path: book/projects/examples/ch30/test_ch30.py
"""Offline tests for Chapter 30: latency budgets, cache layers, cost model, spend guard, fan-out."""
from __future__ import annotations

import asyncio
import time

import pytest
from aie_core.embeddings import FakeEmbeddings
from aie_core.llm.errors import InvalidRequestError
from aie_core.llm.gateway import InMemoryResponseCache, ModelGateway, PricingTable
from aie_core.llm.providers import FakeLLM
from aie_core.llm.types import Completion, CompletionRequest, Message, Usage
from aie_core.observability import InMemoryTracer, JsonlTracer

from attribution import AttributingTracer, bind
from budgets import (
    Alert,
    BudgetedClient,
    BudgetExceededError,
    SpendGuard,
    SpendPolicy,
    TaskTokenBudget,
)
from caching import (
    EmbeddingCache,
    RetrievalCache,
    Scope,
    ScopedResponseCache,
    SemanticCache,
    lint_cache_classes,
    lint_cache_key,
    normalize_text,
)
from cost import UNATTRIBUTED, CostModel, CostScenario, chargeback, load_jsonl
from latency import BudgetError, Deadline, LatencyBudget, LatencyTracker, StageBudget, percentile
from parallel import MicroBatcher, Prefetcher, RequiredStepFailed, Step, fan_out

# Illustrative prices only (USD per million tokens), matching Chapter 35's worked examples.
PRICES = {
    "capable-model": {"input_per_1m": 2.0, "cached_input_per_1m": 0.2, "output_per_1m": 8.0},
    "small-model": {"input_per_1m": 0.2, "cached_input_per_1m": 0.02, "output_per_1m": 0.8},
}


@pytest.fixture
def pricing() -> PricingTable:
    return PricingTable(PRICES)


def req(text: str, **kw) -> CompletionRequest:
    return CompletionRequest(messages=[Message.user(text)], **kw)


class FakeClock:
    def __init__(self, t: float = 1_760_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


# =============================================================================== latency
def northwind_budget() -> LatencyBudget:
    return LatencyBudget(
        total_ms=8000,
        ttft_ms=2000,
        reserve_ms=400,
        stages=(
            StageBudget("auth", 100, before_first_token=True),
            StageBudget("retrieve", 600, before_first_token=True),
            StageBudget("rerank", 300, before_first_token=True),
            StageBudget("generate", 6200),
            StageBudget("persist", 300),
            StageBudget("validate", 100),
        ),
    )


def test_budget_must_fit_total_and_ttft() -> None:
    with pytest.raises(BudgetError, match="total"):
        LatencyBudget(total_ms=1000, stages=(StageBudget("a", 700), StageBudget("b", 400)))
    with pytest.raises(BudgetError, match="first-token"):
        LatencyBudget(total_ms=8000, ttft_ms=500, stages=(StageBudget("retrieve", 600, before_first_token=True),))
    assert northwind_budget()["generate"].budget_ms == 6200


def test_allocate_keeps_reserve_and_proportions() -> None:
    b = LatencyBudget.allocate(8000, {"retrieve": 1, "generate": 3}, reserve_fraction=0.1)
    assert b["retrieve"].budget_ms == pytest.approx(1800)
    assert b["generate"].budget_ms == pytest.approx(5400)
    assert b.reserve_ms == pytest.approx(800)


def test_from_measurements_refuses_infeasible_and_scales_feasible() -> None:
    with pytest.raises(BudgetError, match="retrieve"):
        LatencyBudget.from_measurements(8000, {"retrieve": 6000, "generate": 3000})
    b = LatencyBudget.from_measurements(8000, {"retrieve": 600, "generate": 3000}, reserve_fraction=0.1)
    assert b["retrieve"].budget_ms > 600 and b["generate"].budget_ms > 3000
    assert sum(s.budget_ms for s in b.stages) + b.reserve_ms == pytest.approx(8000)


def test_deadline_caps_stage_timeout_by_remaining_time() -> None:
    clock = FakeClock(0.0)
    d = Deadline(northwind_budget(), clock=clock)
    assert d.timeout_for("retrieve") == pytest.approx(0.6)
    clock.advance(7.5)  # something upstream ate the budget
    assert d.timeout_for("generate") == pytest.approx(0.5)
    clock.advance(1.0)
    assert d.expired() and d.timeout_for("persist") == 0.0


def test_percentile_nearest_rank() -> None:
    values = [float(i) for i in range(1, 101)]
    assert percentile(values, 50) == 50.0
    assert percentile(values, 95) == 95.0
    assert percentile([], 95) != percentile([], 95)  # nan


def span(rid: str, stage: str, start: float, end: float, **attrs) -> dict:
    return {"name": stage, "start": start, "end": end, "attributes": {"request_id": rid, "stage": stage, **attrs}}


def test_tracker_reports_violations_and_wall_clock_end_to_end() -> None:
    t = LatencyTracker(northwind_budget())
    # r1: retrieval and an optional tool call overlap; fine.
    t.record_all([
        span("r1", "auth", 0.0, 0.05),
        span("r1", "retrieve", 0.05, 0.55),
        span("r1", "tool", 0.05, 0.50),
        span("r1", "generate", 0.6, 4.0, first_token_s=1.2),
    ])
    # r2: retrieval blows its budget and pushes TTFT past 2 s.
    t.record_all([
        span("r2", "auth", 0.0, 0.05),
        span("r2", "retrieve", 0.05, 2.05),
        span("r2", "generate", 2.1, 7.0, first_token_s=2.6),
    ])
    rep = t.report()
    assert rep.requests == 2
    stages = {(v.request_id, v.stage) for v in rep.violations}
    assert ("r2", "retrieve") in stages and ("r2", "ttft") in stages
    assert ("r1", "retrieve") not in stages
    assert rep.worst_stage() == "retrieve"
    assert rep.violation_rate("ttft") == 0.5
    # Overlap: r1's stage durations sum past its wall clock.
    assert rep.end_to_end_p95_ms == pytest.approx(7000)


def test_tracker_joins_gateway_spans_through_attribution(pricing: PricingTable) -> None:
    sink = InMemoryTracer()
    tracer = AttributingTracer(sink)
    gw = ModelGateway(FakeLLM(handler=lambda r: "answer"), pricing=pricing, tracer=tracer)
    with bind(request_id="req-7", tenant="retail"):
        with tracer.span("retrieve", stage="retrieve"):
            pass
        gw.complete(req("vacation policy?", model="capable-model"))
    tracker = LatencyTracker(northwind_budget(), stage_for_span={"llm.complete": "generate"})
    tracker.record_all(sink.spans)
    rep = tracker.report()
    assert rep.requests == 1 and set(rep.stage_p95_ms) == {"retrieve", "generate"}
    assert all(s.attributes["tenant"] == "retail" for s in sink.spans)


# =============================================================================== caching
VOCAB = ["refund", "policy", "deadline", "vpn", "reset", "password", "laptop", "return", "days", "how", "long"]


class CountingEmbeddings(FakeEmbeddings):
    def __init__(self) -> None:
        super().__init__(vocabulary=VOCAB)
        self.batches: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return super().embed(texts)


def test_embedding_cache_normalizes_dedupes_and_versions() -> None:
    inner = CountingEmbeddings()
    cache = EmbeddingCache(inner, model_version="2026-01")
    cache.embed(["Refund  policy", "refund policy", "VPN reset"])
    assert inner.batches == [["refund policy", "vpn reset"]]  # one call, duplicates collapsed
    cache.embed_query("REFUND POLICY ")
    assert len(inner.batches) == 1 and cache.store.hits == 1
    other = EmbeddingCache(inner, cache.store, model_version="2026-06")
    other.embed_query("refund policy")
    assert len(inner.batches) == 2  # new model version, same store, no reuse
    prefixed = EmbeddingCache(inner, cache.store, model_version="2026-01", instruction_prefix="query: ")
    prefixed.embed_query("refund policy")
    assert len(inner.batches) == 3 and inner.batches[-1] == ["query: refund policy"]


def test_retrieval_cache_is_scoped_by_acl_and_index_version() -> None:
    rc = RetrievalCache()
    calls: list[str] = []

    def compute(tag: str):
        return lambda: calls.append(tag) or [f"doc-{tag}"]

    hr = Scope.of("retail", ["all", "hr"])
    staff = Scope.of("retail", ["all"])
    ids, hit = rc.get_or_compute("salary bands", hr, compute("hr"), index_version="v12")
    assert not hit
    _, hit = rc.get_or_compute("Salary  bands", Scope.of("retail", ["hr", "all"]), compute("hr2"), index_version="v12")
    assert hit  # same groups in a different order, same normalized query
    _, hit = rc.get_or_compute("salary bands", staff, compute("staff"), index_version="v12")
    assert not hit  # different permissions never share
    _, hit = rc.get_or_compute("salary bands", hr, compute("v13"), index_version="v13")
    assert not hit  # reindex retires entries
    assert calls == ["hr", "staff", "v13"]
    assert rc.purge_tenant("retail") == 3


def test_scoped_response_cache_separates_tenants_and_skips_unsafe(pricing: PricingTable) -> None:
    llm = FakeLLM(handler=lambda r: f"answer {len(llm.requests)}")
    cache = ScopedResponseCache(llm)
    r = req("classify: printer on fire", model="small-model")
    retail, logistics = Scope.of("retail", ["all"]), Scope.of("logistics", ["all"])
    _, hit1 = cache.complete(r, retail, prompt_version="triage@3")
    _, hit2 = cache.complete(r, retail, prompt_version="triage@3")
    _, hit3 = cache.complete(r, logistics, prompt_version="triage@3")
    _, hit4 = cache.complete(r, retail, prompt_version="triage@4")
    assert (hit1, hit2, hit3, hit4) == (False, True, False, False)
    _, hot = cache.complete(req("write a poem", model="small-model", temperature=0.9), retail, prompt_version="p")
    _, hot2 = cache.complete(req("write a poem", model="small-model", temperature=0.9), retail, prompt_version="p")
    assert not hot and not hot2
    truncated = Completion(message=Message.assistant("partial"), usage=Usage(), finish_reason="length", model="m", provider="fake", latency_ms=1)
    tllm = FakeLLM(responses=[truncated, truncated])
    tcache = ScopedResponseCache(tllm)
    tcache.complete(req("long", model="m"), retail, prompt_version="p")
    _, hit = tcache.complete(req("long", model="m"), retail, prompt_version="p")
    assert not hit  # truncated output is never cached


def semantic(clock: FakeClock | None = None, **kw) -> SemanticCache:
    return SemanticCache(FakeEmbeddings(vocabulary=VOCAB), threshold=0.8, ttl_s=600, clock=clock or FakeClock(), **kw)


V1 = {"prompt": "faq@2", "index": "v12"}


def test_semantic_cache_hits_similar_question_within_scope() -> None:
    sc = semantic()
    retail = Scope.of("retail", ["all"])
    sc.store("how long is the refund deadline", "30 days.", retail, V1, source_ids=["policy-refunds"])
    hit = sc.lookup("refund deadline how long", retail, V1)
    assert hit is not None and hit.answer == "30 days." and hit.similarity >= 0.8
    assert sc.lookup("vpn password reset", retail, V1) is None


def test_semantic_cache_eligibility_beats_similarity() -> None:
    clock = FakeClock()
    sc = semantic(clock)
    retail = Scope.of("retail", ["all"])
    sc.store("refund policy deadline", "30 days.", retail, V1)
    assert sc.lookup("refund policy deadline", Scope.of("logistics", ["all"]), V1) is None
    assert sc.lookup("refund policy deadline", Scope.of("retail", ["all", "hr"]), V1) is None
    assert sc.lookup("refund policy deadline", retail, {**V1, "index": "v13"}) is None
    assert sc.stats.ineligible == 3
    clock.advance(601)
    assert sc.lookup("refund policy deadline", retail, V1) is None  # expired


def test_semantic_cache_invalidation_and_refusals() -> None:
    sc = semantic()
    retail = Scope.of("retail", ["all"])
    sc.store("refund policy deadline", "30 days.", retail, V1, source_ids=["policy-refunds"])
    sc.store("vpn reset", "Use the portal.", retail, V1, source_ids=["runbook-vpn"])
    assert sc.store("my laptop return", "Your return ships Friday.", retail, V1, personalized=True) is None
    assert sc.store("laptop return days", "I think 14?", retail, V1, grounded=False) is None
    assert sc.stats.refused_writes == 2
    assert sc.invalidate_sources(["policy-refunds"]) == 1
    assert sc.invalidate_version("index", "v13") == 1
    assert len(sc) == 0


def test_semantic_cache_logs_near_misses_for_threshold_review() -> None:
    sc = SemanticCache(FakeEmbeddings(vocabulary=VOCAB), threshold=0.95, near_miss_margin=0.3, clock=FakeClock())
    retail = Scope.of("retail", ["all"])
    sc.store("refund policy deadline days", "30 days.", retail, V1)
    assert sc.lookup("refund policy", retail, V1) is None
    assert sc.stats.near_misses == 1 and sc.stats.near_miss_log[0][1] == "refund policy deadline days"


def test_linter_passes_book_caches_and_flags_naive_keys() -> None:
    clean = lint_cache_classes({"embedding": EmbeddingCache, "retrieval": RetrievalCache, "response": ScopedResponseCache, "semantic": SemanticCache})
    assert clean == []
    findings = lint_cache_key("retrieval", {"query", "index_version", "request_id"})
    errors = {f.component for f in findings if f.severity == "error"}
    warnings = {f.component for f in findings if f.severity == "warning"}
    assert errors == {"tenant", "acl_scope"} and warnings == {"request_id"}
    assert {f.component for f in lint_cache_key("embedding", {"text", "embedding_model"})} == {"space"}
    with pytest.raises(ValueError):
        lint_cache_key("vibes", set())


def test_normalize_text() -> None:
    assert normalize_text("  Ｒｅｆｕｎｄ\tPolicy ") == "refund policy"


# =============================================================================== cost
def rag_answer(**kw) -> CostScenario:
    base = dict(
        name="rag-capable",
        model="capable-model",
        input_tokens=4000,
        cached_input_tokens=1500,
        output_tokens=400,
        embedding_tokens=20,
        embedding_price_per_1m=0.02,
        rerank_calls=1,
        rerank_price_per_call=0.001,
        retry_rate=0.05,
        success_rate=0.9,
    )
    base.update(kw)
    return CostScenario(**base)


def test_breakdown_arithmetic(pricing: PricingTable) -> None:
    b = CostModel(pricing).breakdown(rag_answer())
    llm_call = (2500 * 2.0 + 1500 * 0.2 + 400 * 8.0) / 1e6  # 0.0085
    assert b.components["llm"] == pytest.approx(llm_call * 1.05)
    assert b.components["rerank"] == pytest.approx(0.001)
    assert b.per_successful_task == pytest.approx(b.per_attempt / 0.9)
    assert b.dominant() == "llm"


def test_cheap_model_with_many_steps_can_cost_more(pricing: PricingTable) -> None:
    cm = CostModel(pricing)
    one_strong = CostScenario(name="strong-once", model="capable-model", input_tokens=6000, output_tokens=600, success_rate=0.9)
    cheap_agent = CostScenario(name="cheap-agent", model="small-model", input_tokens=9000, output_tokens=300, model_calls=20, retry_rate=0.1, success_rate=0.75)
    ranking = cm.compare(one_strong, cheap_agent)
    assert ranking[0].scenario == "strong-once"
    assert ranking[1].delta_vs_baseline_pct > 0


def test_mix_weights_spend_and_successes(pricing: PricingTable) -> None:
    cm = CostModel(pricing)
    strong = rag_answer(name="strong", success_rate=0.9)
    small = rag_answer(name="small", model="small-model", success_rate=0.6)
    mixed = cm.mix("router", [(0.7, small), (0.3, strong)])
    spend = 0.7 * cm.breakdown(small).per_attempt + 0.3 * cm.breakdown(strong).per_attempt
    assert mixed.per_attempt == pytest.approx(spend)
    assert mixed.per_successful_task == pytest.approx(spend / (0.7 * 0.6 + 0.3 * 0.9))


def test_what_if_and_monthly(pricing: PricingTable) -> None:
    cm = CostModel(pricing)
    s = rag_answer()
    more_context = cm.what_if(s, input_tokens=s.input_tokens + 1000)
    assert more_context.components["llm"] - cm.breakdown(s).components["llm"] == pytest.approx(1000 * 2.0 / 1e6 * 1.05)
    m = cm.monthly(s, successful_tasks_per_day=9000)
    assert m.attempts == pytest.approx(9000 * 30 / 0.9)
    assert m.total_usd == pytest.approx(m.attempts * cm.breakdown(s).per_attempt)
    with pytest.raises(ValueError):
        CostScenario(name="bad", model="m", input_tokens=10, cached_input_tokens=20, output_tokens=1)


def test_chargeback_from_gateway_trace_jsonl(tmp_path, pricing: PricingTable) -> None:
    path = tmp_path / "traces.jsonl"
    tracer = AttributingTracer(JsonlTracer(path))
    gw = ModelGateway(FakeLLM(handler=lambda r: "an answer of several words"), pricing=pricing, tracer=tracer, cache=InMemoryResponseCache())
    for i, tenant in enumerate(["retail", "retail", "logistics"]):
        with bind(tenant=tenant, request_id=f"r{i}", feature="chat"):
            with tracer.span("task", success=(i != 1)):
                gw.complete(req(f"question {i}", model="capable-model"))
                with tracer.span("rerank", cost_usd=0.001):
                    pass
    with bind(tenant="retail", request_id="r9", feature="chat"):
        gw.complete(req("question 0", model="capable-model"))  # exact repeat: cache hit
    gw.complete(req("background job", model="small-model"))  # no tenant bound
    report = chargeback(load_jsonl(path), pricing)
    retail, logistics = report.tenants["retail"], report.tenants["logistics"]
    assert retail.model_calls == 2 and retail.cache_hits == 1 and retail.avoided_usd > 0
    # The gateway writes cost_usd=0 and avoided_cost_usd on hits, so no pricing table is needed.
    unpriced = chargeback(load_jsonl(path)).tenants["retail"]
    assert unpriced.avoided_usd == pytest.approx(retail.avoided_usd) and unpriced.spend_usd == pytest.approx(retail.spend_usd)
    assert retail.tasks == 2 and retail.successes == 1
    assert retail.other_usd == pytest.approx(0.002)
    assert logistics.cost_per_successful_task == pytest.approx(logistics.total_usd)
    assert UNATTRIBUTED in report.tenants and 0 < report.unattributed_share < 0.5
    report.allocate_shared(10.0)
    assert retail.allocated_shared_usd + logistics.allocated_shared_usd == pytest.approx(10.0)


# =============================================================================== budgets
def test_guard_alerts_once_per_threshold_and_blocks() -> None:
    alerts: list[Alert] = []
    guard = SpendGuard({"retail": SpendPolicy(daily_limit_usd=1.0)}, on_alert=alerts.append, clock=FakeClock())
    for _ in range(3):
        d = guard.reserve("retail", 0.3)
        guard.commit(d.reservation, 0.3)
    assert [a.threshold for a in alerts] == [0.5, 0.8]
    blocked = guard.reserve("retail", 0.3)
    assert blocked.action == "block" and alerts[-1].kind == "blocked"
    guard.reserve("retail", 0.3)
    assert sum(a.kind == "blocked" for a in alerts) == 1  # paged once, not per request


def test_reservations_prevent_concurrent_overshoot() -> None:
    guard = SpendGuard({"retail": SpendPolicy(daily_limit_usd=1.0)}, clock=FakeClock())
    decisions = [guard.reserve("retail", 0.3) for _ in range(5)]  # five in flight, none committed
    assert [d.action for d in decisions] == ["allow"] * 3 + ["block"] * 2


def test_observe_mode_never_blocks_and_day_rolls_over() -> None:
    clock = FakeClock()
    alerts: list[Alert] = []
    guard = SpendGuard({}, default_policy=SpendPolicy(1.0, mode="observe"), on_alert=alerts.append, clock=clock)
    for _ in range(4):
        d = guard.reserve("logistics", 0.5)
        assert d.action == "allow"
        guard.commit(d.reservation, 0.5)
    assert guard.spent("logistics") == pytest.approx(2.0) and alerts
    clock.advance(86_400)
    assert guard.spent("logistics") == 0.0


def test_budgeted_client_degrades_then_blocks(pricing: PricingTable) -> None:
    llm = FakeLLM(handler=lambda r: "word " * 500)  # about 500 output tokens per call
    guard = SpendGuard({"retail": SpendPolicy(daily_limit_usd=0.02, mode="degrade", soft_limit_fraction=0.5)}, clock=FakeClock())
    client = BudgetedClient(llm, guard, pricing, default_model="capable-model", degrade_model="small-model", degrade_max_tokens=128)
    r = req("hello " * 200, model="capable-model", max_tokens=1000, metadata={"tenant": "retail"})
    client.complete(r)
    assert llm.last_request.model == "capable-model"  # projected 0.0084 of a 0.01 soft limit
    client.complete(r)  # committed 0.0044 + estimate 0.0084 crosses the soft limit
    assert llm.last_request.model == "small-model" and llm.last_request.max_tokens == 128
    with pytest.raises(InvalidRequestError):
        client.complete(req("no tenant"))
    enforce = SpendGuard({"retail": SpendPolicy(daily_limit_usd=0.001)}, clock=FakeClock())
    with pytest.raises(BudgetExceededError):
        BudgetedClient(llm, enforce, pricing, default_model="capable-model").complete(r)


def test_budgeted_client_releases_on_error_and_commits_streams(pricing: PricingTable) -> None:
    guard = SpendGuard({"retail": SpendPolicy(daily_limit_usd=1.0)}, clock=FakeClock())
    failing = BudgetedClient(FakeLLM(responses=[InvalidRequestError("bad")]), guard, pricing, default_model="capable-model")
    with pytest.raises(InvalidRequestError):
        failing.complete(req("x", metadata={"tenant": "retail"}))
    assert guard.spent("retail") == 0.0
    streaming = BudgetedClient(FakeLLM(handler=lambda r: "streamed text"), guard, pricing, default_model="capable-model")
    events = list(streaming.stream(req("y", model="capable-model", metadata={"tenant": "retail"})))
    assert events[-1].type == "done" and guard.spent("retail") > 0


def test_task_token_budget_stops_runaway_loop() -> None:
    budget = TaskTokenBudget(max_input_tokens=10_000, max_output_tokens=2_000)
    steps = 0
    with pytest.raises(BudgetExceededError, match="after 3 steps"):
        while True:
            budget.require(3_000, 500)
            budget.charge(Usage(input_tokens=3_000, output_tokens=400))
            steps += 1
    assert steps == 3


# =============================================================================== parallel
def sleeper(seconds: float, value, log: list | None = None):
    async def fn():
        try:
            await asyncio.sleep(seconds)
            return value
        except asyncio.CancelledError:
            if log is not None:
                log.append(value)
            raise

    return fn


def test_fan_out_latency_is_max_not_sum() -> None:
    steps = {name: Step(sleeper(0.05, name)) for name in ("lexical", "vector", "status")}
    res = asyncio.run(fan_out(steps, deadline_s=1.0))
    assert set(res.results) == {"lexical", "vector", "status"}
    assert res.elapsed_ms < 120  # about 50 ms, not 150


def test_fan_out_degrades_on_optional_timeout_and_cancels() -> None:
    cancelled: list = []
    steps = {
        "vector": Step(sleeper(0.01, ["d1", "d2"])),
        "rerank": Step(sleeper(5.0, "slow", cancelled), required=False),
        "tickets": Step(sleeper(5.0, "tool", cancelled), timeout_s=0.02, required=False),
    }
    res = asyncio.run(fan_out(steps, deadline_s=0.1))
    assert res.results == {"vector": ["d1", "d2"]}
    assert set(res.timed_out) == {"rerank", "tickets"} and res.degraded
    assert set(cancelled) == {"slow", "tool"}


def test_fan_out_raises_when_required_step_fails() -> None:
    async def boom():
        raise RuntimeError("index down")

    with pytest.raises(RequiredStepFailed, match="vector"):
        asyncio.run(fan_out({"vector": Step(boom), "lexical": Step(sleeper(0.01, []))}, deadline_s=0.5))


def test_prefetcher_uses_matching_and_cancels_unused() -> None:
    async def scenario():
        p = Prefetcher(read_only_tools={"get_service_status", "search_tickets"})
        p.start("get_service_status", {"service": "vpn"}, sleeper(0.02, "degraded"))
        p.start("search_tickets", {"q": "vpn"}, sleeper(1.0, ["T-1"]))
        with pytest.raises(ValueError):
            p.start("send_reply", {"id": 1}, sleeper(0, None))
        status = await p.take("get_service_status", {"service": "vpn"}, sleeper(0, "fresh"))
        wasted = await p.cancel_unused()
        return status, wasted, p.stats

    status, wasted, stats = asyncio.run(scenario())
    assert status == "degraded" and wasted == 1 and stats.used == 1 and stats.wasted == 1


def test_micro_batcher_coalesces_and_maps_results() -> None:
    async def scenario():
        calls: list[list[int]] = []

        async def square_all(items: list[int]) -> list[int]:
            calls.append(items)
            return [i * i for i in items]

        mb: MicroBatcher[int, int] = MicroBatcher(square_all, max_batch=4, max_wait_ms=5)
        results = await asyncio.gather(*(mb.submit(i) for i in range(10)))
        return results, mb.batches

    results, batches = asyncio.run(scenario())
    assert results == [i * i for i in range(10)]
    assert batches == [4, 4, 2]


def test_micro_batcher_propagates_batch_failure() -> None:
    async def scenario():
        async def broken(items):
            raise RuntimeError("provider 503")

        mb = MicroBatcher(broken, max_batch=8, max_wait_ms=1)
        return await asyncio.gather(mb.submit(1), mb.submit(2), return_exceptions=True)

    out = asyncio.run(scenario())
    assert all(isinstance(e, RuntimeError) for e in out)


# =============================================================================== worked example
def test_demo_ladder_is_monotone_and_trim_is_the_biggest_step() -> None:
    from demo import PRICES as DEMO_PRICES, ladder

    rows = ladder(CostModel(PricingTable(DEMO_PRICES)))
    costs = [r.per_successful_task for r in rows]
    assert costs == sorted(costs, reverse=True)
    steps = [a - b for a, b in zip(costs, costs[1:])]
    assert steps[0] == max(steps[:3])  # trimming context beats caching for this workload
