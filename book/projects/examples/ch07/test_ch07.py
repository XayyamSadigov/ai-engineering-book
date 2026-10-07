# path: book/projects/examples/ch07/test_ch07.py
"""Offline tests for Chapter 7: catalog, selection harness, router, cascade evaluation.

FakeLLM instances stand in for a "small" and a "large" model; no network, no API keys.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from aie_core import (CompletionRequest, ContentPart, InvalidRequestError, Message, PricingTable,
                      ProviderUnavailableError, Role, ToolSpec, FakeLLM)
from aie_core.embeddings import FakeEmbeddings
from aie_core.llm.tokens import count_tokens
from aie_core.observability import InMemoryTracer

from cascade_eval import (CaseOutcome, UtilityModel, baseline, best_by_router_accuracy,
                          best_by_utility, calibration_table, collect_outcomes,
                          expected_calibration_error, simulate, sweep)
from catalog import (ModelCatalog, ModelProfile, Requirements, Tier, capability_gaps, is_compatible,
                     northwind_catalog)
from fakes import gold_from_cases, make_large_model, make_small_model
from router import (EmbeddingRouteClassifier, NoCompatibleModelError, Route, Router, Rule,
                    northwind_router)
from selection import (CandidateSummary, choose, paired_disagreements, pareto_front, percentile,
                       run_selection, wilson_interval)
from tasks import label_confidence, load_ticket_cases, parse_label, score_label, ticket_request


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def cases():
    return load_ticket_cases()


@pytest.fixture(scope="module")
def gold(cases):
    return gold_from_cases(cases)


@pytest.fixture()
def catalog() -> ModelCatalog:
    return northwind_catalog()


@pytest.fixture()
def pricing(catalog) -> PricingTable:
    return PricingTable(catalog.pricing())


def big_text(tokens: int) -> str:
    # Size the text with the same counter the router uses (tiktoken or the 4-chars heuristic),
    # so the test does not depend on which one is available offline.
    unit = "incident timeline "
    per_unit = count_tokens(unit * 100) / 100
    return unit * math.ceil(tokens / per_unit)


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


# ------------------------------------------------------------------------------ catalog
def test_requirements_are_derived_from_the_request():
    req = CompletionRequest(
        messages=[Message(role=Role.USER, content=[ContentPart(type="text", text="what is on this receipt?"),
                                                    ContentPart(type="image_url", image_url="data:image/png;base64,AAAA")])],
        tools=[ToolSpec(name="search_tickets", description="search")],
        response_schema={"type": "object"}, max_tokens=500, metadata={"data_zone": "eu"},
    )
    need = Requirements.from_request(req)
    assert need.needs_tools and need.needs_vision and need.needs_json_schema
    assert need.data_zone == "eu"
    assert need.min_output_tokens == 500
    assert need.min_context_tokens > 500


def test_tool_choice_none_does_not_require_tools():
    req = CompletionRequest(messages=[Message.user("hi")], tools=[ToolSpec(name="t", description="d")],
                            tool_choice="none")
    assert not Requirements.from_request(req).needs_tools


def test_hard_and_soft_gaps(catalog):
    longctx = catalog.get("nw-longctx")
    need = Requirements(min_context_tokens=10_000, needs_tools=True, needs_json_schema=True)
    gaps = {g.capability: g.hard for g in capability_gaps(longctx, need)}
    assert gaps == {"tools": True, "json_schema": False}
    assert not is_compatible(longctx, need)
    assert is_compatible(longctx, need.model_copy(update={"needs_tools": False}))


def test_data_zone_gap(catalog):
    assert not is_compatible(catalog.get("nw-general"), Requirements(data_zone="onprem"))
    assert is_compatible(catalog.get("nw-small"), Requirements(data_zone="onprem"))
    assert not is_compatible(catalog.get("nw-longctx"), Requirements(data_zone="onprem"))  # "any" is no guarantee


def test_compatible_is_cheapest_first(catalog):
    names = [p.alias for p in catalog.compatible(Requirements(min_context_tokens=50_000))]
    assert names[0] == "nw-general"
    assert "nw-small" not in names


def test_catalog_validation_and_json_roundtrip(tmp_path: Path, catalog):
    with pytest.raises(ValueError):
        ModelProfile(alias="x", model_id="x-1", provider="p", context_tokens=1000, max_output_tokens=2000)
    with pytest.raises(ValueError):
        catalog.add(catalog.get("nw-small"))
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"models": [p.model_dump(mode="json") for p in catalog.profiles.values()]}))
    again = ModelCatalog.from_json(path)
    assert again.get("nw-reasoning").reasoning_efforts == ("low", "medium", "high")
    assert again.get("nw-small").cost_tier is Tier.LOW
    assert again.pricing() == catalog.pricing()


def test_settings_from_environment(monkeypatch, gold):
    from config import RouterSettings
    path = Path(__file__).with_name("catalog.json")
    monkeypatch.setenv("MODEL_CATALOG_PATH", str(path))
    monkeypatch.setenv("ROUTER_MIN_CONFIDENCE", "0.95")
    monkeypatch.setenv("ROUTER_LONG_CONTEXT_TOKENS", "20000")
    settings = RouterSettings(_env_file=None)
    assert settings.load_catalog().pricing() == northwind_catalog().pricing()
    clients = {"nw-small": make_small_model(), "nw-general": make_large_model(gold),
               "nw-reasoning": make_large_model(gold), "nw-longctx": make_large_model(gold)}
    router = settings.build_router(clients)
    assert router.routes["small_first"].min_confidence == 0.95
    d = router.route(CompletionRequest(messages=[Message.user(big_text(25_000))]))
    assert d.route == "long_context"


# ---------------------------------------------------------------------------- selection
def test_wilson_interval_is_wide_at_small_n():
    lo, hi = wilson_interval(8, 10)
    assert 0.48 < lo < 0.50 and 0.94 < hi < 0.95
    lo2, hi2 = wilson_interval(80, 100)
    assert hi2 - lo2 < hi - lo
    assert wilson_interval(10, 10)[1] == pytest.approx(1.0)


def test_percentile_nearest_rank():
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10
    assert percentile([5.0], 50) == 5.0
    assert percentile([], 95) == 0.0


def _summary(name, q, cost, p95):
    return CandidateSummary(candidate=name, n=100, quality=q, quality_low=q - 0.05, quality_high=q + 0.05,
                            p50_latency_ms=p95 / 2, p95_latency_ms=p95, cost_per_request_usd=cost,
                            cost_per_correct_usd=cost / q, error_rate=0.0)


def test_pareto_front_drops_dominated_candidates():
    a = _summary("a", 0.90, 0.010, 2000)
    b = _summary("b", 0.80, 0.001, 300)
    c = _summary("c", 0.85, 0.020, 2500)   # worse than a on every axis
    d = _summary("d", 0.90, 0.010, 2000)   # identical to a: neither dominates
    assert {s.candidate for s in pareto_front([a, b, c, d])} == {"a", "b", "d"}


def test_selection_harness_on_fakes(cases, gold, pricing):
    report = run_selection({"nw-small": make_small_model(), "nw-general": make_large_model(gold)},
                           cases, score_label, pricing)
    small, general = report.by_name("nw-small"), report.by_name("nw-general")
    assert small.quality == pytest.approx(0.80)
    assert general.quality == pytest.approx(56 / 60)
    assert small.cost_per_request_usd < general.cost_per_request_usd / 5
    assert small.p95_latency_ms < general.p95_latency_ms
    assert small.pareto and general.pareto          # a real trade-off: neither dominates
    assert report.summaries[0].candidate == "nw-general"
    a_only, b_only = paired_disagreements(report, "nw-small", "nw-general")
    assert b_only > a_only > 0                      # the large model is not uniformly better
    assert "| nw-small |" in report.to_markdown()


def test_errors_are_scored_not_raised(cases, pricing):
    down = FakeLLM(responses=[ProviderUnavailableError("down")] * 3, model="general-2026-02")
    report = run_selection({"flaky": down}, cases[:3], score_label, pricing)
    s = report.by_name("flaky")
    assert s.error_rate == 1.0 and s.quality == 0.0
    assert all(r.error == "ProviderUnavailableError" for r in report.results)


def test_choose_uses_the_lower_confidence_bound(cases, gold, pricing):
    report = run_selection({"nw-small": make_small_model(), "nw-general": make_large_model(gold)},
                           cases, score_label, pricing)
    assert choose(report, min_quality=0.75, max_p95_ms=5000, use_lower_bound=False).candidate == "nw-small"
    assert choose(report, min_quality=0.75, max_p95_ms=5000).candidate == "nw-general"   # small's low bound < 0.75
    assert choose(report, min_quality=0.75, max_p95_ms=1000) is None                     # nothing fits both


# ------------------------------------------------------------------------------- router
def test_narrow_task_served_by_small_when_confident(catalog, gold, cases):
    router = make_router(catalog, gold)
    confident = next(c for c in cases if make_small_model().complete(c.request).text.find('"confidence": 0.9') > 0)
    r = router.complete(confident.request)
    assert r.decision.route == "small_first" and r.decision.stage == "rule"
    assert r.served_by == "nw-small" and not r.escalated
    assert [a.outcome for a in r.attempts] == ["ok"]


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


def test_invalid_output_escalates_even_with_high_confidence_text(catalog, gold):
    small = FakeLLM(responses=["I think this is about the VPN."], model="small-instruct-2026-03")
    general = FakeLLM(responses=[{"category": "vpn_network", "confidence": 0.9}], model="general-2026-02")
    r = make_router(catalog, gold, small=small, general=general).complete(ticket_request("VPN", "vpn down"))
    assert [a.outcome for a in r.attempts] == ["invalid", "ok"]


def test_retryable_error_falls_back_without_double_escalation(catalog, gold):
    small = FakeLLM(responses=[ProviderUnavailableError("gpu node lost")], model="small-instruct-2026-03")
    general = FakeLLM(responses=[{"category": "vpn_network", "confidence": 0.5}], model="general-2026-02")
    r = make_router(catalog, gold, small=small, general=general).complete(ticket_request("VPN", "vpn down"))
    # general answered as the fallback; it is also the escalation target, so no second call
    assert r.served_by == "nw-general" and not r.escalated
    assert [a.outcome for a in r.attempts] == ["error", "ok"]
    assert general.remaining() == 0


def test_non_retryable_error_is_raised(catalog, gold):
    small = FakeLLM(responses=[InvalidRequestError("bad schema")], model="small-instruct-2026-03")
    with pytest.raises(InvalidRequestError):
        make_router(catalog, gold, small=small).complete(ticket_request("VPN", "vpn down"))


def test_failed_escalation_returns_cheap_answer_marked_degraded(catalog, gold):
    small = FakeLLM(responses=[{"category": "hardware", "confidence": 0.3}], model="small-instruct-2026-03")
    general = FakeLLM(responses=[ProviderUnavailableError("overloaded")], model="general-2026-02")
    r = make_router(catalog, gold, small=small, general=general).complete(ticket_request("x", "y"))
    assert r.degraded and not r.escalated and r.served_by == "nw-small"


def test_disabled_escalation_marks_low_confidence_answer_degraded(catalog, gold):
    # The cascade target has no vision, so for an image request the cascade cannot run. A
    # low-confidence answer must then be flagged, not returned as if it met the route's bar.
    general = FakeLLM(responses=[{"category": "hardware", "confidence": 0.3}], model="general-2026-02")
    clients = {"nw-general": general, "nw-reasoning": make_large_model(gold, model="reasoner-2026-01")}
    router = Router(catalog, clients, [Route(name="g", model="nw-general", escalate_to="nw-reasoning",
                                             min_confidence=0.8)], default_route="g",
                    confidence=label_confidence)
    req = CompletionRequest(messages=[Message(role=Role.USER, content=[
        ContentPart(type="text", text="what is broken in this photo?"),
        ContentPart(type="image_url", image_url="https://example.invalid/laptop.png")])])
    r = router.complete(req)
    assert r.decision.escalation_disabled and r.decision.escalate_to is None
    assert r.degraded and not r.escalated and r.attempts[-1].outcome == "low_confidence"


class _SilentlyRewired:
    """Stands in for a gateway whose own fallback list answered from another model."""

    def __init__(self, inner: FakeLLM, served_model: str) -> None:
        self.inner, self.served_model, self.provider = inner, served_model, inner.provider

    def complete(self, req: CompletionRequest):
        return self.inner.complete(req.model_copy(update={"model": self.served_model}))


def test_model_mismatch_below_the_router_is_detected(catalog, gold):
    tracer = InMemoryTracer()
    reasoning = _SilentlyRewired(FakeLLM(responses=["no errors were found"]), "longctx-2025-12")
    router = make_router(catalog, gold, reasoning=reasoning, tracer=tracer)
    r = router.complete(CompletionRequest(messages=[Message.user("investigate the outage")],
                                          metadata={"risk": "high"}))
    assert r.served_by == "nw-reasoning" and r.model_mismatch
    (span,) = tracer.find("router.complete")
    assert span.attributes["router.model_mismatch"] is True
    assert span.attributes["router.model_id"] == "longctx-2025-12"


def test_oversized_prompt_skips_the_small_model(catalog, gold):
    req = ticket_request("Long ticket", big_text(40_000))
    d = make_router(catalog, gold).route(req)
    assert d.route == "small_first"
    assert d.candidates == ["nw-general"]
    assert d.substitutions and "context" in d.substitutions[0]


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


def test_oversized_onprem_request_is_never_substituted_to_a_cloud_model(catalog, gold):
    big = CompletionRequest(messages=[Message.user("payroll line. " * 20_000)],
                            metadata={"data_zone": "onprem"})
    with pytest.raises(NoCompatibleModelError):   # nw-small's window is too small; nothing else is on-prem
        make_router(catalog, gold).route(big)


def test_high_risk_gets_reasoning_effort_and_toolless_models_do_not(catalog, gold):
    reasoning = FakeLLM(responses=["careful answer"], model="reasoner-2026-01")
    router = make_router(catalog, gold, reasoning=reasoning)
    r = router.complete(CompletionRequest(messages=[Message.user("approve refund over limit?")],
                                          metadata={"risk": "high"}))
    assert r.decision.route == "high_assurance"
    sent = reasoning.last_request
    assert sent.metadata["reasoning_effort"] == "high" and sent.model == "reasoner-2026-01"
    general = FakeLLM(responses=["plain"], model="general-2026-02")
    router2 = make_router(catalog, gold, general=general)
    router2.complete(CompletionRequest(messages=[Message.user("hello")], metadata={"reasoning_effort": "high"}))
    assert "reasoning_effort" not in general.last_request.metadata   # nw-general has no knob


def test_soft_gap_produces_warning(catalog, gold):
    req = CompletionRequest(messages=[Message.user(big_text(150_000))], response_schema={"type": "object"})
    d = make_router(catalog, gold).route(req)
    assert d.candidates == ["nw-longctx", "nw-reasoning"]
    assert d.warnings == ["nw-longctx: no native schema mode; falls back to prompt+parse with repair"]


def test_embedding_classifier_routes_and_abstains(catalog, gold):
    emb = FakeEmbeddings(vocabulary=["why", "plan", "prove", "compare", "tradeoff", "status", "hours",
                                     "where", "link", "office", "migration"])
    clf = EmbeddingRouteClassifier(emb, {
        "reasoning": ["why did the migration fail and plan a fix", "compare tradeoff and prove it"],
        "general": ["office hours", "where is the status link"],
    })
    router = make_router(catalog, gold, classifier=clf)
    hard = CompletionRequest(messages=[Message.user("why would this migration plan fail")])
    easy = CompletionRequest(messages=[Message.user("where is the office")])
    vague = CompletionRequest(messages=[Message.user("hello there")])   # no vocabulary overlap
    d_hard, d_easy, d_vague = router.route(hard), router.route(easy), router.route(vague)
    assert (d_hard.route, d_hard.stage) == ("reasoning", "classifier")
    assert d_hard.reasoning_effort == "medium"
    assert (d_easy.route, d_easy.stage) == ("general", "classifier")
    assert (d_vague.route, d_vague.stage) == ("general", "default")


def test_misconfiguration_fails_at_construction(catalog, gold):
    clients = {"nw-small": make_small_model()}
    with pytest.raises(ValueError, match="no client"):
        Router(catalog, clients, [Route(name="a", model="nw-small", escalate_to="nw-general")], default_route="a")
    with pytest.raises(ValueError, match="unknown route"):
        Router(catalog, clients, [Route(name="a", model="nw-small")], default_route="a",
               rules=[Rule("r", "missing", lambda req, need: True)])
    with pytest.raises(KeyError):
        Router(catalog, clients, [Route(name="a", model="nw-tiny")], default_route="a")


def test_router_emits_a_span(catalog, gold, cases):
    tracer = InMemoryTracer()
    make_router(catalog, gold, tracer=tracer).complete(cases[0].request)
    (span,) = tracer.find("router.complete")
    assert span.attributes["router.route"] == "small_first"
    assert span.attributes["router.model"] in {"nw-small", "nw-general"}
    assert "router.cost_usd" in span.attributes
    assert span.attributes["router.model_mismatch"] is False
    assert span.attributes["router.model_id"] in {"small-instruct-2026-03", "general-2026-02"}


# ------------------------------------------------------------------------ cascade eval
def _o(i, small_ok, conf, large_ok=True):
    return CaseOutcome(case_id=str(i), small_correct=small_ok, small_confidence=conf, small_cost_usd=0.001,
                       small_latency_ms=200, large_correct=large_ok, large_cost_usd=0.01, large_latency_ms=2000)


def test_simulate_extremes_and_counts():
    outs = [_o(1, True, 0.9), _o(2, False, 0.95), _o(3, True, 0.4), _o(4, False, 0.3)]
    u = UtilityModel(cost_silent_error=0.1)
    accept_all = simulate(outs, 0.0, u)
    assert accept_all.accuracy == 0.5 and accept_all.escalation_rate == 0.0
    assert accept_all.false_accept_rate == 0.5
    escalate_all = simulate(outs, 1.01, u)
    assert escalate_all.accuracy == 1.0 and escalate_all.escalation_rate == 1.0
    assert escalate_all.cost_per_request_usd == pytest.approx(0.011)
    assert escalate_all.p95_latency_ms == 2200
    mid = simulate(outs, 0.5, u)
    assert mid.false_accept_rate == 0.25        # the confident wrong answer slips through
    assert mid.false_escalation_rate == 0.25    # the unconfident right answer is escalated
    assert baseline(outs, "large", u).cost_per_request_usd == pytest.approx(0.01)


def test_router_accuracy_and_utility_disagree():
    # One confident-ish miss (0.70) hides above three correct but hesitant answers (0.65).
    outs = [_o(0, False, 0.70)] + [_o(i, True, 0.65) for i in range(1, 4)] + [_o(i, True, 0.9) for i in range(4, 10)]
    thresholds = [0.0, 0.75]
    expensive = sweep(outs, UtilityModel(cost_silent_error=1.0), thresholds)
    # Router accuracy prefers accepting everything: one wrong decision versus three.
    assert best_by_router_accuracy(expensive).threshold == 0.0
    # Utility prefers paying for four large calls to avoid one very expensive miss.
    assert best_by_utility(expensive).threshold == 0.75
    cheap = sweep(outs, UtilityModel(cost_silent_error=0.0001), thresholds)
    assert best_by_utility(cheap).threshold == 0.0


def test_detected_errors_are_redone_on_the_large_model():
    outs = [_o(1, False, 0.9)]
    u = UtilityModel(cost_silent_error=1.0, detect_rate=1.0, rework_cost=0.02)
    m = simulate(outs, 0.5, u)
    assert m.accuracy == 1.0                                       # fixed downstream...
    assert m.cost_per_request_usd == pytest.approx(0.011)          # ...after paying for both calls
    assert m.mean_latency_ms == 2200
    assert m.utility_per_request == pytest.approx(-0.011 - 0.02)   # plus the rework


def test_calibration_metrics():
    perfect = [_o(i, i < 9, 0.9) for i in range(10)]
    assert expected_calibration_error(perfect) == pytest.approx(0.0)
    over = [_o(i, i < 5, 0.95) for i in range(10)]
    assert expected_calibration_error(over) == pytest.approx(0.45)
    (b,) = calibration_table(over)
    assert b.n == 10 and b.accuracy == 0.5


def test_optimal_threshold_rises_with_error_cost(cases, gold, pricing):
    outs = collect_outcomes(cases, make_small_model(), make_large_model(gold), score_label,
                            label_confidence, pricing)
    assert len(outs) == 60
    assert sum(o.small_correct for o in outs) == 48
    best = []
    for cost in (0.0001, 0.001, 0.01, 0.1):
        u = UtilityModel(cost_silent_error=cost)
        best.append(best_by_utility(sweep(outs, u)).threshold)
    assert best == sorted(best) and best[0] < best[-1]
    # When errors are expensive the small model's confident mistakes make the cascade lose
    # to calling the large model directly.
    u = UtilityModel(cost_silent_error=0.05)
    assert baseline(outs, "large", u).utility_per_request > best_by_utility(sweep(outs, u)).utility_per_request


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
