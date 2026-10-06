# path: book/projects/examples/ch34/test_ch34.py
from __future__ import annotations

import asyncio
import math

import pytest
from capacity import (
    ReplicaProfile,
    Workload,
    arrival_rate,
    headroom,
    in_flight,
    mean_time_in_system,
    mm1_response_time,
    queueing_curve,
    replicas_needed,
    utilization,
)
from fake_server import FakeServer
from kv_cache import (
    GiB,
    ModelShape,
    Precision,
    format_bytes,
    gqa_savings_factor,
    kv_bytes_per_sequence,
    kv_bytes_per_token,
    max_concurrent_sequences,
    tokens_that_fit,
    weight_bytes,
)
from loadtest import (
    LoadTestConfig,
    find_operating_point,
    format_report,
    parse_sse_line,
    percentile,
    sweep,
)
from local_target import HOSTED, LOCAL_CPU, LOCAL_VLLM, RequestEnvelope, ServingTarget, check_fit, choose_target

SHAPE = ModelShape(layers=32, kv_heads=8, head_dim=128, params_billion=8)


# ----------------------------------------------------------------------------- kv_cache


def test_kv_bytes_per_token_matches_formula():
    # 2 * 32 layers * 8 kv heads * 128 dim * 2 bytes = 131072 bytes = 128 KiB per token
    assert kv_bytes_per_token(SHAPE) == 131_072


def test_worked_numbers_8k_and_32k():
    assert kv_bytes_per_sequence(SHAPE, 8_192) == 1 * GiB
    assert kv_bytes_per_sequence(SHAPE, 32_768) == 4 * GiB
    # doubling context doubles cache
    assert kv_bytes_per_sequence(SHAPE, 16_384) == 2 * kv_bytes_per_sequence(SHAPE, 8_192)


def test_fp8_cache_halves_memory():
    assert kv_bytes_per_sequence(SHAPE, 8_192, kv_bytes=1.0) == GiB // 2


def test_weight_bytes_for_precisions():
    assert weight_bytes(SHAPE, 2.0) == 16_000_000_000
    assert weight_bytes(SHAPE, 1.0) == 8_000_000_000
    assert weight_bytes(SHAPE, 0.5) == 4_000_000_000


def test_concurrency_estimate_drops_fourfold_from_8k_to_32k():
    e8 = max_concurrent_sequences(SHAPE, 8_192, gpu_memory_bytes=80 * GiB)
    e32 = max_concurrent_sequences(SHAPE, 32_768, gpu_memory_bytes=80 * GiB)
    assert e8.usable_kv_bytes == e32.usable_kv_bytes
    assert e8.max_sequences > 0
    assert e8.max_sequences // 4 >= e32.max_sequences >= e8.max_sequences // 4 - 1
    assert "concurrent sequences" in e8.summary()


def test_quantization_unlocks_concurrency():
    bf16 = max_concurrent_sequences(SHAPE, 32_768, gpu_memory_bytes=80 * GiB)
    low = max_concurrent_sequences(
        SHAPE, 32_768, gpu_memory_bytes=80 * GiB, precision=Precision(weight_bytes=1.0, kv_bytes=1.0)
    )
    assert low.max_sequences > 2 * bf16.max_sequences


def test_tensor_parallel_pools_memory():
    one = max_concurrent_sequences(SHAPE, 8_192, gpu_memory_bytes=24 * GiB)
    two = max_concurrent_sequences(SHAPE, 8_192, gpu_memory_bytes=24 * GiB, tensor_parallel=2)
    assert one.max_sequences < two.max_sequences


def test_model_that_does_not_fit_yields_zero():
    tiny = max_concurrent_sequences(SHAPE, 8_192, gpu_memory_bytes=8 * GiB)
    assert tiny.max_sequences == 0
    assert tiny.usable_kv_bytes == 0


def test_tokens_that_fit_is_inverse():
    assert tokens_that_fit(SHAPE, 4 * GiB) == 32_768
    assert tokens_that_fit(SHAPE, 0) == 0


def test_gqa_savings_and_validation():
    assert gqa_savings_factor(32, 8) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        gqa_savings_factor(8, 32)
    with pytest.raises(ValueError):
        kv_bytes_per_sequence(SHAPE, -1)
    with pytest.raises(ValueError):
        max_concurrent_sequences(SHAPE, 1024, 80 * GiB, headroom_fraction=1.0)


def test_format_bytes():
    assert format_bytes(GiB) == "1.00 GiB"
    assert format_bytes(131_072) == "128.0 KiB"
    assert format_bytes(10) == "10 B"


# ----------------------------------------------------------------------------- capacity


def test_littles_law_round_trips():
    assert in_flight(20, 4) == 80
    assert arrival_rate(80, 4) == 20
    assert mean_time_in_system(80, 20) == 4
    with pytest.raises(ValueError):
        in_flight(-1, 4)
    with pytest.raises(ValueError):
        arrival_rate(80, 0)


def test_queueing_knee_is_steep_near_saturation():
    assert utilization(6, 10) == pytest.approx(0.6)
    curve = queueing_curve(2.0, [0.5, 0.9, 0.99])
    times = [w for _, w in curve]
    assert times == pytest.approx([4.0, 20.0, 200.0])
    with pytest.raises(ValueError):
        mm1_response_time(2.0, 1.0)


def test_headroom():
    assert headroom(10, 6) == pytest.approx(0.4)
    assert headroom(10, 12) < 0


def _northwind() -> tuple[Workload, ReplicaProfile]:
    wl = Workload(
        peak_requests_per_s=6, mean_input_tokens=3000, mean_output_tokens=300,
        p95_input_tokens=6000, p95_output_tokens=600, mean_e2e_s=4.0,
    )
    prof = ReplicaProfile(
        decode_tokens_per_s=1500, prefill_tokens_per_s=20000,
        kv_budget_bytes=55 * GiB, kv_bytes_per_token=131_072,
    )
    return wl, prof


def test_replica_estimate_worked_example():
    wl, prof = _northwind()
    est = replicas_needed(wl, prof, target_utilization=0.6, failover_replicas=1)
    assert est.in_flight_requests == pytest.approx(24.0)
    assert est.decode_bound_replicas == pytest.approx(1800 / 900)  # 2.0
    assert est.prefill_bound_replicas == pytest.approx(18000 / 12000)  # 1.5
    assert est.binding_constraint == "decode"
    assert est.recommended_replicas == 3  # ceil(2.0) + 1 failover
    assert "binding: decode" in est.summary()


def test_kv_becomes_binding_with_long_contexts_and_slow_requests():
    wl, prof = _northwind()
    long_wl = wl.model_copy(update={"p95_input_tokens": 30_000, "p95_output_tokens": 2_000, "mean_e2e_s": 12.0})
    est = replicas_needed(long_wl, prof)
    assert est.binding_constraint == "kv"
    assert est.recommended_replicas > 3


def test_replica_estimate_validation():
    wl, prof = _northwind()
    with pytest.raises(ValueError):
        replicas_needed(wl, prof, target_utilization=0)
    with pytest.raises(ValueError):
        replicas_needed(wl, prof, failover_replicas=-1)


# ----------------------------------------------------------------------------- loadtest


def test_percentile_nearest_rank():
    vals = [float(i) for i in range(1, 101)]
    assert percentile(vals, 50) == 50
    assert percentile(vals, 95) == 95
    assert percentile(vals, 99) == 99
    assert percentile([3.0], 99) == 3.0
    assert math.isnan(percentile([], 50))
    with pytest.raises(ValueError):
        percentile(vals, 101)


def test_parse_sse_line():
    assert parse_sse_line("") is None
    assert parse_sse_line(": keep-alive") is None
    assert parse_sse_line("data: [DONE]") is None
    assert parse_sse_line('data: {"a": 1}') == {"a": 1}


def _cfg(**overrides) -> LoadTestConfig:
    base = dict(
        base_url="http://fake/v1", model="fake-model", prompts=["hello", "a longer prompt " * 10],
        max_tokens_choices=[8, 16], concurrency_levels=[1, 2, 8], requests_per_level=8,
        warmup_requests=1, ttft_slo_s=0.05, e2e_slo_s=1.0,
    )
    base.update(overrides)
    return LoadTestConfig(**base)


def test_sweep_measures_ttft_tpot_e2e_against_fake_server():
    server = FakeServer(capacity=4, prefill_s=0.005, tpot_s=0.002, queue_slot_s=0.03)
    cfg = _cfg()
    summaries = asyncio.run(sweep(cfg, transport=server.transport()))

    assert [s.concurrency for s in summaries] == [1, 2, 8]
    assert all(s.errors == 0 and s.requests == 8 for s in summaries)
    # warm-up plus three levels of 8 requests
    assert len(server.seen) == 1 + 3 * 8
    assert all(body["stream"] is True for body in server.seen)
    assert server.peak_active >= 5  # the fake really ran requests concurrently

    low = summaries[0]
    assert 0 < low.ttft_p50 < low.e2e_p50  # streaming makes TTFT observable
    assert low.tpot_p50 == pytest.approx(0.002, abs=0.004)
    # output_tokens came from usage, so tokens/s is based on 8 or 16 tokens per request
    assert low.output_tokens_per_s > 0

    # Above capacity the fake queues, so p95 TTFT must rise sharply at concurrency 8.
    high = summaries[-1]
    assert high.ttft_p95 > low.ttft_p95 * 3
    # Throughput still improves with concurrency (more requests in flight).
    assert high.requests_per_s > low.requests_per_s


def test_operating_point_and_report():
    server = FakeServer(capacity=4, prefill_s=0.005, tpot_s=0.002, queue_slot_s=0.03)
    cfg = _cfg(concurrency_levels=[1, 4, 16])
    summaries = asyncio.run(sweep(cfg, transport=server.transport()))
    op = find_operating_point(summaries, cfg)
    assert op is not None
    assert op.concurrency == 4  # within capacity: no queueing; 16 breaks the TTFT SLO
    assert summaries[-1].slo_pass_fraction < 1.0
    assert summaries[-1].goodput_requests_per_s < summaries[-1].requests_per_s
    report = format_report(summaries)
    assert "goodput" in report and report.count("\n") == 2 + len(summaries) - 1


def test_errors_are_counted_not_raised():
    server = FakeServer(capacity=2, fail_above=2, prefill_s=0.005, tpot_s=0.001)
    cfg = _cfg(concurrency_levels=[1, 6], requests_per_level=6, warmup_requests=0)
    summaries = asyncio.run(sweep(cfg, transport=server.transport()))
    assert summaries[0].errors == 0
    assert summaries[1].errors > 0
    assert find_operating_point(summaries, cfg).concurrency == 1


# ----------------------------------------------------------------------------- local_target


def test_check_fit_reports_each_problem():
    env = RequestEnvelope(input_tokens=7_000, max_output_tokens=1_500, needs_tools=True, needs_response_schema=True)
    problems = check_fit(LOCAL_CPU, env)
    assert len(problems) == 2
    assert any(p.startswith("context:") for p in problems)
    assert any(p.startswith("tools:") for p in problems)
    assert check_fit(HOSTED, env) == []


def test_choose_target_prefers_first_that_fits():
    order = [LOCAL_CPU, LOCAL_VLLM, HOSTED]
    small = RequestEnvelope(input_tokens=500, max_output_tokens=200)
    assert choose_target(order, small) is LOCAL_CPU
    with_tools = small.model_copy(update={"needs_tools": True})
    assert choose_target(order, with_tools) is LOCAL_VLLM
    huge = RequestEnvelope(input_tokens=100_000, max_output_tokens=1_000)
    assert choose_target(order, huge) is HOSTED
    too_big = RequestEnvelope(input_tokens=200_000, max_output_tokens=1_000)
    assert choose_target(order, too_big) is None


def test_serving_target_validation():
    with pytest.raises(ValueError):
        ServingTarget(name="x", base_url="http://x", model="m", max_context_tokens=0)


def test_gateway_streams_from_a_local_target_and_traces_the_engine():
    pytest.importorskip("aie_core")
    from aie_core import CompletionRequest, Message, ModelGateway
    from aie_core.observability import InMemoryTracer

    from fake_server import FakeServer
    from local_target import make_client

    server = FakeServer(capacity=4, prefill_s=0.0, tpot_s=0.0)
    target = LOCAL_VLLM.model_copy(update={"base_url": "http://inference.test/v1"})
    tracer = InMemoryTracer()
    gw = ModelGateway(make_client(target, async_transport=server.transport()), tracer=tracer)
    req = CompletionRequest(messages=[Message.user("status of the VPN?")], max_tokens=3)

    async def collect() -> list:
        return [ev async for ev in gw.astream(req)]

    events = asyncio.run(collect())
    assert "".join(e.text or "" for e in events if e.type == "text_delta") == "tok0 tok1 tok2 "
    assert server.seen[-1]["model"] == "open-8b-instruct" and server.seen[-1]["stream"] is True
    span = tracer.spans[-1]
    assert span.attributes["provider"] == "vllm:local-gpu"  # the trace says which deployment answered
    assert span.attributes["output_tokens"] == 3


def test_envelope_from_request_and_schema_fallback_on_a_target_without_it():
    pytest.importorskip("aie_core")
    import json

    import httpx
    from pydantic import BaseModel

    from aie_core import CompletionRequest, Message, ToolSpec
    from aie_core.llm.structured import complete_structured

    from local_target import envelope_for, make_client

    class Label(BaseModel):
        label: str

    req = CompletionRequest(messages=[Message.system("Classify."), Message.user("vpn drops")], max_tokens=50,
                            tools=[ToolSpec(name="lookup_employee", description="find an employee")])
    env = envelope_for(req)
    assert env.needs_tools and not env.needs_response_schema and env.max_output_tokens == 50
    assert check_fit(LOCAL_CPU, env) == ["tools: local-cpu does not support tool calling"]

    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "c1", "model": "open-8b-instruct-q4",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": '{"label": "vpn"}'}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5},
        })

    no_schema = LOCAL_CPU.model_copy(update={"supports_response_schema": False, "base_url": "http://cpu.test/v1"})
    client = make_client(no_schema, transport=httpx.MockTransport(handler))
    parsed, _ = complete_structured(client, CompletionRequest(messages=[Message.user("vpn drops")]), Label)
    assert parsed.label == "vpn"
    assert "response_format" not in sent[0]  # prompt-and-parse path, not an unsupported schema mode


def walkthrough_config() -> LoadTestConfig:
    """The setup behind the walkthrough table (fake timings and SLOs, illustrative)."""
    return LoadTestConfig(base_url="http://fake/v1", prompts=[f"Summarize ticket {i}" for i in range(8)],
                          max_tokens_choices=[32, 64], concurrency_levels=[1, 4, 8, 16, 32], requests_per_level=32,
                          warmup_requests=2, ttft_slo_s=0.15, e2e_slo_s=1.0)


def test_walkthrough_shape_has_a_knee_at_capacity():
    server = FakeServer(capacity=8, prefill_s=0.02, tpot_s=0.005, queue_slot_s=0.01)
    cfg = walkthrough_config().model_copy(update={"concurrency_levels": [4, 8, 16, 32]})  # skip the slow level 1
    s = asyncio.run(sweep(cfg, transport=server.transport()))
    by = {x.concurrency: x for x in s}
    assert server.peak_decoding == 8                                     # capacity is a real limit
    assert by[16].ttft_p50 > 5 * by[8].ttft_p50                          # queueing shows up in TTFT
    assert by[32].requests_per_s < 1.2 * by[8].requests_per_s            # throughput levels off
    assert by[16].goodput_requests_per_s < 0.5 * by[8].goodput_requests_per_s
    assert abs(by[32].tpot_p50 - by[4].tpot_p50) < 0.003                 # decode speed did not change
    assert find_operating_point(s, cfg).concurrency == 8


def test_cancelling_a_sweep_stops_it():
    import time
    server = FakeServer(capacity=1, prefill_s=0.05, tpot_s=0.01)
    cfg = walkthrough_config().model_copy(update={"concurrency_levels": [1], "requests_per_level": 20, "warmup_requests": 0})
    t0 = time.perf_counter()
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(asyncio.wait_for(sweep(cfg, transport=server.transport()), timeout=0.2))
    assert time.perf_counter() - t0 < 2.0
