# path: book/projects/examples/ch34/loadtest.py
"""Async load generator for any OpenAI-compatible chat endpoint.

Closed-loop design: at concurrency N, N workers each keep exactly one request open and start
the next one when the previous finishes. Sweeping N and watching p95 TTFT and p95 end-to-end
latency is how you locate the operating point. Streaming is mandatory because TTFT and TPOT
are only observable from the first and subsequent content deltas.

Run against a real server:

    python loadtest.py --base-url http://localhost:8000/v1 --model my-model \
        --levels 1,2,4,8,16 --requests 24 --prompt-file prompts.txt

Tests inject an ``httpx.AsyncBaseTransport`` (see ``fake_server.py``) so no network is used.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field


class LoadTestConfig(BaseModel):
    base_url: str = "http://localhost:8000/v1"
    model: str = "local-model"
    api_key: str = "not-needed"
    prompts: list[str] = Field(min_length=1, description="realistic prompt distribution, sampled uniformly")
    max_tokens_choices: list[int] = Field(default=[64, 128, 256], min_length=1)
    concurrency_levels: list[int] = Field(default=[1, 2, 4, 8], min_length=1)
    requests_per_level: int = Field(default=16, ge=1)
    warmup_requests: int = Field(default=2, ge=0)
    timeout_s: float = Field(default=120.0, gt=0)
    temperature: float = 0.0
    ttft_slo_s: float = Field(default=2.0, gt=0)
    e2e_slo_s: float = Field(default=8.0, gt=0)
    seed: int = 7


class RequestResult(BaseModel):
    concurrency: int
    ok: bool
    error: str | None = None
    status_code: int | None = None
    prompt_chars: int = 0
    max_tokens: int = 0
    output_tokens: int = 0
    ttft_s: float | None = None
    tpot_s: float | None = None  # mean inter-token latency after the first token
    e2e_s: float = 0.0

    def meets_slo(self, cfg: LoadTestConfig) -> bool:
        return self.ok and self.ttft_s is not None and self.ttft_s <= cfg.ttft_slo_s and self.e2e_s <= cfg.e2e_slo_s


class LevelSummary(BaseModel):
    concurrency: int
    requests: int
    errors: int
    wall_s: float
    ttft_p50: float
    ttft_p95: float
    ttft_p99: float
    tpot_p50: float
    tpot_p95: float
    e2e_p50: float
    e2e_p95: float
    e2e_p99: float
    requests_per_s: float
    output_tokens_per_s: float
    slo_pass_fraction: float
    goodput_requests_per_s: float  # completed requests per second that met the SLO


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile; deterministic and good enough for load-test reporting."""
    if not values:
        return math.nan
    if not 0 <= p <= 100:
        raise ValueError("p must be in [0, 100]")
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


def parse_sse_line(line: str) -> dict[str, Any] | None:
    """Return the JSON payload of a ``data:`` line, ``None`` for keep-alives and ``[DONE]``."""
    if not line.startswith("data:"):
        return None
    payload = line[len("data:"):].strip()
    if not payload or payload == "[DONE]":
        return None
    return json.loads(payload)


def _delta_text(chunk: dict[str, Any]) -> str:
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    return delta.get("content") or ""


async def run_one(
    client: httpx.AsyncClient, cfg: LoadTestConfig, prompt: str, max_tokens: int, concurrency: int
) -> RequestResult:
    """One streamed chat completion with TTFT, TPOT, and end-to-end timing."""
    body = {
        "model": cfg.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": cfg.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    result = RequestResult(concurrency=concurrency, ok=False, prompt_chars=len(prompt), max_tokens=max_tokens)
    start = time.perf_counter()
    first_token_at: float | None = None
    last_token_at: float | None = None
    tokens = 0
    reported_tokens: int | None = None
    try:
        async with client.stream("POST", "/chat/completions", json=body, timeout=cfg.timeout_s) as resp:
            result.status_code = resp.status_code
            if resp.status_code != 200:
                await resp.aread()
                result.error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                result.e2e_s = time.perf_counter() - start
                return result
            async for line in resp.aiter_lines():
                chunk = parse_sse_line(line)
                if chunk is None:
                    continue
                usage = chunk.get("usage")
                if usage and usage.get("completion_tokens") is not None:
                    reported_tokens = int(usage["completion_tokens"])
                if _delta_text(chunk):
                    now = time.perf_counter()
                    if first_token_at is None:
                        first_token_at = now
                    last_token_at = now
                    tokens += 1
    except (httpx.HTTPError, json.JSONDecodeError, asyncio.CancelledError) as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        result.e2e_s = time.perf_counter() - start
        return result

    end = time.perf_counter()
    result.e2e_s = end - start
    # Content deltas are an approximation of tokens; prefer the server's usage count if given.
    result.output_tokens = reported_tokens if reported_tokens is not None else tokens
    if first_token_at is None:
        result.error = "no content tokens received"
        return result
    result.ttft_s = first_token_at - start
    if last_token_at is not None and result.output_tokens > 1:
        result.tpot_s = (last_token_at - first_token_at) / (result.output_tokens - 1)
    result.ok = True
    return result


async def run_level(
    client: httpx.AsyncClient, cfg: LoadTestConfig, concurrency: int, n_requests: int, rng: random.Random
) -> list[RequestResult]:
    """Closed loop: ``concurrency`` workers drain a shared queue of ``n_requests`` jobs."""
    queue: asyncio.Queue[tuple[str, int]] = asyncio.Queue()
    for _ in range(n_requests):
        queue.put_nowait((rng.choice(cfg.prompts), rng.choice(cfg.max_tokens_choices)))
    results: list[RequestResult] = []

    async def worker() -> None:
        while True:
            try:
                prompt, max_tokens = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            results.append(await run_one(client, cfg, prompt, max_tokens, concurrency))

    await asyncio.gather(*(worker() for _ in range(min(concurrency, n_requests))))
    return results


def summarize(results: list[RequestResult], wall_s: float, cfg: LoadTestConfig) -> LevelSummary:
    ok = [r for r in results if r.ok]
    ttft = [r.ttft_s for r in ok if r.ttft_s is not None]
    tpot = [r.tpot_s for r in ok if r.tpot_s is not None]
    e2e = [r.e2e_s for r in ok]
    passed = sum(1 for r in results if r.meets_slo(cfg))
    wall = max(wall_s, 1e-9)
    return LevelSummary(
        concurrency=results[0].concurrency if results else 0,
        requests=len(results),
        errors=len(results) - len(ok),
        wall_s=wall_s,
        ttft_p50=percentile(ttft, 50), ttft_p95=percentile(ttft, 95), ttft_p99=percentile(ttft, 99),
        tpot_p50=percentile(tpot, 50), tpot_p95=percentile(tpot, 95),
        e2e_p50=percentile(e2e, 50), e2e_p95=percentile(e2e, 95), e2e_p99=percentile(e2e, 99),
        requests_per_s=len(ok) / wall,
        output_tokens_per_s=sum(r.output_tokens for r in ok) / wall,
        slo_pass_fraction=passed / len(results) if results else 0.0,
        goodput_requests_per_s=passed / wall,
    )


async def sweep(cfg: LoadTestConfig, transport: httpx.AsyncBaseTransport | None = None) -> list[LevelSummary]:
    """Warm up, then run every concurrency level and summarize each one."""
    rng = random.Random(cfg.seed)
    headers = {"Authorization": f"Bearer {cfg.api_key}"}
    summaries: list[LevelSummary] = []
    async with httpx.AsyncClient(base_url=cfg.base_url, headers=headers, transport=transport) as client:
        if cfg.warmup_requests:
            await run_level(client, cfg, concurrency=1, n_requests=cfg.warmup_requests, rng=rng)
        for level in cfg.concurrency_levels:
            t0 = time.perf_counter()
            results = await run_level(client, cfg, level, cfg.requests_per_level, rng)
            summaries.append(summarize(results, time.perf_counter() - t0, cfg))
    return summaries


def find_operating_point(summaries: list[LevelSummary], cfg: LoadTestConfig) -> LevelSummary | None:
    """Highest concurrency whose p95 TTFT and p95 E2E meet the SLO with zero errors."""
    passing = [
        s for s in summaries
        if s.errors == 0 and s.ttft_p95 <= cfg.ttft_slo_s and s.e2e_p95 <= cfg.e2e_slo_s
    ]
    return max(passing, key=lambda s: s.concurrency) if passing else None


def format_report(summaries: list[LevelSummary]) -> str:
    header = (
        f"{'conc':>5} {'n':>4} {'err':>4} {'ttft p50':>9} {'ttft p95':>9} {'tpot p50':>9} "
        f"{'e2e p50':>8} {'e2e p95':>8} {'req/s':>7} {'tok/s':>8} {'slo%':>6} {'goodput':>8}"
    )
    rows = [header, "-" * len(header)]
    for s in summaries:
        rows.append(
            f"{s.concurrency:>5} {s.requests:>4} {s.errors:>4} {s.ttft_p50:>9.3f} {s.ttft_p95:>9.3f} "
            f"{s.tpot_p50:>9.4f} {s.e2e_p50:>8.2f} {s.e2e_p95:>8.2f} {s.requests_per_s:>7.2f} "
            f"{s.output_tokens_per_s:>8.1f} {s.slo_pass_fraction * 100:>6.1f} {s.goodput_requests_per_s:>8.2f}"
        )
    return "\n".join(rows)


def _parse_args(argv: list[str]) -> LoadTestConfig:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", default="local-model")
    p.add_argument("--api-key", default="not-needed")
    p.add_argument("--levels", default="1,2,4,8", help="comma-separated concurrency levels")
    p.add_argument("--requests", type=int, default=16, help="requests per level")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--max-tokens", default="64,128,256", help="comma-separated choices")
    p.add_argument("--prompt-file", type=Path, help="one prompt per line; default is a built-in mix")
    p.add_argument("--ttft-slo", type=float, default=2.0)
    p.add_argument("--e2e-slo", type=float, default=8.0)
    a = p.parse_args(argv)
    prompts = (
        [ln for ln in a.prompt_file.read_text().splitlines() if ln.strip()]
        if a.prompt_file
        else [
            "Summarize the Northwind parental leave policy in three sentences.",
            "List the steps of the IT runbook for a VPN outage. " * 20,
            "Classify this ticket: 'My laptop will not charge since the update.'",
        ]
    )
    return LoadTestConfig(
        base_url=a.base_url, model=a.model, api_key=a.api_key, prompts=prompts,
        max_tokens_choices=[int(x) for x in a.max_tokens.split(",")],
        concurrency_levels=[int(x) for x in a.levels.split(",")],
        requests_per_level=a.requests, warmup_requests=a.warmup,
        ttft_slo_s=a.ttft_slo, e2e_slo_s=a.e2e_slo,
    )


def main(argv: list[str] | None = None) -> int:
    cfg = _parse_args(sys.argv[1:] if argv is None else argv)
    summaries = asyncio.run(sweep(cfg))
    print(format_report(summaries))
    op = find_operating_point(summaries, cfg)
    if op is None:
        print(f"\nNo level met the SLO (TTFT p95 <= {cfg.ttft_slo_s}s, E2E p95 <= {cfg.e2e_slo_s}s).")
        return 1
    print(f"\nOperating point: concurrency {op.concurrency}, goodput {op.goodput_requests_per_s:.2f} req/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
