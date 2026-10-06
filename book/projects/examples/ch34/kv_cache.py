# path: book/projects/examples/ch34/kv_cache.py
"""KV-cache sizing and concurrency estimation for self-hosted decoder models.

Every number this module produces is an *estimate before allocator overhead*: real engines add
block metadata, fragmentation, CUDA graphs, activation workspaces, and communication buffers.
Use these functions to decide whether a plan is plausible, then confirm with a load test
(see ``loadtest.py``). Pure arithmetic, no I/O.
"""
from __future__ import annotations

import math

from pydantic import BaseModel, Field, PositiveInt

KiB = 1024
MiB = 1024**2
GiB = 1024**3


class ModelShape(BaseModel):
    """The handful of architecture facts that drive serving memory.

    Read them from the model's config file (``num_hidden_layers``, ``num_key_value_heads``,
    ``hidden_size // num_attention_heads``). ``kv_heads`` is the number of key/value heads,
    which with grouped-query attention is smaller than the number of query heads.
    """

    name: str = "illustrative-8b"
    layers: PositiveInt
    kv_heads: PositiveInt
    head_dim: PositiveInt
    params_billion: float = Field(gt=0, description="total parameters, in billions")


class Precision(BaseModel):
    """Bytes per element for weights and for the KV cache. They may differ."""

    weight_bytes: float = Field(default=2.0, gt=0, description="2 = BF16/FP16, 1 = INT8/FP8, 0.5 = INT4")
    kv_bytes: float = Field(default=2.0, gt=0, description="2 = BF16 cache, 1 = FP8/INT8 cache")


def kv_bytes_per_token(shape: ModelShape, kv_bytes: float = 2.0) -> int:
    """Bytes of KV cache that one token occupies across all layers.

    Formula: 2 (keys and values) * layers * kv_heads * head_dim * bytes_per_element.
    """
    return int(2 * shape.layers * shape.kv_heads * shape.head_dim * kv_bytes)


def kv_bytes_per_sequence(shape: ModelShape, tokens: int, kv_bytes: float = 2.0) -> int:
    """KV cache for one sequence of ``tokens`` tokens (prompt plus generated so far)."""
    if tokens < 0:
        raise ValueError("tokens must be non-negative")
    return kv_bytes_per_token(shape, kv_bytes) * tokens


def weight_bytes(shape: ModelShape, weight_bytes_per_param: float = 2.0) -> int:
    """Memory for the weights alone. Scales and metadata for quantized formats add a few percent."""
    return int(shape.params_billion * 1e9 * weight_bytes_per_param)


class ConcurrencyEstimate(BaseModel):
    gpu_memory_bytes: int
    weight_bytes: int
    headroom_bytes: int
    runtime_overhead_bytes: int
    usable_kv_bytes: int
    per_sequence_bytes: int
    max_sequences: int

    def summary(self) -> str:
        return (
            f"GPU {self.gpu_memory_bytes / GiB:.0f} GiB, weights {self.weight_bytes / GiB:.1f} GiB, "
            f"headroom {self.headroom_bytes / GiB:.1f} GiB, runtime {self.runtime_overhead_bytes / GiB:.1f} GiB "
            f"-> {self.usable_kv_bytes / GiB:.1f} GiB for KV; {self.per_sequence_bytes / GiB:.2f} GiB per "
            f"sequence -> about {self.max_sequences} concurrent sequences"
        )


def max_concurrent_sequences(
    shape: ModelShape,
    tokens_per_sequence: int,
    gpu_memory_bytes: int,
    precision: Precision | None = None,
    headroom_fraction: float = 0.10,
    runtime_overhead_bytes: int = 2 * GiB,
    tensor_parallel: int = 1,
) -> ConcurrencyEstimate:
    """How many sequences of a given length fit in KV memory once weights are resident.

    ``tensor_parallel`` devices share weights and cache evenly, so the estimate is for the whole
    group. ``headroom_fraction`` is memory you refuse to plan into: fragmentation and bursts.
    ``runtime_overhead_bytes`` is per device (CUDA context, workspaces), so it scales with the group.
    """
    precision = precision or Precision()
    if not 0 <= headroom_fraction < 1:
        raise ValueError("headroom_fraction must be in [0, 1)")
    total = gpu_memory_bytes * tensor_parallel
    weights = weight_bytes(shape, precision.weight_bytes)
    headroom = int(total * headroom_fraction)
    runtime = runtime_overhead_bytes * tensor_parallel
    usable = total - weights - headroom - runtime
    per_seq = kv_bytes_per_sequence(shape, tokens_per_sequence, precision.kv_bytes)
    max_seq = 0 if usable <= 0 or per_seq == 0 else usable // per_seq
    return ConcurrencyEstimate(
        gpu_memory_bytes=total,
        weight_bytes=weights,
        headroom_bytes=headroom,
        runtime_overhead_bytes=runtime,
        usable_kv_bytes=max(usable, 0),
        per_sequence_bytes=per_seq,
        max_sequences=int(max_seq),
    )


def tokens_that_fit(shape: ModelShape, kv_budget_bytes: int, kv_bytes: float = 2.0) -> int:
    """Inverse question: given a KV budget, how many total tokens can be resident at once?"""
    per_token = kv_bytes_per_token(shape, kv_bytes)
    return 0 if kv_budget_bytes <= 0 else kv_budget_bytes // per_token


def format_bytes(n: int | float) -> str:
    """Human-readable binary units; the text of the chapter uses the same units."""
    if n >= GiB:
        return f"{n / GiB:.2f} GiB"
    if n >= MiB:
        return f"{n / MiB:.1f} MiB"
    if n >= KiB:
        return f"{n / KiB:.1f} KiB"
    return f"{int(n)} B"


def gqa_savings_factor(query_heads: int, kv_heads: int) -> float:
    """KV memory ratio of grouped-query attention vs. full multi-head attention."""
    if kv_heads <= 0 or query_heads <= 0 or kv_heads > query_heads:
        raise ValueError("need 0 < kv_heads <= query_heads")
    return kv_heads / query_heads


if __name__ == "__main__":
    # Illustrative shape: 32 layers, 8 KV heads, head dimension 128, 8B parameters.
    shape = ModelShape(layers=32, kv_heads=8, head_dim=128, params_billion=8)
    print(f"{shape.name}: {format_bytes(kv_bytes_per_token(shape))} per token")
    for ctx in (8_192, 32_768):
        print(f"  {ctx:>6} tokens -> {format_bytes(kv_bytes_per_sequence(shape, ctx))} per sequence")
    for ctx in (8_192, 32_768):
        est = max_concurrent_sequences(shape, ctx, gpu_memory_bytes=80 * GiB)
        print(f"  ctx {ctx}: {est.summary()}")
    est_fp8 = max_concurrent_sequences(
        shape, 32_768, gpu_memory_bytes=80 * GiB, precision=Precision(weight_bytes=1.0, kv_bytes=1.0)
    )
    print(f"  ctx 32768 with INT8 weights + FP8 cache: {est_fp8.summary()}")
    print(f"  math.ceil check: {math.ceil(est_fp8.per_sequence_bytes / GiB)} GiB per sequence rounded up")
