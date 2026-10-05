# path: book/projects/examples/ch02/kv_cache_calc.py
"""KV-cache memory calculator.

During decode, the model keeps the key and value vectors of every earlier token in
every layer so it does not recompute them each step. That cache is the resource that
runs out first when you combine long context with concurrency. The size per sequence is

    bytes = 2 * layers * kv_heads * head_dim * tokens * bytes_per_elem

where the leading 2 counts keys and values. Everything else is a model constant except
``tokens`` (prompt + generated so far) and, across sequences, concurrency.

Run:
    python kv_cache_calc.py --layers 32 --kv-heads 8 --head-dim 128 --tokens 16000
    python kv_cache_calc.py --layers 32 --kv-heads 8 --head-dim 128 --tokens 16000 --memory-gb 40
    python kv_cache_calc.py ... --query-heads 32      # compare with a no-GQA cache
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

GIB = 1024**3
MIB = 1024**2


def kv_cache_bytes(layers: int, kv_heads: int, head_dim: int, tokens: int, bytes_per_elem: int) -> int:
    """Bytes of KV cache for ONE sequence of ``tokens`` tokens."""
    for name, value in (
        ("layers", layers),
        ("kv_heads", kv_heads),
        ("head_dim", head_dim),
        ("bytes_per_elem", bytes_per_elem),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if tokens < 0:
        raise ValueError(f"tokens must be >= 0, got {tokens}")
    return 2 * layers * kv_heads * head_dim * tokens * bytes_per_elem


def bytes_per_token(layers: int, kv_heads: int, head_dim: int, bytes_per_elem: int) -> int:
    """KV bytes added for every token kept in context. Handy for cost-per-token intuition."""
    return kv_cache_bytes(layers, kv_heads, head_dim, 1, bytes_per_elem)


def max_concurrent_sequences(memory_budget_bytes: int, per_sequence_bytes: int) -> int:
    """How many sequences of that size fit in the memory left for KV cache (floor)."""
    if per_sequence_bytes <= 0:
        raise ValueError("per_sequence_bytes must be positive")
    if memory_budget_bytes < 0:
        raise ValueError("memory budget must be >= 0")
    return memory_budget_bytes // per_sequence_bytes


def human_bytes(n: int | float) -> str:
    if n >= GIB:
        return f"{n / GIB:.2f} GiB"
    if n >= MIB:
        return f"{n / MIB:.1f} MiB"
    if n >= 1024:
        return f"{n / 1024:.1f} KiB"
    return f"{int(n)} B"


@dataclass(frozen=True)
class Report:
    per_token: int
    per_sequence: int
    total: int
    fits: int | None  # None when no memory budget was given

    def render(self, tokens: int, concurrency: int, memory_budget: int | None) -> str:
        lines = [
            f"per token kept in context : {human_bytes(self.per_token)}",
            f"per sequence ({tokens:,} tokens)  : {human_bytes(self.per_sequence)}",
            f"{concurrency} concurrent sequences    : {human_bytes(self.total)}",
        ]
        if memory_budget is not None and self.fits is not None:
            lines.append(
                f"budget {human_bytes(memory_budget)} fits        : {self.fits} sequences of this length"
            )
        return "\n".join(lines)


def build_report(
    layers: int,
    kv_heads: int,
    head_dim: int,
    tokens: int,
    bytes_per_elem: int,
    concurrency: int,
    memory_budget: int | None,
) -> Report:
    per_seq = kv_cache_bytes(layers, kv_heads, head_dim, tokens, bytes_per_elem)
    fits = max_concurrent_sequences(memory_budget, per_seq) if memory_budget is not None and per_seq else None
    return Report(
        per_token=bytes_per_token(layers, kv_heads, head_dim, bytes_per_elem),
        per_sequence=per_seq,
        total=per_seq * concurrency,
        fits=fits,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--kv-heads", type=int, required=True, help="key/value heads (after GQA/MQA sharing)")
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--tokens", type=int, required=True, help="tokens in context: prompt + output so far")
    parser.add_argument("--bytes", type=int, default=2, help="bytes per element: 2 for BF16/FP16, 1 for FP8/INT8")
    parser.add_argument("--concurrency", type=int, default=1, help="number of simultaneous sequences")
    parser.add_argument("--memory-gb", type=float, default=None, help="GiB available for KV cache after weights")
    parser.add_argument(
        "--query-heads",
        type=int,
        default=None,
        help="if given, also show the cache a plain multi-head model with this many K/V heads would need",
    )
    args = parser.parse_args(argv)

    budget = int(args.memory_gb * GIB) if args.memory_gb is not None else None
    report = build_report(args.layers, args.kv_heads, args.head_dim, args.tokens, args.bytes, args.concurrency, budget)
    print(report.render(args.tokens, args.concurrency, budget))

    if args.query_heads is not None and args.query_heads != args.kv_heads:
        mha = kv_cache_bytes(args.layers, args.query_heads, args.head_dim, args.tokens, args.bytes)
        ratio = mha / report.per_sequence
        print(
            f"\nwithout GQA ({args.query_heads} K/V heads) the same sequence would need "
            f"{human_bytes(mha)}, {ratio:.1f}x more"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
