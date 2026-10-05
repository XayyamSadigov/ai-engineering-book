# path: book/projects/examples/ch02/sampling.py
"""From logits to a token: softmax, temperature, top-k, top-p, and sampling.

This is the last step of every decode iteration, isolated so you can see what the
sampling knobs actually do. Everything here operates on one toy logit vector over a
tiny vocabulary; a real model has tens or hundreds of thousands of entries, but the
math is identical. The functions are pure NumPy so the demo runs anywhere.

Run:
    python sampling.py            # prints how each setting reshapes the distribution
    python sampling.py --draws 20000
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

# A toy next-token distribution for the prefix "The ticket was escalated to".
# Logits are unnormalized scores straight out of the model's output projection.
TOY_VOCAB: list[str] = [" the", " a", " tier", " engineering", " support", " management", " Paris", " purple"]
TOY_LOGITS: np.ndarray = np.array([4.0, 3.2, 3.0, 2.6, 2.4, 1.5, -1.0, -2.5])


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax: subtracting the max does not change the result."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Divide logits by T. T < 1 sharpens, T > 1 flattens, T -> 0 approaches greedy.

    T == 0 is handled by the caller as greedy (argmax) because division by zero is
    undefined; providers do the same.
    """
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if temperature == 0:
        raise ValueError("temperature 0 means greedy; call greedy() instead")
    return np.asarray(logits, dtype=np.float64) / temperature


def top_k_filter(probs: np.ndarray, k: int) -> np.ndarray:
    """Keep the k most probable tokens, zero the rest, renormalize."""
    if k <= 0:
        raise ValueError("k must be >= 1")
    p = np.asarray(probs, dtype=np.float64)
    if k >= p.size:
        return p / p.sum()
    keep = np.argsort(-p, kind="stable")[:k]
    out = np.zeros_like(p)
    out[keep] = p[keep]
    return out / out.sum()


def top_p_filter(probs: np.ndarray, p_threshold: float) -> np.ndarray:
    """Nucleus sampling: keep the smallest prefix (by descending prob) whose mass >= p.

    The surviving set adapts to confidence: a peaked distribution keeps one or two
    tokens, a flat one keeps many. The most probable token is always kept, so the
    result is never empty even for tiny thresholds.
    """
    if not 0 < p_threshold <= 1:
        raise ValueError("p must be in (0, 1]")
    p = np.asarray(probs, dtype=np.float64)
    order = np.argsort(-p, kind="stable")
    cumulative = np.cumsum(p[order])
    # index of the first position where cumulative mass reaches the threshold
    cutoff = int(np.searchsorted(cumulative, p_threshold, side="left"))
    keep = order[: cutoff + 1]
    out = np.zeros_like(p)
    out[keep] = p[keep]
    return out / out.sum()


def greedy(logits: np.ndarray) -> int:
    """Argmax. Deterministic given identical logits; ties resolve to the lowest index."""
    return int(np.argmax(np.asarray(logits, dtype=np.float64)))


def sample(
    logits: np.ndarray,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    rng: np.random.Generator | None = None,
) -> int:
    """One decode step: logits -> temperature -> softmax -> top-k -> top-p -> draw.

    Order matters and mirrors common engine implementations: temperature reshapes the
    logits first, then truncation removes the tail from the resulting probabilities.
    """
    if temperature == 0:
        return greedy(logits)
    probs = softmax(apply_temperature(logits, temperature))
    if top_k is not None:
        probs = top_k_filter(probs, top_k)
    if top_p is not None:
        probs = top_p_filter(probs, top_p)
    rng = rng or np.random.default_rng()
    return int(rng.choice(probs.size, p=probs))


def entropy_bits(probs: np.ndarray) -> float:
    """Shannon entropy in bits. 0 means certain; log2(n) means uniform over n tokens."""
    p = np.asarray(probs, dtype=np.float64)
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum()) + 0.0  # +0.0 turns -0.0 into 0.0


def support_size(probs: np.ndarray) -> int:
    """How many tokens still have nonzero probability after filtering."""
    return int((np.asarray(probs) > 0).sum())


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Setting:
    name: str
    temperature: float
    top_k: int | None = None
    top_p: float | None = None

    def distribution(self, logits: np.ndarray) -> np.ndarray:
        if self.temperature == 0:
            out = np.zeros(len(logits))
            out[greedy(logits)] = 1.0
            return out
        probs = softmax(apply_temperature(logits, self.temperature))
        if self.top_k is not None:
            probs = top_k_filter(probs, self.top_k)
        if self.top_p is not None:
            probs = top_p_filter(probs, self.top_p)
        return probs


DEMO_SETTINGS: list[Setting] = [
    Setting("greedy (T=0)", 0.0),
    Setting("T=0.3", 0.3),
    Setting("T=1.0", 1.0),
    Setting("T=2.0", 2.0),
    Setting("T=1.0, top_k=3", 1.0, top_k=3),
    Setting("T=1.0, top_p=0.9", 1.0, top_p=0.9),
    Setting("T=2.0, top_p=0.9", 2.0, top_p=0.9),
]


def distribution_table(logits: np.ndarray, vocab: list[str], settings: list[Setting]) -> str:
    width = max(len(v) for v in vocab) + 2
    header = f"{'token':<{width}}" + "".join(f"{s.name:>18}" for s in settings)
    lines = [header, "-" * len(header)]
    dists = [s.distribution(logits) for s in settings]
    for i, tok in enumerate(vocab):
        cells = "".join(f"{d[i]:>18.3f}" for d in dists)
        lines.append(f"{tok!r:<{width}}" + cells)
    lines.append("-" * len(header))
    lines.append(f"{'entropy (bits)':<{width}}" + "".join(f"{entropy_bits(d):>18.2f}" for d in dists))
    lines.append(f"{'candidates left':<{width}}" + "".join(f"{support_size(d):>18d}" for d in dists))
    return "\n".join(lines)


def empirical_frequencies(
    logits: np.ndarray, setting: Setting, draws: int, seed: int = 7
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    counts = np.zeros(len(logits))
    for _ in range(draws):
        idx = sample(logits, temperature=setting.temperature, top_k=setting.top_k, top_p=setting.top_p, rng=rng)
        counts[idx] += 1
    return counts / draws


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draws", type=int, default=5000, help="samples per setting for the empirical check")
    args = parser.parse_args(argv)

    print("Next-token distribution for the prefix 'The ticket was escalated to'\n")
    print(distribution_table(TOY_LOGITS, TOY_VOCAB, DEMO_SETTINGS))

    print(f"\nEmpirical frequencies over {args.draws} draws (should track the columns above):")
    for setting in (DEMO_SETTINGS[1], DEMO_SETTINGS[3], DEMO_SETTINGS[5]):
        freq = empirical_frequencies(TOY_LOGITS, setting, args.draws)
        top = np.argsort(-freq)[:4]
        shown = ", ".join(f"{TOY_VOCAB[i]!r}={freq[i]:.3f}" for i in top)
        print(f"  {setting.name:<18} {shown}")

    print("\nGreedy picks the same token every time:", {greedy(TOY_LOGITS) for _ in range(100)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
