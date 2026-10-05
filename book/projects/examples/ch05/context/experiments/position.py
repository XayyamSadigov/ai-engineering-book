# path: book/projects/examples/ch05/context/experiments/position.py
"""Lost-in-the-middle harness: move one required fact through the evidence and measure accuracy.

Run offline with a simulated reader (demonstrates the harness, says nothing about any real
model), or against your configured provider to measure the model you actually ship:

    python -m context.experiments.position                 # simulated, offline
    LLM_PROVIDER=openai LLM_MODEL=... python -m context.experiments.position --live

The simulated reader exists so the harness itself has tests: if the harness cannot detect a
U-shaped curve that we planted, it will not detect one in a real model either.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import random
import re
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel

from aie_core.llm.client import LLMClient
from aie_core.llm.providers import FakeLLM
from aie_core.llm.types import CompletionRequest, Role

from ..builder import BudgetPolicy, ContextBuilder
from ..filters import RequestScope
from ..items import ContextItem, Trust
from ..labels import UNTRUSTED_TAG

SHARED_DOCS = Path(__file__).resolve().parents[4] / "shared-data" / "docs"
INSTRUCTIONS = (
    "You answer questions using only the provided documents. "
    "Reply with the requested value only. If the documents do not contain it, reply NOT FOUND."
)


class Needle(BaseModel):
    text: str
    question: str
    answer: str


DEFAULT_NEEDLE = Needle(
    text="Parking reimbursement for the Riverside office is claimed with expense code PRK-5823.",
    question="Which expense code is used to claim parking reimbursement for the Riverside office?",
    answer="PRK-5823",
)


class PositionRow(BaseModel):
    position: float  # 0.0 = first evidence block, 1.0 = last (right before the question)
    index: int
    trials: int
    correct: int
    mean_prompt_tokens: float

    @property
    def accuracy(self) -> float:
        return self.correct / self.trials if self.trials else 0.0


class SweepResult(BaseModel):
    n_docs: int
    rows: list[PositionRow]

    def spread(self) -> float:
        accs = [r.accuracy for r in self.rows]
        return max(accs) - min(accs)

    def table(self) -> str:
        lines = ["position  index  accuracy  trials  mean_prompt_tokens"]
        for r in self.rows:
            lines.append(f"{r.position:>8.2f}  {r.index:>5}  {r.accuracy:>8.2f}  {r.trials:>6}  {r.mean_prompt_tokens:>18.0f}")
        return "\n".join(lines)


def load_distractors(docs_dir: Path = SHARED_DOCS, min_chars: int = 160) -> list[str]:
    """Paragraphs from the Northwind sample docs; realistic distractors beat lorem ipsum."""
    paragraphs: list[str] = []
    if docs_dir.is_dir():
        for path in sorted(docs_dir.glob("*.md")):
            body = path.read_text(encoding="utf-8").split("---", 2)[-1]
            for para in re.split(r"\n\s*\n", body):
                para = " ".join(para.split())
                if len(para) >= min_chars and not para.startswith("#"):
                    paragraphs.append(para)
    if len(paragraphs) < 40:  # fallback so the harness works outside the book repository
        topics = ["VPN access", "laptop refresh", "travel booking", "PTO carryover", "incident paging"]
        paragraphs += [
            f"Northwind guidance note {i} on {topics[i % len(topics)]}: follow the standard procedure, "
            f"record the request in the service desk, and wait for approval from the owning team before proceeding."
            for i in range(60)
        ]
    return paragraphs


def build_haystack_prompt(
    needle: Needle, distractors: Sequence[str], index: int, counter: Callable[[str], int] | None = None
) -> tuple[list, int]:
    """Evidence in exactly the given order, needle at `index`; returns messages and prompt tokens."""
    docs = list(distractors)
    docs.insert(index, needle.text)
    n = len(docs)
    items = [ContextItem(kind="instructions", content=INSTRUCTIONS, source_id="sys:position", trust=Trust.TRUSTED, pinned=True)]
    for i, doc in enumerate(docs):
        items.append(
            ContextItem(
                kind="evidence",
                content=doc,
                source_id=f"doc:{i}",
                priority=1.0 - i / n,  # strictly decreasing, so "ranked" placement keeps this exact order
                metadata={"tenant": "shared", "acl_groups": ["all"]},
            )
        )
    items.append(ContextItem(kind="query", content=needle.question, source_id="user:q"))
    builder = ContextBuilder(
        BudgetPolicy(context_window=1_000_000, output_reserve=64),
        placement="ranked",
        near_duplicate_threshold=1.01,  # the experiment controls content; do not let dedupe move things
        counter=counter,
    )
    result = builder.build(items, RequestScope(user_id="experiment", tenant="retail", groups=["all"]))
    return result.messages, result.estimated_prompt_tokens


def run_position_sweep(
    client: LLMClient,
    needle: Needle = DEFAULT_NEEDLE,
    distractors: Sequence[str] | None = None,
    *,
    positions: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    n_docs: int = 20,
    trials: int = 10,
    seed: int = 0,
    model: str | None = None,
    counter: Callable[[str], int] | None = None,
) -> SweepResult:
    pool = list(distractors if distractors is not None else load_distractors())
    if len(pool) < n_docs - 1:
        raise ValueError(f"need {n_docs - 1} distractors, have {len(pool)}")
    rng = random.Random(seed)
    rows: list[PositionRow] = []
    # Same distractor samples for every position: only the needle's position varies.
    samples = [rng.sample(pool, n_docs - 1) for _ in range(trials)]
    for pos in positions:
        index = round(pos * (n_docs - 1))
        correct, tokens = 0, 0
        for sample in samples:
            messages, prompt_tokens = build_haystack_prompt(needle, sample, index, counter)
            req = CompletionRequest(messages=messages, model=model, temperature=0.0, max_tokens=32,
                                    metadata={"experiment": "position", "position": pos})
            text = client.complete(req).text
            correct += needle.answer.lower() in text.lower()
            tokens += prompt_tokens
        rows.append(PositionRow(position=pos, index=index, trials=trials, correct=correct,
                                mean_prompt_tokens=tokens / trials))
    return SweepResult(n_docs=n_docs, rows=rows)


class SimulatedPositionalReader:
    """A FakeLLM handler that answers correctly with a probability that depends on position.

    accuracy(rel) = edge - (edge - middle) * sin(pi * rel), rel in [0, 1]. With middle == edge
    the reader is position-blind. Values are illustrative, chosen to make the harness testable.
    Deterministic: the coin flip is seeded by a hash of the prompt.
    """

    def __init__(self, answer: str, edge: float = 0.95, middle: float = 0.55) -> None:
        self.answer = answer
        self.edge = edge
        self.middle = middle
        self._block = re.compile(rf"<{UNTRUSTED_TAG}[^>]*>\n(.*?)\n</{UNTRUSTED_TAG}>", re.S)

    def __call__(self, req: CompletionRequest) -> str:
        user = next(m.text for m in reversed(req.messages) if m.role is Role.USER)
        blocks = self._block.findall(user)
        idx = next((i for i, b in enumerate(blocks) if self.answer in b), None)
        if idx is None:
            return "NOT FOUND"
        rel = idx / (len(blocks) - 1) if len(blocks) > 1 else 0.0
        p = self.edge - (self.edge - self.middle) * math.sin(math.pi * rel)
        coin = int(hashlib.sha256(user.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        return self.answer if coin < p else "NOT FOUND"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--live", action="store_true", help="use the provider configured via LLM_PROVIDER")
    parser.add_argument("--docs", type=int, default=20)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.live:
        from aie_core.settings import make_llm_client

        client: LLMClient = make_llm_client()
    else:
        client = FakeLLM(handler=SimulatedPositionalReader(DEFAULT_NEEDLE.answer))
    result = run_position_sweep(client, n_docs=args.docs, trials=args.trials, seed=args.seed)
    print(result.table())
    print(f"spread (max - min accuracy): {result.spread():.2f}")


if __name__ == "__main__":
    main()


__all__ = [
    "Needle", "DEFAULT_NEEDLE", "PositionRow", "SweepResult", "run_position_sweep",
    "build_haystack_prompt", "load_distractors", "SimulatedPositionalReader", "main",
]
