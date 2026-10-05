# path: book/projects/examples/ch23/signature.py
"""DSPy's core idea in plain Python: a typed prompt *specification*
(a Signature) that a Module renders into prompt text and parses back, plus
an Optimizer that picks few-shot demonstrations by a metric.

The model client is any `Callable[[str], str]`. No framework imports.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

LLMFn = Callable[[str], str]


@dataclass(frozen=True)
class Field:
    name: str
    desc: str
    kind: type = str          # str, int, float, bool; parse happens in `parse`


@dataclass(frozen=True)
class Signature:
    """What goes in, what comes out, and one sentence of instruction.
    Prompt wording lives in the Module, not here: the spec is stable, the
    rendering is swappable. This is the property Chapter 4's registry wants."""

    instruction: str
    inputs: tuple[Field, ...]
    outputs: tuple[Field, ...]

    def render(self, values: dict[str, Any], demos: Sequence[dict[str, Any]] = ()) -> str:
        lines = [self.instruction, "", "Fields:"]
        for f in self.inputs + self.outputs:
            lines.append(f"- {f.name}: {f.desc}")
        for ex in demos:
            lines.append("")
            lines += [f"{f.name}: {ex[f.name]}" for f in self.inputs + self.outputs]
        lines.append("")
        lines += [f"{f.name}: {values[f.name]}" for f in self.inputs]
        lines.append(f"{self.outputs[0].name}:")
        return "\n".join(lines)

    def parse(self, text: str) -> dict[str, Any]:
        """Read `name: value` sections for every output field, typed."""
        out: dict[str, Any] = {}
        names = [f.name for f in self.outputs]
        for i, f in enumerate(self.outputs):
            pattern = rf"(?:^|\n){f.name}:\s*(.*?)(?=\n(?:{'|'.join(map(re.escape, names))}):|\Z)"
            m = re.search(pattern, text, flags=re.S)
            if m:
                raw = m.group(1)
            elif i == 0:   # first field may arrive unlabeled: take text up to the next label
                raw = re.split(rf"\n(?:{'|'.join(map(re.escape, names))}):", text, maxsplit=1)[0]
            else:
                raw = ""
            raw = raw.strip()
            out[f.name] = _coerce(raw, f.kind, f.name)
        return out


def _coerce(raw: str, kind: type, name: str) -> Any:
    try:
        if kind is bool:
            return raw.lower() in {"true", "yes", "1"}
        return kind(raw)
    except ValueError as exc:
        raise ValueError(f"field {name!r}: cannot parse {raw!r} as {kind.__name__}") from exc


@dataclass
class Predict:
    """The simplest Module: render, call, parse. Demos are *state* of the
    module that an optimizer may set; the signature never changes."""

    signature: Signature
    llm: LLMFn
    demos: list[dict[str, Any]] = field(default_factory=list)
    calls: int = 0

    def __call__(self, **inputs: Any) -> dict[str, Any]:
        self.calls += 1
        completion = self.llm(self.signature.render(inputs, self.demos))
        result = self.signature.parse(completion)
        # First output field may come back without its label; keep that lenient path.
        return result


Metric = Callable[[dict[str, Any], dict[str, Any]], float]   # (example, prediction) -> score


@dataclass
class BootstrapFewShot:
    """Optimizer: run the module over a train set, keep the demonstrations the
    metric accepts, then choose the demo subset that scores best on a dev set.
    'Compiling' a prompt means exactly this: a search over prompt *contents*
    driven by a metric and data, nothing more mysterious."""

    metric: Metric
    max_demos: int = 3
    threshold: float = 1.0

    def compile(self, module: Predict, train: Sequence[dict[str, Any]],
                dev: Sequence[dict[str, Any]]) -> Predict:
        input_names = [f.name for f in module.signature.inputs]
        candidates: list[dict[str, Any]] = []
        for ex in train:
            pred = module(**{k: ex[k] for k in input_names})
            if self.metric(ex, pred) >= self.threshold:
                candidates.append({**{k: ex[k] for k in input_names}, **pred})
        best_demos, best_score = [], self.evaluate(module, dev)
        for n in range(1, min(self.max_demos, len(candidates)) + 1):
            trial = Predict(module.signature, module.llm, demos=candidates[:n])
            score = self.evaluate(trial, dev)
            if score > best_score:
                best_demos, best_score = candidates[:n], score
        return Predict(module.signature, module.llm, demos=best_demos)

    def evaluate(self, module: Predict, dataset: Sequence[dict[str, Any]]) -> float:
        input_names = [f.name for f in module.signature.inputs]
        if not dataset:
            return 0.0
        scores = [self.metric(ex, module(**{k: ex[k] for k in input_names})) for ex in dataset]
        return sum(scores) / len(scores)
