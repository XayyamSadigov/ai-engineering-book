# path: book/projects/examples/ch38/voice_gate.py
"""Three small, testable pieces of a realtime voice agent's harness.

- `TurnGate`: decides what a transcript segment may trigger. Unstable partials may start
  speculative, read-only work; side-effecting tools need a final, confident transcript.
- `PlaybackController`: tracks how much of the reply the caller actually heard, so that on
  barge-in the conversation history records the spoken prefix, not the planned text.
- `LatencyBudget`: per-stage p95 budgets for one conversational turn, checked against
  measurements. Numbers are illustrative defaults, not SLOs.

The audio stack itself (VAD, ASR, TTS, transport) is out of scope; these classes sit between
it and the agent runtime and hold the rules that must not depend on model behavior.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from agentkit import SideEffect

GateDecision = Literal["ignore", "speculate", "commit", "reprompt"]


class Segment(BaseModel):
    text: str
    is_final: bool
    stability: float = Field(ge=0, le=1, default=0.0)    # ASR's estimate that this partial will not change
    confidence: float = Field(ge=0, le=1, default=0.0)   # ASR confidence for final segments


@dataclass
class TurnGate:
    min_stability: float = 0.8
    min_confidence: float = 0.85

    def classify(self, seg: Segment) -> GateDecision:
        if not seg.text.strip():
            return "ignore"
        if not seg.is_final:
            return "speculate" if seg.stability >= self.min_stability else "ignore"
        return "commit" if seg.confidence >= self.min_confidence else "reprompt"

    def allowed_effects(self, decision: GateDecision) -> set[SideEffect]:
        """Speculation may read (look up the account, prefetch the policy); only a committed
        turn may write, and writes still go through the normal policy and confirmation."""
        if decision == "speculate":
            return {SideEffect.READ}
        if decision == "commit":
            return set(SideEffect)
        return set()


@dataclass
class PlaybackController:
    """Text goes to TTS in chunks; the audio clock reports how many characters were played."""

    planned: list[str] = field(default_factory=list)
    played_chars: int = 0
    cancelled: bool = False

    def enqueue(self, chunk: str) -> None:
        if not self.cancelled:
            self.planned.append(chunk)

    def on_played(self, chars: int) -> None:
        self.played_chars = min(self.played_chars + chars, len("".join(self.planned)))

    def barge_in(self) -> str:
        """Stop playback, drop unplayed chunks, return what the caller heard. The caller must
        also cancel the in-flight generation so no more chunks arrive."""
        self.cancelled = True
        heard = "".join(self.planned)[: self.played_chars]
        self.planned = [heard]
        return heard

    def history_text(self) -> str:
        text = "".join(self.planned)[: self.played_chars] if self.cancelled else "".join(self.planned)
        return text + (" [interrupted by caller]" if self.cancelled else "")


DEFAULT_BUDGET_MS = {            # illustrative p95 allocations for one turn
    "endpointing": 250,          # VAD + end-of-turn decision after the caller stops
    "asr_final": 200,
    "llm_ttft": 500,
    "tts_first_audio": 200,
    "transport": 150,
}


@dataclass
class LatencyBudget:
    stages: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_BUDGET_MS))

    @property
    def total_ms(self) -> float:
        return sum(self.stages.values())

    def check(self, measured_p95: dict[str, float], target_ms: float = 1300) -> dict[str, object]:
        over = {k: round(v - self.stages[k], 1) for k, v in measured_p95.items()
                if k in self.stages and v > self.stages[k]}
        missing = sorted(set(self.stages) - set(measured_p95))
        total = sum(measured_p95.get(k, self.stages[k]) for k in self.stages)
        return {"total_p95_ms": round(total, 1), "within_target": total <= target_ms and not over,
                "over_budget": over, "unmeasured": missing}


__all__ = ["Segment", "TurnGate", "GateDecision", "PlaybackController", "LatencyBudget", "DEFAULT_BUDGET_MS"]
