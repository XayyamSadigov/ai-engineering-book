# path: book/projects/examples/ch04/prompts/rollout.py
"""Serve the right prompt version at runtime: sticky canary splits and safe alias reloads.

The registry answers "what is ticket.classify@canary?". A running service needs two more
answers. Which requests get the canary? A stable hash of (prompt id, unit key) puts each
user or ticket in one arm for the whole rollout, so one conversation never flips between
versions and the arms can be compared. What happens when someone pushes a broken
aliases.toml or prompt file? The new registry is built and checked off to the side, and it
replaces the serving one only if it loads and matches the lock. Otherwise the service keeps
the last known good registry and reports the failure. A bad prompt push should page
someone; it should never take the feature down.
"""
from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from .registry import PromptRegistry, PromptVersion


def bucket(prompt_id: str, unit_key: str) -> float:
    """A stable position in [0, 100) for this unit. Salting with the prompt id keeps the
    same users from being the canary population for every prompt at once."""
    digest = hashlib.sha256(f"{prompt_id}:{unit_key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 * 100


@dataclass(frozen=True)
class ReloadResult:
    ok: bool
    error: str | None = None


class PromptRollout:
    """Holds the serving registry and picks prod or canary per request."""

    def __init__(
        self,
        loader: Callable[[], PromptRegistry],
        *,
        canary_percent: Mapping[str, float] | None = None,
        lock: Mapping[str, str] | None = None,
    ) -> None:
        self._loader = loader
        self._lock_entries = dict(lock) if lock is not None else None
        self.canary_percent = dict(canary_percent or {})
        self._mutex = threading.Lock()
        # Startup is fail-fast: a service that cannot load its prompts must not start.
        self._registry = self._build()
        self.loaded_at = time.time()
        self.reload_failures = 0
        self.last_error: str | None = None

    def _build(self) -> PromptRegistry:
        registry = self._loader()
        if self._lock_entries is not None:
            problems = registry.verify_lock(self._lock_entries)
            if problems:
                raise ValueError("; ".join(problems))
        return registry

    @property
    def registry(self) -> PromptRegistry:
        return self._registry

    def reload(self) -> ReloadResult:
        """Swap in a freshly loaded registry, or keep the current one if loading fails."""
        try:
            fresh = self._build()
        except Exception as exc:  # any load failure is a reason to keep serving the old state
            with self._mutex:
                self.reload_failures += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
            return ReloadResult(ok=False, error=self.last_error)
        with self._mutex:
            self._registry = fresh
            self.loaded_at = time.time()
            self.last_error = None
        return ReloadResult(ok=True)

    def arm(self, prompt_id: str, unit_key: str) -> str:
        """`canary` or `prod` for this unit. No canary alias, or 0%, means everyone gets prod."""
        aliases = self._registry.aliases.get(prompt_id, {})
        percent = self.canary_percent.get(prompt_id, 0.0)
        if "canary" in aliases and bucket(prompt_id, unit_key) < percent:
            return "canary"
        return "prod"

    def select(self, prompt_id: str, unit_key: str) -> PromptVersion:
        return self._registry.get(prompt_id, self.arm(prompt_id, unit_key))


__all__ = ["bucket", "ReloadResult", "PromptRollout"]
