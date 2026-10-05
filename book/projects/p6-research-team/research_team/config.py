# path: book/projects/p6-research-team/research_team/config.py
"""Environment-driven settings for the CLI and the container. Library classes take explicit
arguments; only entry points read the environment."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

from .contracts import BudgetSlice
from .ledger import TeamBudget
from .team import TeamConfig


class P6Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="P6_", env_file=".env", extra="ignore")

    offline: bool = True                   # scripted policy instead of the model from LLM_PROVIDER
    shared_data_dir: str | None = None
    log_dir: str = ".runs"
    max_tokens: int = 80_000
    max_cost_usd: float | None = None
    deadline_s: float = 120.0
    max_children: int = 6
    max_parallel: int = 4
    child_max_tokens: int = 12_000
    child_max_steps: int = 6
    max_rounds: int = 2

    def team_config(self) -> TeamConfig:
        child = BudgetSlice(max_steps=self.child_max_steps, max_tokens=self.child_max_tokens)
        return TeamConfig(budget=TeamBudget(max_tokens=self.max_tokens, max_cost_usd=self.max_cost_usd,
                                            deadline_s=self.deadline_s, max_children=self.max_children,
                                            max_parallel=self.max_parallel, child=child),
                          max_rounds=self.max_rounds)


__all__ = ["P6Settings"]
