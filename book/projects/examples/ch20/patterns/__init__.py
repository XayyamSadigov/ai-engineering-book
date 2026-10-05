# path: book/projects/examples/ch20/patterns/__init__.py
"""Chapter 20 agent architectures, each a composition over agentkit.AgentRuntime."""
from .common import Meter, PatternResult, ask, ask_structured, make_agent, role_of, role_prompt
from .evaluator_optimizer import Evaluation, EvaluatorOptimizer, agent_generator, checks_then_judge
from .hierarchical import Team, build_tree, run_hierarchy
from .parallel import Branch, fan_out
from .planner_executor import Plan, PlannerExecutor, PlanStep, StepOutcome, empty_evidence, step_failed, validate_plan
from .react import react
from .reflection import CriticCheck, Critique, reflective_agent
from .router import AgentRouter, Rule, Specialist
from .sequential import ChainContext, Step, StepFailed, agent_step, code_step, llm_step, run_chain
from .supervisor import SpawnBudget, Supervisor, Worker

__all__ = [
    "Meter", "PatternResult", "ask", "ask_structured", "make_agent", "role_of", "role_prompt",
    "react", "AgentRouter", "Rule", "Specialist", "Plan", "PlanStep", "PlannerExecutor", "StepOutcome",
    "step_failed", "empty_evidence", "validate_plan", "Supervisor", "Worker", "SpawnBudget", "Team", "build_tree",
    "run_hierarchy", "CriticCheck", "Critique", "reflective_agent", "Evaluation", "EvaluatorOptimizer",
    "agent_generator", "checks_then_judge", "Branch", "fan_out", "ChainContext", "Step", "StepFailed",
    "agent_step", "code_step", "llm_step", "run_chain",
]
