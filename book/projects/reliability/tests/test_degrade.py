# path: book/projects/reliability/tests/test_degrade.py
from aie_core import CompletionRequest, Message

from reliability import DegradeLevel, DegradePolicy


def test_normal_when_healthy():
    plan = DegradePolicy().resolve(0, set())
    assert plan.level is DegradeLevel.NORMAL and plan.rerank and plan.reasons == []


def test_admission_level_selects_plan():
    plan = DegradePolicy().resolve(2, set())
    assert plan.level is DegradeLevel.MINIMAL and plan.model_tier == "small" and not plan.allow_agents


def test_one_provider_open_reduces_load():
    plan = DegradePolicy().resolve(0, {"llm:primary"})
    assert plan.level is DegradeLevel.REDUCED and "open:llm:primary" in plan.reasons


def test_all_models_open_goes_static():
    plan = DegradePolicy().resolve(0, {"llm:primary", "llm:backup"})
    assert plan.level is DegradeLevel.STATIC and not plan.use_model and not plan.allows_tool(False)


def test_component_breakers_adjust_plan():
    plan = DegradePolicy().resolve(0, {"rerank", "retrieval"})
    assert plan.level is DegradeLevel.NORMAL and not plan.rerank and plan.retrieval_k == 0


def test_read_only_blocks_side_effects_but_not_answers():
    policy = DegradePolicy()
    policy.read_only = True
    plan = policy.resolve(0, set())
    assert plan.allows_tool(side_effecting=False)
    assert not plan.allows_tool(side_effecting=True)
    assert "read_only" in plan.reasons


def test_apply_stamps_request():
    plan = DegradePolicy().resolve(2, set())
    req = CompletionRequest(messages=[Message.user("q")], max_tokens=1024)
    out = plan.apply(req, {"primary": "assist-large", "small": "assist-small"})
    assert out.model == "assist-small" and out.max_tokens == 384
    assert out.metadata["degrade_level"] == 2
