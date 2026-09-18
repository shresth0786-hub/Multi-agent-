import pytest

from app.orchestrator.planner import fallback_plan, plan
from app.services.llm import get_chat_model


@pytest.fixture(autouse=True)
def no_api_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("TAVILY_API_KEY", "")
    from app.config import get_settings

    get_settings.cache_clear()


def test_fallback_plan_is_deterministic():
    plan_a = fallback_plan("Analyze Q2 revenue")
    plan_b = fallback_plan("Analyze Q2 revenue")
    assert len(plan_a) == 3
    assert plan_a == plan_b


def test_plan_without_llm_returns_valid_subtasks(monkeypatch):
    monkeypatch.setattr(get_chat_model, "__call__", lambda *a, **k: None)
    monkeypatch.setattr("app.orchestrator.planner.get_chat_model", lambda *a, **k: None)

    result = plan("Compare our retention metrics to industry benchmarks", llm=None)
    assert result, "plan should never be empty"
    assert all({"id", "title", "description", "goal"} <= set(item) for item in result)


def test_plan_via_function_uses_fallback():
    result = plan("Some request", llm=None)
    assert all({"id", "title", "description", "goal"} <= set(item) for item in result)
