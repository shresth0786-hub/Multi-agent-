"""Chat UX backend: conversation memory, quick follow-up answers, SSE stream."""

import json
import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def no_api_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("TAVILY_API_KEY", "")
    from app.config import get_settings

    get_settings.cache_clear()


@pytest.fixture(scope="module")
def client():
    from app.main import app

    return TestClient(app)


def _seed(session_id: str, turns: list[tuple[str, str]]) -> None:
    from app.services.task_store import get_task_store

    store = get_task_store()
    store.clear_conversation(session_id)
    for i, (role, content) in enumerate(turns):
        store.append_conversation(session_id, f"t{i}", role, content)


def _sse_events(resp) -> list[dict]:
    events = []
    for line in resp.iter_lines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: ") :]))
    return events


def test_conversation_roundtrip_and_prune():
    from app.services.task_store import get_task_store

    store = get_task_store()
    store.clear_conversation("test-conv")
    for i in range(12):
        store.append_conversation("test-conv", f"t{i}", "user", f"msg {i}", max_turns=8)
    history = store.get_conversation("test-conv")
    assert [m["content"] for m in history] == [f"msg {i}" for i in range(4, 12)]
    assert all(m["role"] == "user" for m in history)
    store.clear_conversation("test-conv")


def test_first_message_runs_full_pipeline_even_with_quick(client):
    resp = client.post(
        "/api/v1/agent/run",
        json={"user_request": "Summarize retention notes.", "quick": True},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["quick"] is False
    assert body["research_depth"] >= 1


def test_quick_followup_skips_pipeline(client):
    session_id = "test-quick-followup"
    _seed(session_id, [("user", "What were Q3 numbers?"), ("assistant", "Revenue up 12%.")])
    resp = client.post(
        "/api/v1/agent/run",
        json={"user_request": "And retention?", "session_id": session_id, "quick": True},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["quick"] is True
    assert body["status"] == "completed"
    assert "retention" in body["report"].lower() or body["report"]

    from app.services.task_store import get_task_store

    turns = get_task_store().get_conversation(session_id)
    assert turns[-1]["role"] == "assistant"
    get_task_store().clear_conversation(session_id)


def test_stream_endpoint_emits_status_chunk_and_answer(client):
    session_id = "test-stream"
    _seed(session_id, [("user", "Q3 metrics?"), ("assistant", "Revenue up 12%.")])
    resp = client.post(
        "/api/v1/agent/stream",
        json={"user_request": "What about retention?", "session_id": session_id, "quick": True},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = _sse_events(resp)
    types = [e["type"] for e in events]
    assert types[0] == "status"
    assert "chunk" in types
    assert types[-1] == "done"
    answer = next(e for e in events if e["type"] == "answer")
    assert answer["data"]["quick"] is True
    assert answer["data"]["report"]

    from app.services.task_store import get_task_store

    get_task_store().clear_conversation(session_id)


def test_settings_endpoint_masks_keys_and_applies(client):
    resp = client.get("/api/v1/settings")
    assert resp.status_code == 200
    assert resp.json()["provider"] == "openai"
    assert resp.json()["openai_api_key"] == ""

    saved = client.post(
        "/api/v1/settings",
        json={"llm_provider": "gemini", "gemini_api_key": "AQ.fakekey1234567890abcd"},
    )
    assert saved.status_code == 200
    body = saved.json()
    assert body["provider"] == "gemini"
    assert "fakekey" not in body["gemini_api_key"]
    assert body["gemini_api_key"].startswith("AQ.fa")
    assert body["llm_configured"] is True

    health = client.get("/api/v1/health").json()
    assert health["provider"] == "gemini"
    assert health["llm_configured"] is True


def test_stream_full_pipeline_reports_stage_events(client):
    t0 = time.time()
    resp = client.post(
        "/api/v1/agent/stream",
        json={"user_request": "Give a short Q3 health summary."},
    )
    events = _sse_events(resp)
    done = time.time() - t0
    assert events[-1]["type"] == "done"

    statuses = [e.get("text", "") for e in events if e["type"] == "status"]
    assert any("Planning" in s for s in statuses), statuses
    assert any("Researching" in s for s in statuses), statuses
    assert any("Writing the report" in s for s in statuses), statuses

    answer = next(e for e in events if e["type"] == "answer")
    assert answer["data"]["status"] == "completed"
    assert answer["data"]["report"]
    assert done < 120

    # Streamed runs must also land in task history so they can be reopened.
    task_id = answer["data"]["task_id"]
    listed = client.get("/api/v1/tasks").json()
    assert any(t["task_id"] == task_id for t in listed)
    detail = client.get(f"/api/v1/tasks/{task_id}").json()
    assert detail["status"] == "completed"
    assert detail["report"] == answer["data"]["report"]


def test_gemini_chat_model_is_rate_throttled(monkeypatch):
    """Gemini free tier is 5 req/min, so every request must be paced."""
    from app.services import llm

    calls = []
    monkeypatch.setattr(llm, "throttle", lambda: calls.append("throttled"))

    class FakeModel:
        def invoke(self, messages):
            return "invoked"

        def stream(self, messages):
            return iter(["token"])

    wrapped = llm._ThrottledChatModel(FakeModel())
    assert wrapped.invoke("m") == "invoked"
    assert list(wrapped.stream("m")) == ["token"]
    assert calls == ["throttled", "throttled"]


def test_gemini_research_uses_grounded_call_not_tools(monkeypatch):
    """Gemini 3.x rejects function calls without thought signatures, so the
    researcher must ground in the prompt instead of running a ReAct tool loop."""
    from app.agents import researcher
    from app.agents.researcher import run_research

    seen: dict = {}

    class FakeModel:
        def invoke(self, messages):
            seen["prompt"] = "\n".join(str(m.content) for m in messages)
            return type("Reply", (), {"content": "Grounded findings [1]."})()

    monkeypatch.setattr(researcher, "get_chat_model", lambda *a, **k: FakeModel())
    monkeypatch.setattr(researcher, "active_provider", lambda: "gemini")
    monkeypatch.setattr(
        researcher,
        "_grounded_context",
        lambda q: ("### Local document excerpts\nWAU 84,200", ["WAU 84,200"]),
    )

    def _no_react(*args, **kwargs):
        raise AssertionError("ReAct tool loop must not run on Gemini")

    monkeypatch.setattr(researcher, "create_react_agent", _no_react)

    out = run_research({"title": "Metrics", "description": "Q3 metrics"}, "how are we doing")
    assert out["findings"] == "Grounded findings [1]."
    assert out["sources"] == ["WAU 84,200"]
    assert "WAU 84,200" in seen["prompt"]
