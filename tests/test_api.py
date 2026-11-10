import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def no_api_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("TAVILY_API_KEY", "")
    from app.config import get_settings

    get_settings.cache_clear()


@pytest.fixture(scope="module")
def client():
    from app.main import app

    return TestClient(app)


def _poll_until_terminal(client, task_id, timeout=90):
    deadline = time.time() + timeout
    body = None
    while time.time() < deadline:
        resp = client.get(f"/api/v1/tasks/{task_id}")
        assert resp.status_code == 200
        body = resp.json()
        if body["status"] in {"completed", "failed"}:
            return body
        time.sleep(0.25)
    status = body and body["status"]
    pytest.fail(f"Task {task_id} did not finish within {timeout}s (last status: {status})")


def test_health(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["llm_configured"] is False


def test_analyze_sandbox_endpoint(client):
    code = (
        "import json\n"
        "with open('analysis_result.json', 'w') as fh:\n"
        "    json.dump({'answer': 42}, fh)"
    )
    resp = client.post("/api/v1/analyze", json={"code": code})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["result"]["answer"] == 42


def test_webhook_is_async_and_task_is_durable(client):
    resp = client.post(
        "/api/v1/webhook/agent",
        json={"user_request": "Summarize the company retention metrics from local notes."},
    )
    assert resp.status_code == 202
    task_id = resp.json()["task_id"]
    assert resp.json()["status"] in {"queued", "running"}

    final = _poll_until_terminal(client, task_id)
    assert final["task_id"] == task_id
    assert final["status"] == "completed"
    assert final["approved"] is True
    assert final["report"]
    assert final["iterations"] >= 1
    assert 0.0 <= final["coverage_score"] <= 1.0
    assert isinstance(final["evidence"], list)
    assert final["research_depth"] >= 1


def test_sync_run_returns_full_result(client):
    resp = client.post(
        "/api/v1/agent/run",
        json={"user_request": "Give a quick health summary of our Q3 numbers."},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["approved"] is True
    assert body["task_id"]


def test_task_listing_and_lookup(client):
    tasks = client.get("/api/v1/tasks").json()
    assert isinstance(tasks, list)
    assert all({"task_id", "status"} <= set(t) for t in tasks)

    created = client.post(
        "/api/v1/agent/run", json={"user_request": "Confirm the pipeline works."}
    ).json()
    fetched = client.get(f"/api/v1/tasks/{created['task_id']}")
    assert fetched.status_code == 200
    assert fetched.json()["task_id"] == created["task_id"]

    assert client.get("/api/v1/tasks/does-not-exist").status_code == 404
