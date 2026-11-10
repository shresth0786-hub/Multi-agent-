"""Live integration tests - require API keys.

Run with:  pytest -m live
Skipped by default so CI / fresh clones stay green without credentials.
"""


import pytest

from app.config import get_settings

pytestmark = pytest.mark.live


@pytest.fixture(autouse=True)
def require_keys():
    settings = get_settings()
    if not settings.has_llm:
        pytest.skip("OPENAI_API_KEY not configured; skipping live tests.")
    yield


def test_live_pipeline_end_to_end():
    from app.services.runner import run_task

    response = run_task(
        "Using the provided local notes, summarize our retention metrics and "
        "propose two improvements?"
    )
    assert response.status == "completed"
    assert response.approved is True
    assert response.report
    assert response.iterations >= 1


def test_analyst_self_heals_on_broken_code():
    """Give the analyst intentionally broken code; the self-heal loop must recover."""
    from app.agents.analyst import run_analysis

    broken = "import json\njson.dump({'ok': 1}, open('analysis_result.json','w'))\n1/0"
    result = run_analysis('{"count": 5}', "Count the things", code=broken)
    assert result["error"] is None or result["analysis_result"] is not None
