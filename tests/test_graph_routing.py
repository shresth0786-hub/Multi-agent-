from app.orchestrator.router import route_after_critique
from app.orchestrator.state import initial_state


def _state(**overrides):
    state = initial_state("test request")
    state.update(overrides)
    return state


def test_approved_routes_to_finalize():
    state = _state(approved=True, report="Final report body")
    assert route_after_critique(state) == "finalize"


def test_unapproved_routes_back_to_report():
    state = _state(approved=False, report="Draft needs work")
    assert route_after_critique(state) == "report"


def test_initial_state_shape():
    state = initial_state("hello")
    assert state["status"] == "planned"
    assert state["research_results"] == []
    assert state["report_drafts"] == []
    assert state["task_id"]
