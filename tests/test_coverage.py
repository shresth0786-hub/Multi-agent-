from app.orchestrator.nodes import _coverage_score
from app.orchestrator.router import route_after_coverage
from app.orchestrator.state import initial_state


def test_coverage_scoring_rich_findings_pass_threshold():
    score = _coverage_score(
        "Revenue grew 34% in FY2025 to $24M (source: annual report, 2026). "
        "Pipeline summary continues with several more detailed quantified "
        "observations about margins, headcount, and expansion plans.",
        ["https://example.com/annual-report", "https://example.com/press"],
    )
    assert score >= 0.6


def test_coverage_scoring_thin_findings_fail_threshold():
    score = _coverage_score("No relevant info found.", [])
    assert score < 0.6


def test_route_after_coverage_goes_deeper_when_gaps():
    state = initial_state("test")
    state["status"] = "needs_research"
    state["research_focus"] = [{"id": 0, "title": "deepen", "description": "go deeper"}]
    assert route_after_coverage(state) == "research"


def test_route_after_coverage_proceeds_when_saturated():
    state = initial_state("test")
    state["status"] = "researched"
    state["research_focus"] = []
    assert route_after_coverage(state) == "analyze"
