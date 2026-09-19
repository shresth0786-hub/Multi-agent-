"""Routing helpers used by conditional edges in the orchestrator graph."""


def route_after_critique(state: dict) -> str:
    """Decide whether to loop back to the Writer or finalize.

    ``max_critique_iterations`` was already enforced in the critique node by
    force-approving the best draft, so reaching here with a report means done.
    """
    if state.get("approved") and state.get("report"):
        return "finalize"
    return "report"


def route_after_coverage(state: dict) -> str:
    """Adaptive research depth: another research round (with refined focus)
    while evidence is thin, otherwise move on to the analyst."""
    if state.get("status") == "needs_research" and state.get("research_focus"):
        return "research"
    return "analyze"
