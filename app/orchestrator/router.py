"""Routing helpers used by conditional edges in the orchestrator graph."""


def route_after_critique(state: dict) -> str:
    """Decide whether to loop back to the Writer or finalize.

    ``max_critique_iterations`` was already enforced in the critique node by
    force-approving the best draft, so reaching here with a report means done.
    """
    if state.get("approved") and state.get("report"):
        return "finalize"
    return "report"
