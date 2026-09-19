"""Runs the LangGraph pipeline for one task and marshals its final state
into the API response shape. Shared by the sync route and the worker queue.
"""

from uuid import uuid4

from app.api.schemas import AgentRunResponse
from app.config import get_settings
from app.orchestrator.graph import build_graph
from app.orchestrator.state import initial_state
from app.services.checkpoint import sqlite_checkpointer


def state_to_response(state: dict) -> AgentRunResponse:
    return AgentRunResponse(
        task_id=state.get("task_id", ""),
        status=state.get("status", ""),
        report=state.get("report", "") or (state.get("report_drafts") or [""])[-1],
        report_drafts=state.get("report_drafts", []),
        critique_feedback=state.get("critique_feedback", []),
        approved=bool(state.get("approved")),
        iterations=state.get("iterations", 0),
        subtasks=state.get("subtasks", []),
        research_depth=state.get("research_depth", 0),
        coverage_score=state.get("coverage_score", 0.0),
        evidence=state.get("evidence", []),
        research_sources=state.get("research_sources", []),
        analysis_code=state.get("analysis_code", ""),
        analysis_output=state.get("analysis_output", ""),
        analysis_result=state.get("analysis_result"),
        charts=state.get("charts", []),
        error=state.get("error", ""),
    )


def run_task(
    user_request: str, metadata: dict | None = None, task_id: str | None = None
) -> AgentRunResponse:
    """Execute the full graph for *task_id* (or a new one) on the SQLite checkpointer."""
    settings = get_settings()
    task_id = task_id or str(uuid4())

    with sqlite_checkpointer() as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        final = graph.invoke(
            initial_state(user_request, task_id),
            config={
                "configurable": {"thread_id": task_id, "metadata": metadata or {}},
                "recursive_limit": 1
                + 2 * (settings.max_critique_iterations + 6)
                + 3 * (settings.max_research_depth + 2),
            },
        )
    return state_to_response(final)
