"""Graph nodes: one function per stage of the pipeline."""

from app.agents.analyst import run_analysis
from app.agents.critiquer import critique_report
from app.agents.researcher import run_all_research
from app.agents.writer import write_report
from app.config import get_settings
from app.orchestrator.planner import plan


def plan_node(state: dict) -> dict:
    subtasks = plan(state["user_request"])
    return {"subtasks": subtasks, "status": "planned", "last_agent": "planner"}


def research_node(state: dict) -> dict:
    output = run_all_research(state.get("subtasks") or [], state["user_request"])
    raw_data = "\n\n".join(output["findings"])
    return {
        "research_results": output["findings"].split("\n\n"),
        "research_sources": output["sources"],
        "raw_data": raw_data,
        "status": "researched",
        "last_agent": "researcher",
    }


def analyze_node(state: dict) -> dict:
    result = run_analysis(state.get("raw_data", ""), state["user_request"])
    return {
        "analysis_code": result["analysis_code"],
        "analysis_output": result["analysis_output"],
        "analysis_result": result["analysis_result"],
        "charts": result["charts"],
        "status": "analyzed",
        "last_agent": "analyst",
    }


def report_node(state: dict) -> dict:
    draft = write_report(
        {
            "user_request": state["user_request"],
            "research_results": state.get("research_results") or [],
            "research_sources": state.get("research_sources") or [],
            "analysis_output": state.get("analysis_output") or "",
            "analysis_result": state.get("analysis_result"),
            "charts": state.get("charts") or [],
        }
    )
    return {
        "report": draft,
        "report_drafts": [draft],
        "iterations": state.get("iterations", 0) + 1,
        "status": "drafting",
        "last_agent": "writer",
    }


def critique_node(state: dict) -> dict:
    settings = get_settings()
    verdict = critique_report(state.get("report", ""), state["user_request"])
    iterations = state.get("iterations", 0)
    feedback = verdict["feedback"] or []

    max_reached = iterations >= settings.max_critique_iterations
    approved = bool(verdict["passed"]) or max_reached

    return {
        "approved": approved,
        "critique_feedback": [f"Score {verdict['score']:.2f}: " + fb for fb in feedback[:6]],
        "status": "approved" if approved else "needs_revision",
        "last_agent": "critiquer",
    }


def finalize_node(state: dict) -> dict:
    return {
        "report": state.get("report", ""),
        "status": "completed",
        "last_agent": "orchestrator",
    }
