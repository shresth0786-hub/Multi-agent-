"""Graph nodes: one function per stage of the pipeline."""

import re

from app.agents.analyst import run_analysis
from app.agents.critiquer import critique_report
from app.agents.researcher import run_all_research
from app.agents.writer import write_report
from app.config import get_settings
from app.orchestrator.planner import plan

_SUFFICIENCY_MARKERS = ("http", "source", "reference", "dataset")


def plan_node(state: dict) -> dict:
    subtasks = plan(state["user_request"])
    return {"subtasks": subtasks, "status": "planned", "last_agent": "planner"}


def research_node(state: dict) -> dict:
    """Run one research round. After the first pass the Coverage Scout feeds
    ``research_focus`` so later rounds only deepen the weak sub-tasks."""
    depth = state.get("research_depth", 0) + 1
    focus = state.get("research_focus") or []
    if focus:
        subtasks = [dict(s) for s in focus]
    else:
        subtasks = [dict(s) for s in (state.get("subtasks") or [])]

    output = run_all_research(subtasks, state["user_request"])

    raw = state.get("raw_data", "") or ""
    if raw and output["findings"]:
        combined = f"{raw}\n\n{output['findings']}"
    else:
        combined = output["findings"]

    return {
        "research_results": output["findings"].split("\n\n"),
        "research_sources": output["sources"],
        "round_evidence": output["per_subtask"],
        "raw_data": combined,
        "research_depth": depth,
        "status": "researched",
        "last_agent": "researcher",
    }


def _coverage_score(findings: str, sources: list[str]) -> float:
    """Heuristic evidence sufficiency score for one sub-task (0..1).

    Rewards concrete, quantified, sourced material over filler.
    """
    text = findings or ""
    score = 0.0
    if len(text) >= 300:
        score += 0.35
    elif len(text) >= 80:
        score += 0.2
    if re.search(r"\d", text):
        score += 0.30
    if sources:
        score += 0.25
    elif any(marker in text.lower() for marker in _SUFFICIENCY_MARKERS):
        score += 0.15
    return round(min(score, 1.0), 2)


def coverage_node(state: dict) -> dict:
    """Coverage Scout: scores evidence gathered per sub-task and, when the
    research is too thin (no numbers, no sources), routes back into a deeper
    research round with a sharper focus. This is the adaptive depth loop."""
    settings = get_settings()
    threshold = settings.research_coverage_threshold
    depth = state.get("research_depth", 0)
    subtasks = state.get("subtasks") or []

    scored = []
    for entry in state.get("round_evidence") or []:
        title = entry.get("title") or "general"
        score = _coverage_score(entry.get("findings", ""), entry.get("sources") or [])
        scored.append(
            {
                "subtask": title,
                "score": score,
                "sufficient": score >= threshold,
                "sources": len(entry.get("sources") or []),
                "characters": len(entry.get("findings", "") or ""),
            }
        )

    sufficient_titles = {
        e["subtask"] for e in scored if e["sufficient"]
    } | {
        e["subtask"]
        for e in (state.get("evidence") or [])
        if e.get("sufficient")
    }

    gaps = [s for s in subtasks if s.get("title") not in sufficient_titles]
    coverage_score = round((len(subtasks) - len(gaps)) / max(len(subtasks), 1), 2)

    if gaps and depth < settings.max_research_depth:
        focus = [
            {
                "id": s.get("id", 0),
                "title": f"{s.get('title', 'research')} (deepen)",
                "description": (
                    f"{s.get('description', '')} GO DEEPER: obtain concrete, "
                    "quantified facts plus at least one named, citable source."
                ),
            }
            for s in gaps
        ]
        return {
            "evidence": scored,
            "coverage_score": coverage_score,
            "research_focus": focus,
            "status": "needs_research",
            "last_agent": "coverage_scout",
        }

    return {
        "evidence": scored,
        "coverage_score": coverage_score,
        "research_focus": [],
        "status": "researched",
        "last_agent": "coverage_scout",
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
            "evidence": state.get("evidence") or [],
            "coverage_score": state.get("coverage_score", 0.0),
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
