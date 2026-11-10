from typing import Dict, Any
from state import ResearchState
from config import get_llm

def _extract_content(resp):
    c = resp.content if hasattr(resp, "content") else str(resp)
    if isinstance(c, list):
        return "\n".join([x.get("text","") if isinstance(x, dict) else str(x) for x in c])
    return c

def _truncate(txt: str, limit: int) -> str:
    if len(txt) <= limit:
        return txt
    return txt[:limit].rsplit(" ", 1)[0] + "..."

def report_agent(state: ResearchState) -> Dict[str, Any]:
    query = state["query"]
    problem = state.get("problem", "")
    analyses = state.get("analysis_results", [])
    comparison = state.get("comparison_matrix", {}).get("comparison_summary", "")
    critique = state.get("critique_findings", {}).get("critique", "")

    analyses_str = "\n".join([f"- {a['title']}: {_truncate(a['analysis'], 500)}" for a in analyses])
    comparison = _truncate(comparison, 1500)
    critique = _truncate(critique, 1200)
    problem_block = f"\nUser's Specific Problem/Context:\n{problem}\n\nTailor recommendations and open directions to solve this problem." if problem else ""

    prompt = f"""Generate a comprehensive technical research report in Markdown (max 1000 words):

Topic Query: {query}
{problem_block}
Individual Analyses:
{analyses_str}
Comparative Synthesis: {comparison}
Peer Review Critique: {critique}

Structure:
# Research Report: {query}
## Executive Summary (3-4 sentences, mention how findings relate to user's problem if provided)
## Paper-by-Paper Breakdown (Methodology & Findings)
## Comparative Analysis & Trade-offs
## Critical Audit & Claim Grounding
## Conclusion & Open Directions (provide actionable next steps for user's problem)

Be concise, avoid repetition, cite papers by title. If a problem is provided, explicitly address it."""

    llm = get_llm()
    resp = llm.invoke(prompt)
    content = _extract_content(resp)
    return {"final_report": content}
