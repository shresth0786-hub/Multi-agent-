"""The LangGraph supervisor: plans, routes, and persists state across agents."""

from functools import lru_cache

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.orchestrator.nodes import (
    analyze_node,
    coverage_node,
    critique_node,
    finalize_node,
    plan_node,
    report_node,
    research_node,
)
from app.orchestrator.router import route_after_coverage, route_after_critique
from app.orchestrator.state import ResearchState


def build_graph(checkpointer=None):
    """Compile the orchestrator graph:

        START -> plan -> research -> coverage --+-> analyze -> report -> critique -> finalize -> END
                                          ^-----|                ^-+(revision loop)-+--| (capped)

    Two adaptive loops:
      1. Coverage Scout  - research -> coverage -> research (deeper) until evidence is sufficient
      2. Writer/Critique - report -> critique -> report (revision) until quality threshold

    Pass a persistent checkpointer (e.g. ``SqliteSaver``) to make node state
    survive restarts; defaults to an in-memory ``MemorySaver``.
    """
    graph = StateGraph(ResearchState)

    graph.add_node("plan", plan_node)
    graph.add_node("research", research_node)
    graph.add_node("coverage", coverage_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("report", report_node)
    graph.add_node("critique", critique_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "research")
    graph.add_edge("research", "coverage")
    graph.add_conditional_edges(
        "coverage",
        route_after_coverage,
        {"research": "research", "analyze": "analyze"},
    )
    graph.add_edge("analyze", "report")
    graph.add_edge("report", "critique")
    graph.add_conditional_edges(
        "critique",
        route_after_critique,
        {"finalize": "finalize", "report": "report"},
    )
    graph.add_edge("finalize", END)

    checkpointer = checkpointer or MemorySaver()
    return graph.compile(checkpointer=checkpointer)


@lru_cache(maxsize=1)
def get_graph():
    return build_graph()
