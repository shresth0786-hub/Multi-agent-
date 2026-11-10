try:
    from langgraph.graph import StateGraph, END

    _HAS_LANGGRAPH = True
except ImportError:
    _HAS_LANGGRAPH = False
    StateGraph = None
    END = "__end__"

from state import ResearchState
from agents.research import research_agent
from agents.analysis import analysis_agent
from agents.comparison import comparison_agent
from agents.critic import critic_agent
from agents.report import report_agent


def _fallback_graph():
    """Simple sequential executor when langgraph not installed - same efficiency, no dependency."""

    class FallbackApp:
        def invoke(self, state, config=None):
            state = research_agent(state)
            # merge returned dict into state
            for k, v in research_agent(state).items():
                pass  # placeholder
            # Actually run sequentially correctly:
            s = state
            for fn in [research_agent, analysis_agent, comparison_agent, critic_agent, report_agent]:
                out = fn(s)
                s = {**s, **out}
            return s

        def stream(self, *a, **kw):
            yield self.invoke(*a, **kw)

    # Fix duplicate call bug above - redo cleanly
    class _App:
        def invoke(self, state, config=None):
            s = dict(state)
            for fn in [research_agent, analysis_agent, comparison_agent, critic_agent, report_agent]:
                out = fn(s)
                s.update(out)
            return s

    return _App()


def create_research_graph():
    if not _HAS_LANGGRAPH:
        # No langgraph installed - use lightweight fallback (pip install langgraph for visualization/persistence)
        return _fallback_graph()

    workflow = StateGraph(ResearchState)
    workflow.add_node("research", research_agent)
    workflow.add_node("analysis", analysis_agent)
    workflow.add_node("comparison", comparison_agent)
    workflow.add_node("critic", critic_agent)
    workflow.add_node("report", report_agent)
    workflow.set_entry_point("research")
    workflow.add_edge("research", "analysis")
    workflow.add_edge("analysis", "comparison")
    workflow.add_edge("comparison", "critic")
    workflow.add_edge("critic", "report")
    workflow.add_edge("report", END)
    return workflow.compile()
