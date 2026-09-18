"""Researcher agent: tool-calling agent over web search (Tavily) + FAISS.

Gives the agent real tool-calling powers via ``create_react_agent``. When no
Tavily/OpenAI keys are configured it degrades to a pure local-vectorstore
retrieval, so the pipeline still runs end-to-end for testing.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from app.prompts.researcher import RESEARCHER_SYSTEM
from app.services.llm import get_chat_model
from app.services.tools import build_tools
from app.services.vectorstore import search_local_documents

_MAX_FINDINGS_CHARS = 6000
_MAX_SOURCE_CHARS = 800


def _fallback_research(query: str) -> tuple[str, list[str]]:
    hits = search_local_documents(query)
    if not hits:
        return (
            "No web search or local documents were available (missing API keys "
            "and no indexed documents). Pipeline ran on configuration defaults.",
            [],
        )
    body = "\n\n---\n\n".join(hits)
    return body[: _MAX_FINDINGS_CHARS], [h[: _MAX_SOURCE_CHARS] for h in hits]


def _gather_tool_sources(messages: list) -> list[str]:
    sources: list[str] = []
    for msg in messages:
        content = getattr(msg, "content", "")
        if not content:
            continue
        if isinstance(msg, ToolMessage):
            text = str(content)[: _MAX_SOURCE_CHARS]
            sources.append(text)
        elif isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for call in msg.tool_calls:
                arg_text = str(call.get("args", "") or "")[: _MAX_SOURCE_CHARS / 4]
                sources.append(arg_text)
    return sources[:25]


def run_research(subtask: dict, user_request: str) -> dict:
    """Run the research step for one sub-task; returns findings + sources."""
    description = subtask.get("description") or subtask.get("title", "")
    query = f"{description}\nOriginal request: {user_request}"

    tools = build_tools()
    model = get_chat_model()
    if not tools or model is None:
        findings, sources = _fallback_research(query)
        return {"findings": findings, "sources": sources}

    agent = create_react_agent(model, tools, prompt=RESEARCHER_SYSTEM)
    try:
        result = agent.invoke(
            {
                "messages": [
                    SystemMessage(content=RESEARCHER_SYSTEM),
                    HumanMessage(content=query),
                ]
            }
        )
        final = result["messages"][-1].content
        findings = str(final)[: _MAX_FINDINGS_CHARS]
        sources = _gather_tool_sources(result["messages"])
        return {"findings": findings, "sources": sources}
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, never crash the graph
        fallback, fallback_sources = _fallback_research(query)
        return {
            "findings": f"{fallback}\n\n[research agent error: {exc}]",
            "sources": fallback_sources,
        }


def run_all_research(subtasks: list[dict], user_request: str) -> dict:
    """Run research across every planned sub-task, accumulating results."""
    if not subtasks:
        generic = {"title": "General research", "description": user_request}
        return run_research(generic, user_request)

    findings: list[str] = []
    sources: list[str] = []
    for subtask in subtasks:
        output = run_research(subtask, user_request)
        findings.append(f"### {subtask.get('title', 'research')}\n" + output["findings"])
        sources.extend(output["sources"])
    return {"findings": "\n\n".join(findings), "sources": sources}
