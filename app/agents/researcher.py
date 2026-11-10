"""Researcher agent: gathers evidence for one sub-task.

Two strategies:
- OpenAI (and any tool-capable provider): a ReAct tool-calling agent over web
  search (Tavily) + the local FAISS store via ``create_react_agent``.
- Gemini: a single grounded call with web/local context supplied directly.
  Gemini 3.x rejects function-call parts that arrive without a thought
  signature, so a ReAct tool loop fails there; grounding in the prompt avoids
  tool calls entirely.

Either way it degrades to pure local-vectorstore retrieval when no keys are
configured, so the pipeline still runs end-to-end for testing.
"""

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from app.prompts.researcher import RESEARCHER_SYSTEM
from app.services.llm import active_provider, get_chat_model
from app.services.tools import build_tools, build_web_search_tool
from app.services.vectorstore import search_local_documents

logger = logging.getLogger(__name__)

_MAX_FINDINGS_CHARS = 6000
_MAX_SOURCE_CHARS = 800


def _short(exc: object, limit: int = 160) -> str:
    text = " ".join(str(exc).split())
    return text[:limit] + ("…" if len(text) > limit else "")


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


def _grounded_context(query: str) -> tuple[str, list[str]]:
    """Gather web + local evidence up front so no tool calls are needed."""
    parts: list[str] = []
    sources: list[str] = []
    web = build_web_search_tool()
    if web is not None:
        try:
            results = str(web.invoke({"query": query}) or "").strip()
            if results:
                parts.append(
                    "### Web search results\n" + results[: _MAX_FINDINGS_CHARS // 2]
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Web search failed for %r: %s", query, _short(exc))
    local = search_local_documents(query)
    if local:
        parts.append(
            "### Local document excerpts\n" + "\n\n---\n\n".join(local)[:_MAX_FINDINGS_CHARS]
        )
        sources.extend(h[:_MAX_SOURCE_CHARS] for h in local)
    return "\n\n".join(parts), sources


def _single_shot_research(query: str, model, context: str) -> str:
    prompt = (
        f"{query}\n\n"
        "Grounding context below comes from web search and our internal "
        "documents. Answer from it, cite the file or section names you used, "
        "and flag anything the context does not cover.\n\n"
        f"{context}"
    )
    reply = model.invoke(
        [SystemMessage(content=RESEARCHER_SYSTEM), HumanMessage(content=prompt)]
    )
    return str(getattr(reply, "content", "") or "").strip()[:_MAX_FINDINGS_CHARS]


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

    model = get_chat_model()
    if model is None:
        findings, sources = _fallback_research(query)
        return {"findings": findings, "sources": sources}

    if active_provider() == "gemini":
        context, sources = _grounded_context(query)
        if not context:
            findings, sources = _fallback_research(query)
            return {"findings": findings, "sources": sources}
        try:
            for attempt in (1, 2):
                try:
                    findings = _single_shot_research(query, model, context)
                    break
                except Exception as exc:  # noqa: BLE001
                    if attempt == 2:
                        raise
                    logger.info(
                        "Research retrying after %s for %r", _short(exc), query[:60]
                    )
            if findings:
                return {"findings": findings, "sources": sources}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Grounded research failed for %r: %s", query, _short(exc))
            fallback, fallback_sources = _fallback_research(query)
            return {
                "findings": (
                    f"{fallback}\n\n[research fell back to local retrieval: "
                    f"{_short(exc)}]"
                ),
                "sources": fallback_sources or sources,
            }

    tools = build_tools()
    if not tools:
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
        logger.warning("Research agent error for %r: %s", query, _short(exc))
        fallback, fallback_sources = _fallback_research(query)
        return {
            "findings": f"{fallback}\n\n[research agent error: {_short(exc)}]",
            "sources": fallback_sources,
        }


def run_all_research(subtasks: list[dict], user_request: str) -> dict:
    """Run research across the given sub-tasks, accumulating results.

    Returns findings (joined markdown), sources, and ``per_subtask`` evidence
    (title, findings, sources) used by the Coverage Scout node.
    """
    if not subtasks:
        generic = {"title": "General research", "description": user_request}
        output = run_research(generic, user_request)
        return {
            "findings": output["findings"],
            "sources": output["sources"],
            "per_subtask": [
                {
                    "title": generic["title"],
                    "findings": output["findings"],
                    "sources": output["sources"],
                }
            ],
        }

    findings: list[str] = []
    sources: list[str] = []
    per_subtask: list[dict] = []
    for subtask in subtasks:
        output = run_research(subtask, user_request)
        findings.append(f"### {subtask.get('title', 'research')}\n" + output["findings"])
        sources.extend(output["sources"])
        per_subtask.append(
            {
                "title": subtask.get("title", "research"),
                "findings": output["findings"],
                "sources": output["sources"],
            }
        )
    return {"findings": "\n\n".join(findings), "sources": sources, "per_subtask": per_subtask}
