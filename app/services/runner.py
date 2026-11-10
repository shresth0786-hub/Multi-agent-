"""Runs the LangGraph pipeline for one task and marshals its final state
into the API response shape. Shared by the sync route, the worker queue,
and the SSE streaming route.

Also implements "quick" chat answers (multi-turn follow-ups that skip the
full pipeline when the session already has context).
"""

import logging
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage

from app.api.schemas import AgentRunResponse
from app.config import get_settings
from app.orchestrator.graph import build_graph
from app.orchestrator.state import initial_state
from app.services.checkpoint import sqlite_checkpointer
from app.services.progress import set_active

logger = logging.getLogger(__name__)


def conversation_context(history: list[dict] | None) -> str:
    turns = history or []
    if not turns:
        return ""
    limit = get_settings().max_conversation_turns
    lines = []
    for turn in turns[-limit:]:
        role = "user" if turn.get("role") == "user" else "assistant"
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if role == "assistant":
            content = content[:1500]
        lines.append(f"{role}: {content}")
    return "\n---\n".join(lines)


def _quick_prompt(request: str, context: str) -> HumanMessage:
    settings = get_settings()
    if context:
        return HumanMessage(
            content=(
                "Below is our earlier conversation in this chat. Respond to the "
                "LATEST request concisely, as a direct follow-up. Ground your "
                "answer in what we discussed; do not repeat the earlier report "
                "verbatim.\n\n"
                f"Earlier conversation:\n{context}\n\n"
                f"Latest request:\n{request}\n\n"
                f"Answer in under {settings.quick_answer_max_words} words."
            )
        )
    return HumanMessage(
        content=(
            f"Answer the request below concisely and directly, in under "
            f"{settings.quick_answer_max_words} words.\n\nRequest:\n{request}"
        )
    )


def _quick_fallback(request: str, context: str) -> str:
    """No-LLM fallback: point the user at the session's last report."""
    last_assistant = ""
    for turn in reversed(context.split("\n---\n")):
        if turn.startswith("assistant:"):
            last_assistant = turn[len("assistant:") :].strip()
            break
    if last_assistant:
        body = last_assistant[:400].replace("\n", " ")
        return (
            "# Quick answer (fallback mode)\n\n"
            "Continuing our conversation, here is a snapshot from the most recent "
            "report in this chat:\n\n"
            f"> {body}\n\n"
            "Configure `OPENAI_API_KEY`/`GEMINI_API_KEY` plus `TAVILY_API_KEY` in "
            "`.env` to get a live, context-aware answer for follow-ups."
        )
    return (
        "# Quick answer (fallback mode)\n\n"
        "No LLM credentials are configured yet, so follow-ups can't be answered "
        "live. Add `OPENAI_API_KEY` (or `GEMINI_API_KEY`) and `TAVILY_API_KEY` to "
        "`.env` and restart the server for live multi-turn answers."
    )


def _streamed_text(model, messages):
    """Yield ('ok', token) for streamed tokens; yield ('error', msg) on failure."""
    try:
        for chunk in model.stream(messages):
            piece = getattr(chunk, "content", None)
            if piece:
                yield ("ok", str(piece))
    except Exception as exc:  # noqa: BLE001
        yield ("error", f"{type(exc).__name__}: {exc}")


def _quick_error(message: str) -> str:
    head = str(message)[:200]
    return (
        "# Quick answer unavailable\n\n"
        f"The model errored while answering (often a temporary rate limit): "
        f"{head}\n\nRetry the message in a minute, or send a full 'deep' "
        "question to re-run the pipeline."
    )


def quick_answer_generator(
    user_request: str, context: str, task_id: str | None = None
):
    """Yield quick-answer pieces for live streaming. Falls back on one block."""
    from app.services.llm import get_chat_model

    model = get_chat_model()
    yield {"type": "status", "text": "Drafting a quick answer…"}
    messages = [
        SystemMessage(
            content=(
                "You are a concise, knowledgeable assistant in a research chat. "
                "Answer directly, no filler."
            )
        ),
        _quick_prompt(user_request, context),
    ]
    if model is not None:
        for attempt in (1, 2):
            ok_pieces = 0
            failure = None
            for kind, payload in _streamed_text(model, messages):
                if kind == "error":
                    failure = payload
                    break
                ok_pieces += 1
                yield {"type": "chunk", "text": payload}
            if ok_pieces:
                return
            if failure is None:
                break
            if attempt == 2:
                logger.warning(
                    "Quick answer stream failed for task=%s: %s", task_id, failure
                )
                yield {"type": "chunk", "text": _quick_error(failure)}
                return
    yield {"type": "chunk", "text": _quick_fallback(user_request, context)}


def run_quick(
    user_request: str,
    history: list[dict] | None = None,
    context: str | None = None,
    task_id: str | None = None,
) -> AgentRunResponse:
    """Produce a short follow-up answer; falls back to a summary without keys."""
    context = context if context is not None else conversation_context(history)
    from app.services.llm import get_chat_model

    model = get_chat_model()
    if model is not None:
        try:
            messages = [
                SystemMessage(
                    content=(
                        "You are a concise, knowledgeable assistant in a research "
                        "chat. Answer directly, no filler."
                    )
                ),
                _quick_prompt(user_request, context),
            ]
            answer = str(model.invoke(messages).content or "").strip()
            if answer:
                return AgentRunResponse(
                    task_id=task_id or str(uuid4()),
                    status="completed",
                    quick=True,
                    report=answer,
                )
        except Exception:  # noqa: BLE001
            pass
    return AgentRunResponse(
        task_id=task_id or str(uuid4()),
        status="completed",
        quick=True,
        report=_quick_fallback(user_request, context),
    )


def state_to_response(state: dict, quick: bool = False) -> AgentRunResponse:
    return AgentRunResponse(
        task_id=state.get("task_id", ""),
        status=state.get("status", ""),
        quick=quick,
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
    user_request: str,
    metadata: dict | None = None,
    task_id: str | None = None,
    session_id: str = "",
    history: list[dict] | None = None,
    quick: bool = False,
    on_event=None,
) -> AgentRunResponse:
    """Execute the full graph on the SQLite checkpointer (or quick answer)."""
    settings = get_settings()
    task_id = task_id or str(uuid4())
    history = history if history is not None else _load_history(session_id)
    context = conversation_context(history)

    from app.services.task_store import get_task_store

    store = get_task_store()
    store.ensure(task_id, user_request, metadata)
    store.set_status(task_id, "running")

    def note(**event) -> None:
        if on_event is not None:
            on_event(**event)

    note(type="status", text="Starting the research pipeline…")

    # First message -> full pipeline; follow-ups with quick=true stay short.
    if quick and history:
        response = run_quick(user_request, context=context, task_id=task_id)
        persist_conversation(session_id, task_id, user_request, response.report)
        store.store_result(task_id, response.model_dump(), status="completed")
        note(type="status", text="Quick answer ready.")
        return response

    try:
        with sqlite_checkpointer() as checkpointer:
            graph = build_graph(checkpointer=checkpointer)
            set_active(task_id)
            try:
                final = graph.invoke(
                    initial_state(user_request, task_id, conversation=context),
                    config={
                        "configurable": {
                            "thread_id": task_id,
                            "metadata": metadata or {},
                        },
                        "recursive_limit": 1
                        + 2 * (settings.max_critique_iterations + 6)
                        + 3 * (settings.max_research_depth + 2),
                    },
                )
            finally:
                set_active(None)
    except Exception as exc:  # noqa: BLE001
        store.set_status(task_id, "failed", error=str(exc))
        raise
    if on_event is not None:
        note(type="status", text="Answer ready.")
    response = state_to_response(final, quick=False)
    store.store_result(task_id, response.model_dump(), status=response.status or "completed")
    persist_conversation(session_id, task_id, user_request, response.report or "")
    return response


def _load_history(session_id: str) -> list[dict]:
    if not session_id:
        return []
    from app.services.task_store import get_task_store

    return get_task_store().get_conversation(session_id)


def persist_conversation(
    session_id: str, task_id: str, user_text: str, assistant_text: str
) -> None:
    if not session_id:
        return
    from app.services.task_store import get_task_store

    store = get_task_store()
    store.append_conversation(session_id, task_id, "user", user_text)
    store.append_conversation(session_id, task_id, "assistant", assistant_text)
