"""Planner: decomposes the user request into sub-tasks (LLM-first, fallback)."""

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.prompts.planner import PLANNER_SYSTEM
from app.services.llm import get_chat_model


class SubtaskItem(BaseModel):
    id: int
    title: str
    description: str = ""
    goal: str = ""


class SubtaskSet(BaseModel):
    subtasks: list[SubtaskItem] = Field(default_factory=list)


def fallback_plan(user_request: str) -> list[dict]:
    """Deterministic 3-step plan so the pipeline runs with zero credentials."""
    return [
        {
            "id": 0,
            "title": "Gather raw information",
            "description": (
                f"Collect facts, figures, named sources, and datasets relevant to: {user_request}"
            ),
            "goal": "Raw research findings with citations",
        },
        {
            "id": 1,
            "title": "Analyze the findings",
            "description": (
                f"Run statistical analysis over the gathered data to answer: {user_request}"
            ),
            "goal": "Statistical findings and charts",
        },
        {
            "id": 2,
            "title": "Synthesize the report",
            "description": (
                f"Write a comprehensive report answering: {user_request}"
            ),
            "goal": "Final report",
        },
    ]


def _conversation_context(conversation: str | None) -> str:
    text = (conversation or "").strip()
    return text or None


def plan(user_request: str, llm=None, conversation: str | None = None) -> list[dict]:
    """Decompose a request into sub-tasks; guaranteed to return a valid plan."""
    request = (user_request or "").strip()
    llm = llm if llm is not None else get_chat_model()
    context = _conversation_context(conversation)

    if llm is not None and request:
        try:
            structured = llm.with_structured_output(SubtaskSet)
            message = request
            if context:
                message = (
                    f"Earlier conversation (reference only):\n{context}\n\n"
                    f"Plan sub-tasks for the latest request:\n{request}"
                )
            result = structured.invoke(
                [
                    SystemMessage(content=PLANNER_SYSTEM),
                    HumanMessage(content=message),
                ]
            )
            subtasks: list[dict] = []
            for item in (result.subtasks or [])[:5]:
                if item.title.strip():
                    subtasks.append(item.model_dump())
            if subtasks:
                return subtasks
        except Exception:  # noqa: BLE001
            pass

    return fallback_plan(request or "General research request")
