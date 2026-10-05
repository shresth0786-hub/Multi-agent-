import operator
from typing import Annotated, List, NotRequired, TypedDict
from uuid import uuid4

from langchain_core.messages import BaseMessage


class SubtaskDict(TypedDict, total=False):
    id: int
    title: str
    description: str
    goal: str


class ResearchState(TypedDict):
    """Shared state flowing through the orchestrator graph.

    List fields annotated with ``operator.add`` are automatically *merged*
    across nodes by LangGraph (repeat visits to the writer/critique loop
    accumulate drafts and feedback).
    """

    task_id: str
    user_request: str
    conversation: str
    subtask_index: int
    subtasks: List[dict]
    research_depth: int
    research_focus: List[dict]
    research_results: Annotated[List[str], operator.add]
    research_sources: Annotated[List[str], operator.add]
    round_evidence: NotRequired[List[dict]]
    evidence: Annotated[List[dict], operator.add]
    coverage_score: float
    raw_data: str
    analysis_code: str
    analysis_output: str
    analysis_result: NotRequired[object]
    charts: Annotated[List[str], operator.add]
    report_drafts: Annotated[List[str], operator.add]
    critique_feedback: Annotated[List[str], operator.add]
    iterations: int
    report: str
    approved: bool
    last_agent: str
    status: str
    error: str
    messages: NotRequired[List[BaseMessage]]


def initial_state(user_request: str, task_id: str | None = None, conversation: str = "") -> dict:
    return {
        "task_id": task_id or str(uuid4()),
        "user_request": user_request,
        "conversation": conversation,
        "subtask_index": 0,
        "subtasks": [],
        "research_depth": 0,
        "research_focus": [],
        "research_results": [],
        "research_sources": [],
        "evidence": [],
        "coverage_score": 0.0,
        "raw_data": "",
        "analysis_code": "",
        "analysis_output": "",
        "charts": [],
        "report_drafts": [],
        "critique_feedback": [],
        "iterations": 0,
        "report": "",
        "approved": False,
        "last_agent": "planner",
        "status": "planned",
        "error": "",
    }
