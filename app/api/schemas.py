from typing import Any

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    user_request: str = Field(min_length=1, description="The complex request to solve")
    metadata: dict[str, Any] = Field(default_factory=dict)
    session_id: str = Field(
        default="", description="Conversation id; enables memory across messages"
    )
    quick: bool = Field(
        default=False,
        description="Short direct answer for follow-ups; pipeline is skipped "
        "when the session already has context",
    )


class WebhookRequest(BaseModel):
    user_request: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskAcceptedResponse(BaseModel):
    task_id: str
    status: str = "queued"
    detail: str = "task accepted; poll GET /tasks/{task_id} for completion"


class AnalyzeRequest(BaseModel):
    code: str = Field(min_length=1, description="Python code to run in the sandbox")


class AgentRunResponse(BaseModel):
    task_id: str
    status: str
    quick: bool = False
    report: str = ""
    report_drafts: list[str] = Field(default_factory=list)
    critique_feedback: list[str] = Field(default_factory=list)
    approved: bool = False
    iterations: int = 0
    subtasks: list[dict] = Field(default_factory=list)
    research_depth: int = 0
    coverage_score: float = 0.0
    evidence: list[dict] = Field(default_factory=list)
    research_sources: list[str] = Field(default_factory=list)
    analysis_code: str = ""
    analysis_output: str = ""
    analysis_result: Any = None
    charts: list[str] = Field(default_factory=list, description="Base64-encoded PNGs")
    error: str = ""


class AnalyzeResponse(BaseModel):
    ok: bool
    stdout: str = ""
    stderr: str = ""
    result: Any = None
    charts: list[str] = Field(default_factory=list)
    error: str | None = None
