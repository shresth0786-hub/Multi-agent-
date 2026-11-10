"""Critiquer agent: scores drafts against quality guidelines.

Returns verdict {score, passed, feedback}. On failure the orchestrator routes
back to the Writer with the feedback for a revision pass.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.prompts.critiquer import CRITIQUE_SYSTEM
from app.services.llm import get_chat_model


class CritiqueVerdict(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    feedback: list[str] = Field(default_factory=list)


def _heuristic_verdict(report: str, user_request: str) -> dict:
    """Deterministic scoring when no LLM is configured (keeps the loop testable)."""
    score = 0.0
    feedback: list[str] = []

    sections = ["executive summary", "findings", "data analysis", "conclusions", "references"]
    lower = report.lower()
    found = [s for s in sections if s in lower]
    score += 0.5 * len(found) / len(sections)
    if len(found) < len(sections):
        feedback.append("Ensure all five report sections are present.")
    else:
        feedback.append("All required sections present.")

    words = len(report.split())
    if words >= 100:
        score += 0.2
    else:
        feedback.append("Report is too short; expand each section.")
    if "- http" in lower or "http" in lower or "references" in lower:
        score += 0.15
    else:
        feedback.append("Add a references section with numbered sources.")
    if user_request.lower().strip() in lower:
        score += 0.15
    else:
        feedback.append("Tie the report back to the original request explicitly.")

    score = round(min(score, 1.0), 2)
    passed = score >= 0.85
    if passed:
        feedback = ["Draft meets quality threshold."]
    return {"score": score, "passed": passed, "feedback": feedback[:6]}


def critique_report(report: str, user_request: str) -> dict:
    """Evaluate a draft; returns {score, passed, feedback}."""
    model = get_chat_model()
    if model is not None:
        try:
            structured = model.with_structured_output(CritiqueVerdict)
            verdict = structured.invoke(
                [
                    SystemMessage(content=CRITIQUE_SYSTEM),
                    HumanMessage(
                        content=f"Original request:\n{user_request}\n\nReport draft:\n{report}"
                    ),
                ]
            )
            return {
                "score": verdict.score,
                "passed": verdict.passed,
                "feedback": verdict.feedback or [],
            }
        except Exception:  # noqa: BLE001
            pass
    return _heuristic_verdict(report, user_request)
