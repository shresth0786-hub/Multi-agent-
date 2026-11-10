"""Writer agent: drafts the comprehensive research report."""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import get_settings
from app.prompts.writer import WRITER_SYSTEM
from app.services.llm import get_chat_model
from app.services.style_examples import style_prompt_section

logger = logging.getLogger(__name__)

_MAX_FALLBACK_FINDINGS = 12


def _short(exc: object, limit: int = 160) -> str:
    text = " ".join(str(exc).split())
    return text[:limit] + ("…" if len(text) > limit else "")


def _fallback_report(payload: dict) -> str:
    request = payload["user_request"]
    lines = [
        f"# Research Report: {request[:80]}",
        "",
        "## Executive Summary",
        "- Pipeline ran end-to-end; this draft was assembled by the fallback "
        "writer (no LLM key configured, or the model call failed).",
        "- Findings and analysis artifacts below are from the research/analysis stages.",
        "",
        "## Findings",
    ]
    research = payload.get("research_results") or []
    sources = payload.get("research_sources") or []
    if research:
        shown = research[:_MAX_FALLBACK_FINDINGS]
        for idx, item in enumerate(shown, 1):
            lines.append(f"### Finding {idx}")
            lines.append(str(item)[:1500])
        if len(research) > len(shown):
            lines.append(
                f"_{len(research) - len(shown)} further finding(s) omitted._"
            )
    else:
        lines.append("No research findings were produced.")

    lines.extend(["", "## Data Analysis"])
    analysis = payload.get("analysis_output") or ""
    analysis_result = payload.get("analysis_result")
    if analysis_result is not None:
        lines.append("```json")
        lines.append(_safe_json(analysis_result)[:2000])
        lines.append("```")
    elif analysis.strip():
        lines.append("```")
        lines.append(str(analysis)[:2000])
        lines.append("```")
    else:
        lines.append("No analysis output was produced.")
    lines.append("")
    if payload.get("charts"):
        count = len(payload["charts"])
        lines.append(f"Charts generated: {count} PNG artifact(s) embedded in response.")

    lines.extend(["", "## Conclusions"])
    lines.append(
        "The request was processed by the full multi-agent pipeline "
        "(plan -> research -> coverage -> analysis -> writer/critique). "
        "Add a valid OPENAI_API_KEY or GEMINI_API_KEY in Settings for "
        "LLM-written reports."
    )
    lines.extend(["", "## Evidence & Coverage"])
    evidence = payload.get("evidence") or []
    if evidence:
        lines.append(f"Coverage score: {float(payload.get('coverage_score', 0.0)):.2f}")
        lines.append("")
        lines.append("| Sub-task | Score | Sufficient | Sources |")
        lines.append("|---|---|---|---|")
        for e in evidence[:20]:
            lines.append(
                f"| {e.get('subtask', '?')} | {float(e.get('score', 0)):.2f} "
                f"| {'yes' if e.get('sufficient') else 'no'} | {e.get('sources', 0)} |"
            )
    else:
        lines.append("No per-sub-task evidence was recorded for this run.")
    lines.extend(["", "## References"])
    if sources:
        for src in sources[:25]:
            lines.append(f"- {str(src)[:300]}")
    else:
        lines.append("- No external sources were retrieved.")
    return "\n".join(lines)


def _safe_json(value) -> str:
    try:
        import json

        return json.dumps(value, indent=2, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        return str(value)


def write_report(payload: dict) -> str:
    """Produce a new draft (agents use latest state at time of call)."""
    settings = get_settings()
    model = get_chat_model()
    max_words = settings.report_max_words

    context = {
        "user_request": payload["user_request"],
        "conversation": payload.get("conversation") or "",
        "research_results": payload.get("research_results") or [],
        "research_sources": payload.get("research_sources") or [],
        "analysis_output": payload.get("analysis_output") or "",
        "analysis_result": payload.get("analysis_result"),
        "charts_count": len(payload.get("charts") or []),
        "evidence": payload.get("evidence") or [],
        "coverage_score": payload.get("coverage_score", 0.0),
    }
    if model is not None:
        conversation = context["conversation"]
        conversation_section = (
            conversation
            and "\n\nEarlier conversation in this chat (use for continuity "
            f"when relevant):\n{conversation}"
        ) or ""
        messages = [
            SystemMessage(content=WRITER_SYSTEM),
            HumanMessage(
                content=(
                    f"Original request:\n{payload['user_request']}\n"
                    f"{conversation_section}\n\n"
                    f"Word budget: {max_words} words.\n\n"
                    "Research findings:\n"
                    + "\n\n".join(str(r) for r in context["research_results"])
                    + "\n\nData analysis:\n"
                    + str(context["analysis_output"])
                    + "\n\nSources:\n"
                    + "\n".join(f"- {s}" for s in context["research_sources"])
                    + "\n\nEvidence coverage:\n"
                    + str(context["evidence"])
                    + style_prompt_section()
                )
            ),
        ]
        # Two attempts: rate-limited providers can reject a burst, and the
        # second try lands after the throttle's pacing delay.
        for attempt in (1, 2):
            try:
                draft = model.invoke(messages)
                text = str(draft.content or "").strip()
                if text:
                    return text
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 2:
                    logger.warning("Writer model call failed: %s", _short(exc))
                    break
    return _fallback_report(context)
