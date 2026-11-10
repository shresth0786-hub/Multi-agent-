"""Data Analyst agent: writes analysis code and executes it in the sandbox.

Features a self-heal loop: when the sandbox fails, the LLM is asked to repair
the script using the captured error; a deterministic fallback script is the
last safety net so the pipeline always produces results.
"""

from langchain_core.messages import HumanMessage, SystemMessage

from app.prompts.analyst import ANALYST_SYSTEM
from app.services.llm import get_chat_model
from app.services.sandbox import run_code

_MAX_ATTEMPTS = 3
_MAX_CODE_CHARS = 12_000

_FALLBACK_CODE = """\
try:
    import json

    raw = open("input.json", encoding="utf-8").read()
    data = json.loads(raw)
except Exception as e:
    raw = open("input.json", encoding="utf-8").read()
    data = {"note": "input was not JSON; fell back to raw text", "raw": raw[:2000]}

report = {}

def shape(v):
    if isinstance(v, dict):
        return {k: shape(x) for k, x in list(v.items())[:5]}
    if isinstance(v, list):
        return {"length": len(v), "sample": shape(v[:3]) if v else None}
    return v

if isinstance(data, dict):
    report["top_level_keys"] = list(data.keys())[:20]
    report["summary"] = {k: shape(v) for k, v in list(data.items())[:8]}
elif isinstance(data, list):
    report["records"] = len(data)
    report["sample"] = data[:5]
else:
    report["type"] = type(data).__name__
    report["characters"] = len(str(data))

with open("analysis_result.json", "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2, ensure_ascii=False)
"""


def _extract_code(text: str) -> str:
    if not text.strip():
        return text
    if "```" not in text:
        return text

    blocks = text.split("```")
    for block in blocks[1::2]:
        if block.startswith("python") or block.startswith("py"):
            block = block.split("\n", 1)[1] if "\n" in block else ""
        if block.strip():
            return block
    return text


def _generate_code(raw_data: str, user_request: str) -> str | None:
    model = get_chat_model()
    if model is None or not raw_data.strip():
        return None
    try:
        response = model.invoke(
            [
                SystemMessage(content=ANALYST_SYSTEM),
                HumanMessage(
                    content=(
                        f"Research question: {user_request}\n\n"
                        "Raw data (input.json in sandbox):\n```\n"
                        + raw_data[:8000]
                        + "\n```\n\nGenerate the Python analysis script now."
                    )
                ),
            ]
        )
        code = str(response.content).strip()
        if not code:
            return None
        return _extract_code(code)[: _MAX_CODE_CHARS]
    except Exception:  # noqa: BLE001
        return None


def _fix_code(code: str, raw_data: str, user_request: str, stderr: str) -> str | None:
    model = get_chat_model()
    if model is None:
        return None
    try:
        response = model.invoke(
            [
                SystemMessage(content=ANALYST_SYSTEM),
                HumanMessage(
                    content=(
                        f"Research question: {user_request}\n\n"
                        "Your previous script FAILED with the following error:\n"
                        + (stderr or "unknown error")[:3000]
                        + "\n\nPrevious script:\n```python\n"
                        + code[:6000]
                        + "\n```\n\n"
                        "Return ONLY a fixed Python script that avoids that error.\n"
                    )
                ),
            ]
        )
        fixed = str(response.content).strip()
        if not fixed:
            return None
        return _extract_code(fixed)[: _MAX_CODE_CHARS]
    except Exception:  # noqa: BLE001
        return None


def run_analysis(raw_data: str, user_request: str, code: str | None = None) -> dict:
    """Write, execute, and if needed repair analysis code; returns outputs."""
    candidates: list[str] = []
    if code and code.strip():
        candidates.append(code.strip())
    else:
        generated = _generate_code(raw_data, user_request)
        if generated:
            candidates.append(generated)
        candidates.append(_FALLBACK_CODE)

    last_result = None
    for attempt, analysis_code in enumerate(candidates):
        result = run_code(analysis_code, files={"input.json": raw_data})
        last_result = result
        if result["ok"]:
            return _pack(analysis_code, result)
        if attempt < _MAX_ATTEMPTS - 1:
            fixed = _fix_code(analysis_code, raw_data, user_request, result["stderr"] or "")
            if fixed and fixed != analysis_code:
                candidates.append(fixed)

    # All generated candidates failed; fallback is guaranteed-safe and last in line.
    if not last_result or not last_result["ok"]:
        fallback_result = run_code(_FALLBACK_CODE, files={"input.json": raw_data})
        return _pack(_FALLBACK_CODE, fallback_result)
    return _pack(candidates[-1], last_result)


def _pack(analysis_code: str, result: dict) -> dict:
    return {
        "analysis_code": analysis_code,
        "analysis_output": result["stdout"] or result["stderr"],
        "analysis_result": result["result"],
        "charts": result["charts"],
        "ok": result["ok"],
        "error": result["error"],
    }
