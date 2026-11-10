from typing import Dict, Any
from state import ResearchState
from config import get_llm

def _extract_content(resp):
    c = resp.content if hasattr(resp, "content") else str(resp)
    if isinstance(c, list):
        return "\n".join([x.get("text","") if isinstance(x, dict) else str(x) for x in c])
    return c

MAX_CHARS_PER_ANALYSIS = 1200  # truncate to stay within context window

def comparison_agent(state: ResearchState) -> Dict[str, Any]:
    analyses = state["analysis_results"]
    if not analyses:
        return {"comparison_matrix": {"comparison_summary": "No analyses to compare."}}

    truncated = []
    for a in analyses:
        txt = a["analysis"]
        if len(txt) > MAX_CHARS_PER_ANALYSIS:
            txt = txt[:MAX_CHARS_PER_ANALYSIS].rsplit(" ", 1)[0] + "..."
        truncated.append(f"### {a['title']}\n{txt}")

    formatted = "\n\n".join(truncated)

    prompt = f"""Synthesize a comparative analysis (max 400 words):

Papers:
{formatted}

Address:
- Shared architectural or algorithmic paradigms
- Key differences in methodology
- Relative strengths and trade-offs

Be concise, bullet-point where possible."""

    llm = get_llm()
    resp = llm.invoke(prompt)
    content = _extract_content(resp)
    return {"comparison_matrix": {"comparison_summary": content}}
