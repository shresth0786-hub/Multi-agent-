from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from state import ResearchState
from config import get_llm

def _extract_content(resp):
    c = resp.content if hasattr(resp, "content") else str(resp)
    if isinstance(c, list):
        return "\n".join([x.get("text","") if isinstance(x, dict) else str(x) for x in c])
    return c

MAX_WORKERS = 4  # parallelize I/O-bound LLM calls - ~3-4x speedup for 4 papers
MAX_ABSTRACT_CHARS = 1500  # truncate to save tokens
SYSTEM_PROMPT = "You are a precise academic assistant. Be concise and structured."

def _analyze_one(paper: Dict[str, Any]) -> Dict[str, Any]:
    llm = get_llm()
    summary = paper["summary"]
    if len(summary) > MAX_ABSTRACT_CHARS:
        summary = summary[:MAX_ABSTRACT_CHARS].rsplit(" ", 1)[0] + "..."
    prompt = f"""{SYSTEM_PROMPT}

Analyze this academic paper abstract:

Title: {paper['title']}
Abstract: {summary}

Extract in 150 words or less:
1. Primary Methodology / Technical Approach
2. Key Results and Empirical Findings
3. Stated Limitations"""
    # Single invoke per paper but run concurrently
    resp = llm.invoke(prompt)
    content = _extract_content(resp)
    return {"id": paper["id"], "title": paper["title"], "analysis": content}


def analysis_agent(state: ResearchState) -> Dict[str, Any]:
    papers = state["paper_metadata"]
    if not papers:
        return {"analysis_results": []}

    # Efficiency: single-paper fast path avoids thread overhead
    if len(papers) == 1:
        return {"analysis_results": [_analyze_one(papers[0])]}

    analysis_results: List[Dict[str, Any]] = []
    # ThreadPoolExecutor is optimal for network-bound OpenAI calls (vs ProcessPool)
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(papers))) as ex:
        fut_to_paper = {ex.submit(_analyze_one, p): p for p in papers}
        for fut in as_completed(fut_to_paper):
            try:
                analysis_results.append(fut.result())
            except Exception as e:
                p = fut_to_paper[fut]
                # Graceful degradation - don't fail entire batch for one paper
                analysis_results.append({"id": p["id"], "title": p["title"], "analysis": f"[ERROR analyzing paper: {e}]"})

    # Preserve original paper order
    order = {p["id"]: i for i, p in enumerate(papers)}
    analysis_results.sort(key=lambda x: order.get(x["id"], 999))
    return {"analysis_results": analysis_results}
