from typing import Dict, Any
from state import ResearchState
from config import get_llm

def _extract_content(resp):
    c = resp.content if hasattr(resp, "content") else str(resp)
    if isinstance(c, list):
        return "\n".join([x.get("text","") if isinstance(x, dict) else str(x) for x in c])
    return c
from util import get_vectorstore

MAX_CONTEXT_CHARS = 3000

def critic_agent(state: ResearchState) -> Dict[str, Any]:
    comparison = state["comparison_matrix"]["comparison_summary"]
    query = state["query"]

    # Reuse singleton vectorstore; retriever is cheap
    vs = get_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": 4})
    try:
        context_docs = retriever.invoke(query)
    except Exception as e:
        # Fallback: use analysis_results as grounding if vectorstore fails
        print(f"[critic_agent] retriever failed: {e}, using analysis_results fallback")
        context_docs = []

    context_str = "\n\n".join([d.page_content for d in context_docs]) if context_docs else ""
    if not context_str:
        # Build grounding from analysis_results to avoid empty critique
        context_str = "\n\n".join([f"{a['title']}: {a['analysis'][:600]}" for a in state.get("analysis_results", [])])

    if len(context_str) > MAX_CONTEXT_CHARS:
        context_str = context_str[:MAX_CONTEXT_CHARS].rsplit(" ", 1)[0] + "..."
    if len(comparison) > 3000:
        comparison = comparison[:3000].rsplit(" ", 1)[0] + "..."

    prompt = f"""You are a rigorous peer reviewer. Evaluate the comparative analysis against the source abstracts.

--- Source Abstracts ---
{context_str}

--- Analysis to Critique ---
{comparison}

Task (max 300 words):
1. Identify unsupported claims.
2. Flag exaggerations or missing caveats.
3. Provide actionable corrections."""

    llm = get_llm()
    resp = llm.invoke(prompt)
    content = _extract_content(resp)
    return {"critique_findings": {"critique": content}}
