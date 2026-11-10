import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import time
import hashlib
from typing import Dict, Any

try:
    from langchain_core.documents import Document
except ImportError:
    # Fallback for environments without langchain_core (e.g. structure checks)
    from dataclasses import dataclass, field

    @dataclass
    class Document:
        page_content: str
        metadata: dict = field(default_factory=dict)

from state import ResearchState
from util import get_vectorstore

ARXIV_TIMEOUT = 15
ARXIV_RETRIES = 2
MAX_RESULTS = 4
USER_AGENT = "MultiAgentResearch/1.0 (mailto:shrwsthkumar@gmail.com)"

def _fetch_arxiv(query: str, max_results: int = MAX_RESULTS) -> bytes:
    encoded = urllib.parse.quote(query)
    url = f"https://export.arxiv.org/api/query?search_query=all:{encoded}&start=0&max_results={max_results}"
    last_err = None
    for attempt in range(ARXIV_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=ARXIV_TIMEOUT) as resp:
                return resp.read()
        except Exception as e:
            last_err = e
            if attempt < ARXIV_RETRIES:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"arXiv fetch failed after {ARXIV_RETRIES+1} tries: {last_err}") from last_err


def _sanitize_query(q: str, limit: int = 150) -> str:
    # arXiv 500 often from too-long / special-char query (problem adds many words)
    # Keep first ~25 words / 150 chars, strip quotes
    q = q.replace('"', " ").replace("'", " ").strip()
    if len(q) > limit:
        q = q[:limit].rsplit(" ", 1)[0]
    # Collapse whitespace
    return " ".join(q.split())


def research_agent(state: ResearchState) -> Dict[str, Any]:
    query = state["query"]
    problem = state.get("problem", "").strip()
    # Use problem keywords only (first 20 words) to avoid 500; don't send full paragraph to arXiv
    if problem:
        short_problem = " ".join(problem.split()[:20])
        search_query = _sanitize_query(f"{query} {short_problem}")
    else:
        search_query = _sanitize_query(query)
    try:
        xml_data = _fetch_arxiv(search_query)
    except Exception as e:
        # Graceful fallback: arXiv 500 is temporary - continue workflow with empty papers
        # so analysis/report still run (or fallback to local FAISS via app/ path)
        print(f"[research_agent] arXiv failed for '{search_query}': {e} - continuing with empty results")
        return {"paper_metadata": []}

    root = ET.fromstring(xml_data)
    ns = {"arxiv": "http://www.w3.org/2005/Atom"}
    papers, documents = [], []

    for entry in root.findall("arxiv:entry", ns):
        # Fix bug: .text is property not callable; also handle None entries (e.g. <feed> without entries)
        title_el = entry.find("arxiv:title", ns)
        summary_el = entry.find("arxiv:summary", ns)
        id_el = entry.find("arxiv:id", ns)
        if title_el is None or summary_el is None:
            continue
        title = (title_el.text or "").strip().replace("\n", " ")
        summary = (summary_el.text or "").strip().replace("\n", " ")
        paper_id = (id_el.text or "").strip() if id_el is not None and id_el.text else hashlib.md5(title.encode()).hexdigest()[:8]

        # Truncate summary to ~1200 chars to save embedding tokens (~300 tokens)
        if len(summary) > 1200:
            summary = summary[:1200].rsplit(" ", 1)[0] + "..."

        papers.append({"id": paper_id, "title": title, "summary": summary})
        documents.append(
            Document(page_content=f"Title: {title}\nAbstract: {summary}", metadata={"title": title, "id": paper_id})
        )

    if documents:
        # Use batch add; skip duplicates by checking existing ids
        vs = get_vectorstore()
        try:
            # Deduplicate via simple id check in metadata (cheap before embedding)
            existing = set()
            try:
                # Chroma get() is cheap and avoids re-embedding duplicates
                data = vs.get()
                for m in data.get("metadatas", []) or []:
                    if m and "id" in m:
                        existing.add(m["id"])
            except Exception:
                pass
            new_docs = [d for d in documents if d.metadata.get("id") not in existing]
            if new_docs:
                # add_documents batches internally; chunk_size handled by embeddings config
                vs.add_documents(new_docs)
        except Exception as e:
            # Don't fail workflow if vectorstore is down - papers are still returned
            print(f"[research_agent] vectorstore add failed (non-fatal): {e}")

    return {"paper_metadata": papers}
