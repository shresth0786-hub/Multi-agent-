"""Train the agents on the local knowledge base and prove it works.

1. Checks which capabilities are configured (LLM / web search keys).
2. Indexes everything in data/raw_docs into the FAISS vectorstore.
3. Runs a probe query through the full pipeline and shows the result,
   including the retrieved evidence coverage.

You need an OPENAI_API_KEY in .env for the embedding step; a TAVILY_API_KEY
enables live web research on top of local knowledge.

Run:  python -m scripts.train
"""

from app.config import get_settings, usable_key
from app.services.runner import run_task
from app.services.vectorstore import build_index, load_documents, reset_index_cache

PROBE_REQUEST = (
    "Analyze our weekly-active-user retention for Q3 2026 and recommend three "
    "evidence-backed improvements, citing the supporting numbers."
)


def main() -> int:
    settings = get_settings()
    llm_ok = usable_key(settings.openai_api_key)
    web_ok = usable_key(settings.tavily_api_key) and settings.enable_web_search

    print("=" * 64)
    print("Agent training harness  (RAG knowledge base)")
    print("=" * 64)
    print(f"LLM configured:      {'yes' if llm_ok else 'no  - add OPENAI_API_KEY to .env'}")
    print(f"Web search:          {'yes' if web_ok else 'no  - add TAVILY_API_KEY to .env'}")
    docs = load_documents()
    print(f"Documents found:     {len(docs)}  in {settings.data_dir}")
    if not docs:
        print("Nothing to index. Put your knowledge base files (.md/.txt/.csv/.json/.html)")
        print("into data/raw_docs/ and re-run.")

    if not llm_ok:
        print("\n[!] Embeddings need a real OpenAI key (current value is a placeholder).")
        print("    In .env set:  OPENAI_API_KEY=sk-your-real-key")
        print("    Optionally:   TAVILY_API_KEY=tvly-your-real-key")
        print("    Then re-run:  python -m scripts.train")
        return 1

    print("\n[1] Indexing local knowledge base into FAISS...")
    reset_index_cache()
    try:
        index = build_index(force=True)
    except Exception as exc:  # noqa: BLE001 - surface a helpful message, not a traceback
        print(f"    Indexing failed: {exc}")
        return 1
    if index is None:
        print("    Indexing failed (empty corpus or embedding error).")
        return 1
    print(f"    Index ready: {index.index.ntotal} chunks in {settings.index_dir}")

    print("\n[2] Running probe query through the full pipeline...")
    response = run_task(PROBE_REQUEST)
    print(f"\nTask:       {response.task_id}")
    print(f"Status:     {response.status}  |  Approved: {response.approved}")
    print(f"Depth:      {response.research_depth}  |  Coverage: {response.coverage_score:.2f}")
    print(f"Sources:    {len(response.research_sources)} retrieved")
    print(f"Sub-tasks:  {len(response.subtasks)}")
    for sub in response.subtasks:
        print(f"  - {sub.get('title', sub)}")

    print("\nEvidence & coverage (per sub-task):")
    if not response.evidence:
        print("  (none recorded)")
    for e in response.evidence:
        flag = "OK" if e.get("sufficient") else "THIN"
        print(f"  [{flag}] {e.get('subtask')} score={e.get('score')} sources={e.get('sources')}")

    print("\n===== REPORT (first 2500 chars) =====")
    print((response.report or "")[:2500])

    out_dir = settings.data_dir.parent / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "train_report.md"
    path.write_text(response.report or "", encoding="utf-8")
    print(f"\nSaved report: {path}")
    print("\nDone. Ask the same question in the web UI (http://127.0.0.1:8000) and")
    print("the agents will answer from your indexed knowledge.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
