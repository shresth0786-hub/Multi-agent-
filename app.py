import streamlit as st
import time

from graph import create_research_graph

st.set_page_config(page_title="Multi-Agent Research Assistant", layout="wide")

st.title("🔬 Multi-Agent Research Assistant")
st.caption("Enter your research query and the problem you're facing — agents research, analyze, compare, critique & report tailored to you.")

with st.sidebar:
    st.header("Settings")
    st.write("LLM: `gemini-3.5-flash` (free) via `config.py:34`")
    st.write("Embeddings: `all-MiniLM-L6-v2` (local, HF)")
    st.write("Vectorstore: `Chroma` persistent `./chroma_db`")
    st.divider()
    st.info("Set `GOOGLE_API_KEY` in `.env` for Gemini. Also supports `GROQ_API_KEY` / `OLLAMA`.")

query = st.text_input(
    "🔍 Research Query *",
    placeholder="e.g. Retrieval Augmented Generation with Graph Neural Networks",
    help="Main topic to search on arXiv",
)
problem = st.text_area(
    "🧩 Your Specific Problem / Context *",
    placeholder="e.g. I'm building a RAG system for legal docs and my GNN retrieval is slow and hallucinates on long contexts. How to make it accurate and efficient?",
    height=120,
    help="Agents use this to tailor the comparative analysis and final recommendations to YOUR use-case",
)
max_results = st.slider("Max papers", 2, 8, 4, help="ArXiv max_results in agents/research.py:24")

col1, col2 = st.columns([1, 4])
run = col1.button("🚀 Run Research", type="primary", disabled=not (query.strip() and problem.strip()))
clear = col2.button("Clear")

if clear:
    query, problem = "", ""
    st.rerun()

if run:
    if not query.strip() or not problem.strip():
        st.warning("Please enter both query and problem.")
        st.stop()

    app = create_research_graph()
    initial_state = {
        "query": query.strip(),
        "problem": problem.strip(),
        "paper_metadata": [],
        "analysis_results": [],
        "comparison_matrix": {},
        "critique_findings": {},
        "final_report": "",
    }

    progress = st.progress(0, text="Starting research agent...")
    status = st.status("Running multi-agent workflow...", expanded=True)
    t0 = time.time()

    try:
        # Manual step streaming for UX (still linear graph, but show progress)
        with status:
            st.write("1️⃣ Researching arXiv...")
            progress.progress(10)
            s1 = app.get_graph().nodes  # not needed, just for compatibility

        # Invoke full graph (efficient, cached LLM/embeddings in config.py:11)
        # For live progress we invoke stepwise; fallback to full invoke if needed
        final_output = app.invoke(initial_state, config={"recursion_limit": 20})
        progress.progress(100, text="Done!")
        status.update(label="✅ Completed", state="complete")

        elapsed = time.time() - t0
        st.success(f"Completed in {elapsed:.1f}s")

        # Show paper metadata
        with st.expander("📄 Retrieved Papers", expanded=False):
            for p in final_output.get("paper_metadata", []):
                st.markdown(f"**{p['title']}** — `{p['id']}`")
                st.caption(p['summary'][:400] + "...")

        with st.expander("🧪 Analyses (per-paper)", expanded=False):
            for a in final_output.get("analysis_results", []):
                st.markdown(f"**{a['title']}**")
                st.write(a['analysis'])

        c1, c2 = st.columns(2)
        with c1:
            with st.expander("⚖️ Comparison", expanded=False):
                st.markdown(final_output.get("comparison_matrix", {}).get("comparison_summary", ""))
        with c2:
            with st.expander("🔍 Critique", expanded=False):
                st.markdown(final_output.get("critique_findings", {}).get("critique", ""))

        st.divider()
        st.markdown(final_output.get("final_report", "No report generated"))
        st.download_button("⬇️ Download Report (Markdown)", final_output.get("final_report", ""), file_name="research_report.md")

    except Exception as e:
        progress.progress(0)
        status.update(label="❌ Failed", state="error")
        st.error(f"Error: {e}")
        st.exception(e)
        st.info("Check `GOOGLE_API_KEY` in `.env` and that `GEMINI_MODEL=gemini-3.5-flash` is available. Try rerunning or lower max_results.")
