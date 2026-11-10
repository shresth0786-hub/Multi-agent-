import time
import os
import sys
from graph import create_research_graph

if __name__ == "__main__":
    # Support: python main.py "your query"  OR  interactive prompt  OR  default
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        problem = ""
    else:
        try:
            q = input("Enter research query [press Enter for default: Retrieval Augmented Generation with Graph Neural Networks]: ").strip()
            query = q if q else "Retrieval Augmented Generation with Graph Neural Networks"
            problem = input("Describe your specific problem/context (optional, press Enter to skip): ").strip()
        except EOFError:
            query = "Retrieval Augmented Generation with Graph Neural Networks"
            problem = ""

    # Warn only if no provider key at all
    if not any(os.getenv(k) for k in ["GROQ_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"]) and not os.getenv("OLLAMA_BASE_URL"):
        print("WARNING: No LLM key found (GROQ_API_KEY/GOOGLE_API_KEY/OPENAI_API_KEY). Workflow will fail at first LLM call.")

    app = create_research_graph()

    initial_state = {
        "query": query,
        "problem": problem,
        "paper_metadata": [],
        "analysis_results": [],
        "comparison_matrix": {},
        "critique_findings": {},
        "final_report": "",
    }

    print(f"\nExecuting Multi-Agent Workflow for query: '{query}'\n")
    t0 = time.time()
    final_output = app.invoke(initial_state, config={"recursion_limit": 20})
    elapsed = time.time() - t0

    print(f"\n=== FINAL RESEARCH REPORT (completed in {elapsed:.1f}s) ===\n")
    print(final_output["final_report"])
