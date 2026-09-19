"""End-to-end demo with LIVE LLM credentials.

Requires OPENAI_API_KEY (and optionally TAVILY_API_KEY + an indexed corpus).
Run: python -m scripts.run_demo
"""

import base64
from pathlib import Path

from app.config import get_settings
from app.services.runner import run_task

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "generated"

DEFAULT_REQUEST = (
    "Analyze our product's weekly-active-user retention and recommend three "
    "evidence-backed improvements, citing sources."
)


def main() -> int:
    settings = get_settings()
    if not settings.has_llm:
        print("OPENAI_API_KEY is not set - this demo runs the fallback path.")
        print("Add keys in .env to exercise the live LLM agents.\n")

    request = input(f"Request [{DEFAULT_REQUEST}]: ").strip() or DEFAULT_REQUEST

    print("\nRunning full pipeline (plan -> research -> analyze -> report -> critique)...")
    response = run_task(request)

    print(f"\nTask:      {response.task_id}")
    print(f"Status:    {response.status}")
    print(f"Iters:     {response.iterations}  |  Approved: {response.approved}")
    print(f"Sub-tasks: {len(response.subtasks)}")
    print(f"Research depth: {response.research_depth}  |  Coverage: {response.coverage_score:.2f}")
    print(f"Research sources: {len(response.research_sources)}")
    for sub in response.subtasks:
        print(f"  - {sub.get('title', sub)}")

    print("\nEvidence & coverage (per sub-task):")
    for e in response.evidence:
        flag = "OK" if e.get("sufficient") else "THIN"
        print(f"  [{flag}] {e.get('subtask')} score={e.get('score')} sources={e.get('sources')}")

    print("\nCritique feedback:")
    for fb in response.critique_feedback:
        print(f"  ! {fb}")

    if response.analysis_code:
        print("\n===== ANALYSIS CODE =====")
        print(response.analysis_code[:2000])
    if response.analysis_output:
        print("\n===== ANALYSIS OUTPUT =====")
        print(response.analysis_output[:1500])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for idx, chart in enumerate(response.charts):
        path = OUTPUT_DIR / f"chart_{idx}.png"
        path.write_bytes(base64.b64decode(chart))
        print(f"Saved chart: {path}")

    report_path = OUTPUT_DIR / "report.md"
    report_path.write_text(response.report or "(empty report)", encoding="utf-8")
    print(f"Saved report: {report_path}")

    print("\n===== REPORT (first 1500 chars) =====")
    print((response.report or "")[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
