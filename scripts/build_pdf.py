"""Builds the PDF overview of the multi-agent system (repo root level).

Run:  python -m scripts.build_pdf
Output: Multi-Agent-System-Overview.pdf in the project root.
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "Multi-Agent-System-Overview.pdf"

TITLE = "Multi-Agent Autonomous Research &amp; Execution System"
SUBTITLE = (
    "Feature reference, functions, and pipeline documentation "
    "- generated from the codebase (LangGraph + FastAPI + Docker + AWS + n8n)"
)

SYSTEM_MAP = """n8n (email trigger)
     |
     |  HTTP POST  /api/v1/agent/run (sync)  |  /api/v1/webhook/agent (async)
     v
FastAPI bridge  ---->  Task Queue ---->  LangGraph Orchestrator
                           |                   |  plan -> research -> coverage -> analyze
                    SQLite task store          |  report <-> critique -> finalize
                           |___________________|

PIPELINE (one run of the graph):

START -> plan -> research -> coverage -> analyze -> report -> critique -> finalize -> END
                       ^        |  thin                             ^         |  fail
                       +-- GO DEEPER ---+                          +- revise --+"""

SEMANTIC_FLOW = """POST /webhook/agent {user_request}        -> 202 {task_id}   (submitted)
   -> worker pool picks it up                -> status "running"
   -> run_task(): graph + SQLite checkpointer -> invoke()
   -> result written to SQLite               -> status "completed"/"failed"
   -> n8n polls GET /tasks/{id} until done   -> reads report + charts"""

DATA_FLOW = """raw docs -> chunk (1200 chars / 200 overlap) -> embed -> FAISS index
   -> runtime: query -> embedding -> top-k similar chunks -> analyst/report"""

RESEARCH_FLOW = """sub-task description -> react agent decides:
   -> web_search(query)...................-> Tavily returns articles + URLs
   -> retrieve_local_documents(query)....-> FAISS top-k chunks
-> findings synthesis + sources list     -> flow into coverage scoring"""

ANALYSIS_FLOW = """raw_data -> LLM writes analysis.py
   -> run in sandbox (python -I, timeout, isolated)
   -> OK? ......-> return JSON result + base64 charts
   -> FAILED?...-> error fed to LLM -> rewrite + retry (<=3)
   -> all failed?-> deterministic fallback script summarizes input"""

WRITER_CRITIQUE_FLOW = """draft_1 -> critiquer score 0.4 (FAIL)  -> feedback appended
draft_2 (revised w/ feedback) -> 0.7 (FAIL)   -> feedback appended
draft_3 -> 0.9 (PASS) -> final report"""

SECTIONS = [
    (
        "1. System map",
        [
            ("pre", SYSTEM_MAP),
            "One run of the orchestrator is a stateful, event-driven walk over the graph on the left.",
        ],
    ),
    (
        "2. Feature reference (what each component does)",
        [
            ("h2", "A. Orchestrator - the supervisor brain"),
            ("table3", [
                ("State graph", "app/orchestrator/graph.py - build_graph()",
                 "Declares node wiring; LangGraph runs nodes, passes state between them, and loops via conditional edges."),
                ("Shared state", "app/orchestrator/state.py - ResearchState",
                 "One dict that lives across the whole run: user_request, subtasks, research_results, evidence, analysis_code, report_drafts, critique_feedback, iterations, and more."),
                ("Reducers", "state.py - Annotated[List, operator.add]",
                 "Auto-merge on every node return. This is what makes the loops work: each pass appends drafts, feedback and sources instead of overwriting them."),
                ("Planner", "app/orchestrator/planner.py - plan()",
                 "Decomposes a complex request into 2-5 sub-tasks via LLM structured output; deterministic 3-step fallback (gather -> analyze -> synthesize) without keys."),
                ("Adaptive loops", "app/orchestrator/router.py + nodes.py",
                 "Two closed feedback cycles (coverage deepening + report revision)."),
                ("Checkpointing", "app/services/checkpoint.py - sqlite_checkpointer()",
                 "Persists graph execution state to SQLite between node visits (WAL mode, busy timeout), so runs survive restarts."),
            ]),
            ("h2", "B. Researcher agent - gather knowledge"),
            ("table3", [
                ("Tool-calling agent", "app/agents/researcher.py - run_research()",
                 "Wraps the LLM + tools in create_react_agent; the model decides when to call a tool."),
                ("Web search tool", "app/services/tools.py - build_web_search_tool()",
                 "Tavily live internet search (WEB_SEARCH_MAX_RESULTS)."),
                ("Document retrieval tool", "tools.py - build_retrieve_tool()",
                 "Semantic find-relevant-chunks-in-FAISS tool the agent can call."),
                ("Evidence extraction", "researcher.py - _gather_tool_sources()",
                 "Collects raw tool output so facts stay attributable."),
                ("Graceful degradation", "researcher.py - _fallback_research()",
                 "No keys? Still runs on the FAISS store - the pipeline never breaks in demo/CI."),
                ("Per-sub-task evidence", "researcher.py - run_all_research()",
                 "Returns per_subtask = [{title, findings, sources}] - input to the Coverage Scout."),
            ]),
            ("h2", "C. Coverage Scout - self-tuning research depth (unique)"),
            ("table3", [
                ("Evidence scoring", "app/orchestrator/nodes.py - _coverage_score()",
                 "Scores a sub-task 0-1: rewards length (0.35), real numbers (0.30), named sources/URLs (0.25). Punishes filler."),
                ("Gap detection", "coverage_node()",
                 "Flags sub-tasks scoring below RESEARCH_COVERAGE_THRESHOLD (default 0.6)."),
                ("GO DEEPER re-query", "coverage_node() - sets research_focus",
                 "Rewrites thin sub-tasks with a sharper focus and re-routes them into research again."),
                ("Depth cap", "route_after_coverage() + MAX_RESEARCH_DEPTH",
                 "Prevents infinite loops; after max depth it accepts what it has."),
                ("Audit trail", "coverage_node() - evidence + coverage_score",
                 "Every task carries a public evidence table + overall score."),
            ]),
            ("h2", "D. Data Analyst agent - sandboxed computation"),
            ("table3", [
                ("Code generation", "app/agents/analyst.py - _generate_code()",
                 "LLM writes a Python analysis script for the raw data."),
                ("Sandbox REPL", "app/services/sandbox.py - run_code()",
                 "Executes in a real subprocess with python -I (isolated mode), a fresh temp dir, and a hard timeout."),
                ("Result conventions", "sandbox.py",
                 "Code writes JSON to analysis_result.json and PNGs to charts/; the sandbox returns parsed JSON + base64 charts + stdout/stderr."),
                ("Self-heal loop", "analyst.py run_analysis() / _fix_code()",
                 "Script failed? The error goes back to the LLM, it rewrites the code, execution retries (up to 3)."),
                ("Safety net", "analyst.py - _FALLBACK_CODE",
                 "A guaranteed-safe summarizing script - analysis can never come back empty."),
                ("Output caps", "sandbox.py",
                 "Truncates stdout/stderr at SANDBOX_MAX_OUTPUT_BYTES and kills runaway processes at timeout."),
            ]),
            ("h2", "E. Writer + Critiquer - quality-gated generation"),
            ("table3", [
                ("Report draft", "app/agents/writer.py - write_report()",
                 "Writes a structured markdown report: title, exec summary, findings, analysis, conclusions, evidence & coverage, references."),
                ("Grounding", "writer.py",
                 "Prompt receives research findings + analysis + sources + evidence; instructed to never invent facts."),
                ("Critique gate", "app/agents/critiquer.py - critique_report()",
                 "Scores a draft 0-1 against guidelines (structure, completeness, accuracy, evidence, length). Structured LLM output; heuristic fallback without keys."),
                ("Revision loop", "nodes.py report_node/critique_node + route_after_critique()",
                 "Fail -> feedback list appended -> Writer revises; pass or MAX_CRITIQUE_ITERATIONS -> finalize."),
                ("Draft history", "state.py - report_drafts",
                 "Every revision is kept - you can diff attempts via the API response."),
            ]),
            ("h2", "F. Infrastructure & durability"),
            ("table3", [
                ("Sync entry point", "app/api/routes.py - run_agent()",
                 "POST /agent/run blocks until the report is finished and returns the full response + charts."),
                ("Async entry point", "routes.py - webhook_agent()",
                 "Returns 202 {task_id}; the task is enqueued and runs in the background."),
                ("Worker pool", "app/services/task_queue.py - submit()",
                 "ThreadPoolExecutor (WORKER_COUNT) drains queued tasks."),
                ("Durable task store", "app/services/task_store.py - get_task_store()",
                 "SQLite table of status + full result per task_id (WAL mode, survives restarts)."),
                ("Polling API", "routes.py - get_task() / list_tasks()",
                 "GET /tasks/{id} -> queued/running/completed/failed; GET /tasks lists recent."),
                ("Sandbox playground", "routes.py - analyze_code()",
                 "POST /analyze runs arbitrary Python and returns JSON + charts - great for testing."),
                ("Index rebuild", "routes.py - rebuild_index()",
                 "POST /index reindexes local docs into FAISS."),
                ("Health check", "routes.py - health()",
                 "Reports whether LLM and web-search are configured."),
            ]),
            ("h2", "G. Vectorstore - your knowledge base"),
            ("table3", [
                ("Document loading", "app/services/vectorstore.py - load_documents()",
                 "Reads .md/.txt/.csv/.json/.html from data/raw_docs."),
                ("Index build / load", "vectorstore.py - build_index() / get_index()",
                 "Embeds chunks (OpenAI embeddings) -> FAISS index -> saved to data/faiss_index; cached in memory."),
                ("Retriever", "vectorstore.py - get_retriever()",
                 "LangChain retriever for semantic top-k search."),
            ]),
        ],
    ),
    (
        "3. The pipelines, step by step",
        [
            ("h2", "Pipeline 1 - Main orchestration graph"),
            ("table2", [
                ("START", "Plan decomposes the request into sub-tasks."),
                ("research", "For each sub-task the agent calls web_search / retrieve_local_documents."),
                ("coverage", "Score evidence per sub-task. Weak & depth remains -> research (GO DEEPER); sufficient / max depth -> analyze."),
                ("analyze", "Generate Python -> run in sandbox -> analysis_result.json + charts."),
                ("report", "Writer drafts the report from findings + analysis + evidence."),
                ("critique", "Critiquer scores the draft. Fail & iterations remain -> report (with feedback); pass / max -> finalize."),
                ("finalize", "status = 'completed', report finalized."),
            ]),
            ("pre", SEMANTIC_FLOW),
            "Cycle counts: up to MAX_RESEARCH_DEPTH coverage loops + MAX_CRITIQUE_ITERATIONS revision loops per run.",
            ("h2", "Pipeline 2 - Async job lifecycle (n8n flow)"),
            ("table2", [
                ("Trigger", "n8n email trigger fires."),
                ("POST", "/webhook/agent {user_request} -> 202 {task_id}."),
                ("Running", "Worker pool picks it up; status set to 'running'."),
                ("Execute", "run_task(): build graph + SQLite checkpointer -> invoke()."),
                ("Persist", "Result written to SQLite; status 'completed' or 'failed'."),
                ("Retrieve", "n8n polls GET /tasks/{id} until completed, then reads report + charts."),
            ]),
            ("h2", "Pipeline 3 - Research (per sub-task)"),
            ("pre", RESEARCH_FLOW),
            ("h2", "Pipeline 4 - Data analysis with self-heal"),
            ("pre", ANALYSIS_FLOW),
            ("h2", "Pipeline 5 - Report refinement loop"),
            ("pre", WRITER_CRITIQUE_FLOW),
            ("h2", "Pipeline 6 - Retrieval & indexing"),
            ("pre", DATA_FLOW),
        ],
    ),
    (
        "4. Configuration knobs (.env)",
        [
            ("table3", [
                ("Key", "Default", "Controls"),
                ("OPENAI_API_KEY / OPENAI_MODEL", "gpt-4o-mini", "The reasoning brain"),
                ("TAVILY_API_KEY / WEB_SEARCH_MAX_RESULTS", "5", "Live web research"),
                ("RESEARCH_COVERAGE_THRESHOLD / MAX_RESEARCH_DEPTH", "0.6 / 2", "Coverage Scout strictness"),
                ("MAX_CRITIQUE_ITERATIONS / CRITIQUE_THRESHOLD / REPORT_MAX_WORDS", "3 / 0.85 / 2500", "Writer-Critiquer gate"),
                ("SANDBOX_TIMEOUT_SECONDS / SANDBOX_MAX_OUTPUT_BYTES", "60 / 200000", "Analyst sandbox isolation"),
                ("WORKER_COUNT / TASK_DB_PATH / CHECKPOINT_DIR", "2 / sqlite / checkpoints", "Durability & concurrency"),
            ]),
        ],
    ),
]


def _build_styles():
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "PageTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0F2557"),
        spaceAfter=4,
    )
    subtitle = ParagraphStyle(
        "PageSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#555555"),
        spaceAfter=18,
    )
    h1 = ParagraphStyle(
        "SectionH1",
        parent=styles["Heading1"],
        fontSize=13,
        leading=16,
        spaceBefore=14,
        spaceAfter=6,
        textColor=colors.HexColor("#0F2557"),
    )
    h2 = ParagraphStyle(
        "SectionH2",
        parent=styles["Heading2"],
        fontSize=11,
        leading=14,
        spaceBefore=10,
        spaceAfter=4,
        textColor=colors.HexColor("#1F4890"),
    )
    body = ParagraphStyle(
        "BodyDoc",
        parent=styles["Normal"],
        fontSize=9,
        leading=12.5,
        spaceAfter=5,
    )
    mono = ParagraphStyle(
        "MonoDoc",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=7.5,
        leading=10.5,
        backColor=colors.HexColor("#F4F6FB"),
        borderPadding=6,
        borderColor=colors.HexColor("#D5DCE8"),
        borderWidth=0.5,
        textColor=colors.HexColor("#22304A"),
        spaceAfter=8,
    )
    return {"title": title, "subtitle": subtitle, "h1": h1, "h2": h2, "body": body, "mono": mono}


def _esc(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _cell(text, style):
    return Paragraph(_esc(str(text)), style)


def _table(rows, ncols, styles):
    widths = {2: [30 * mm, 138 * mm], 3: [34 * mm, 52 * mm, 82 * mm]}[ncols]
    bold = styles["body"].clone("BoldCell", textColor=colors.HexColor("#0F2557"))
    data = [[_cell(c, bold if r == 0 else styles["body"]) for c in row] for r, row in enumerate(rows)]
    table = Table(data, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F2557")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C9D2E0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F6FB")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def main() -> int:
    styles = _build_styles()
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="Multi-Agent Autonomous Research & Execution System - Overview",
        author="Multi-agent project",
    )

    story = [Paragraph(TITLE, styles["title"]), Paragraph(SUBTITLE, styles["subtitle"])]

    for section_title, content in SECTIONS:
        story.append(Paragraph(_esc(section_title), styles["h1"]))
        for item in content:
            if isinstance(item, tuple) and item[0] == "pre":
                story.append(Paragraph(_esc(item[1]).replace("\n", "<br/>"), styles["mono"]))
            elif isinstance(item, tuple) and item[0] == "h2":
                story.append(Paragraph(_esc(item[1]), styles["h2"]))
            elif isinstance(item, tuple) and item[0].startswith("table"):
                ncols = 3 if item[0] == "table3" else 2
                story.append(_table(item[1], ncols, styles))
            else:
                story.append(Paragraph(_esc(str(item)), styles["body"]))

    story.append(Spacer(1, 14))
    story.append(Paragraph(
        "Generated from the codebase - Multi-Agent Autonomous Research &amp; Execution System.",
        styles["body"],
    ))

    doc.build(story)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
