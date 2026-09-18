# Multi-Agent Autonomous Research & Execution System

A stateful, event-driven architecture where specialized AI agents collaborate
to solve multi-step problems — beyond single-prompt chatbot interactions.

```
n8n (email/webhook trigger)
   │  POST /api/v1/webhook/agent
   ▼
FastAPI bridge ──► LangGraph Orchestrator (supervisor)
                        │  plan │  (decomposes request into sub-tasks)
                        ▼
                  Researcher Agent
                        │  tools: Tavily web search + FAISS local docs
                        ▼
                 Data Analyst Agent
                        │  sandboxed Python REPL (subprocess / python -I)
                        ▼
               Writer ──► Critiquer ──► (pass?) ──► Final report
                 ▲            │ fail
                 └── feedback loop ──┘
```

## Components

| Component | Location | Responsibility |
| --- | --- | --- |
| **Orchestrator** | `app/orchestrator/` | LangGraph state graph: `plan → research → analyze → report ↔ critique → finalize`. Manages shared state with reducers (drafts, feedback, sources accumulate across loop iterations). |
| **Researcher** | `app/agents/researcher.py` | Agent with tool-calling (`create_react_agent`): web search via **Tavily**, local semantic search via **FAISS** vectorstore. Falls back to local-only retrieval without keys. |
| **Data Analyst** | `app/agents/analyst.py` | Writes Python to analyze the dataset, executes it in a sandboxed REPL (`app/services/sandbox.py`), returns statistics + base64 charts. The sandbox runs `python -I` subprocesses with wall-clock timeout and output caps. |
| **Writer & Critiquer** | `app/agents/writer.py`, `app/agents/critiquer.py` | Writer drafts; Critiquer scores against quality guidelines (structure, completeness, evidence, length). Drafts that fail route back to the Writer with feedback, up to `MAX_CRITIQUE_ITERATIONS`. |
| **Bridge** | `app/api/` | FastAPI backend: n8n webhook, task lookup, sandbox playground, index rebuild, health. |

## Quick start (local)

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -e ".[dev]"

cp .env.example .env                                 # add OPENAI_API_KEY / TAVILY_API_KEY
python -m scripts.seed_docs                          # sample corpus
python -m scripts.index_docs                         # build FAISS index

uvicorn app.main:app --reload --port 8000
```

The pipeline runs end-to-end **even without API keys** — every stage has a
deterministic fallback — so you can verify wiring first, then add keys.

## Calling it

```bash
# A) ASYNC (recommended for n8n): 202 + poll. State is persisted in SQLite,
#    so tasks survive restarts.
curl -X POST http://localhost:8000/api/v1/webhook/agent \
  -H "Content-Type: application/json" \
  -d '{"user_request":"Analyze our Q2 acquisition funnel and recommend optimizations."}'
# -> {"task_id":"...","status":"queued","detail":"task accepted; poll GET /tasks/{id}"}

curl http://localhost:8000/api/v1/tasks/<task_id>        # poll until status == "completed"
curl http://localhost:8000/api/v1/tasks                  # list recent tasks

# B) SYNC run (blocks until the report is finalized; returns everything):
curl -X POST http://localhost:8000/api/v1/agent/run \
  -H "Content-Type: application/json" \
  -d '{"user_request":"Analyze our Q2 acquisition funnel and recommend optimizations."}'

# sandbox playground
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"code":"x=[1,2,3,4,5]\nimport json;\njson.dump({\"mean\":sum(x)/len(x)}, open(\"analysis_result.json\",\"w\"))"}'

# rebuild the FAISS index
curl -X POST "http://localhost:8000/api/v1/index?force=true"
```

Durability: task results + statuses live in `data/tasks.sqlite` and graph
checkpoints in `data/checkpoints/` (both volume-mounted in Docker). A
background worker pool (`WORKER_COUNT`) drains queued tasks.

## n8n

Import `n8n/workflow-email-to-agent.json`. It reads an email (IMAP), posts the
body to `/api/v1/agent/run`, and the orchestrator returns the final report in
the same HTTP response. For longer jobs, switch the node URL to
`/api/v1/webhook/agent` and add a polling step on `/api/v1/tasks/{task_id}`.

## Docker

```bash
docker compose up --build
# http://localhost:8000/api/v1/health
```

## n8n

Import `n8n/workflow-email-to-agent.json`. It reads an email (IMAP), posts the
body to the FastAPI webhook, and the orchestrator returns the final report.
Change the URL to your deployed host.

## AWS deployment

See [`deploy/aws/README.md`](deploy/aws/README.md) for ECS Fargate and
EC2 + systemd paths (Docker image pushed to ECR, secrets from Secrets Manager).

## Graph schema

```text
START ─► plan ─► research ─► analyze ─► report ─► critique ─► finalize ─► END
                                                    ▲            │ fail (≤ MAX iterations)
                                                    └────────────┘
```

Conditional edge after `critique` routes back to `report` (with accumulated
`critique_feedback`) until `approved` or the iteration cap is hit.

## Project layout

```
app/
  agents/        researcher, analyst, writer, critiquer
  api/           FastAPI routes + schemas
  orchestrator/  state, planner, nodes, router, graph
  prompts/       system prompts per agent
  services/      llm, tools, vectorstore (FAISS), sandbox
scripts/         seed_docs, index_docs
n8n/             sample email -> agent workflow
deploy/aws/      ECR/ECS + EC2 systemd units
tests/           pytest suite (no keys required)
```

## Tests

```bash
pytest
```