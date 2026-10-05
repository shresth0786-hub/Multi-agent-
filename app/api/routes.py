"""HTTP layer: the n8n bridge into the LangGraph orchestrator.

Webhook endpoints are async (202 + poll); ``/agent/run`` blocks and returns the
full result for direct/report use.
"""

from uuid import uuid4

from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    AgentRunResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    RunRequest,
    TaskAcceptedResponse,
    WebhookRequest,
)
from app.config import get_settings
from app.services.runner import run_task
from app.services.sandbox import run_code
from app.services.task_queue import submit as submit_task
from app.services.task_store import get_task_store

router = APIRouter()


def _build_response(
    status: str, task_id: str, payload: dict | None, error: str | None
) -> AgentRunResponse:
    base = AgentRunResponse(task_id=task_id, status=status, error=error or "")
    if payload:
        merged = base.model_dump()
        merged.update(payload)
        merged["status"] = status  # keep outer status (queued/running/completed/failed)
        return AgentRunResponse(**merged)
    return base


@router.post("/agent/run", response_model=AgentRunResponse, tags=["orchestrator"])
def run_agent(payload: RunRequest):
    """Synchronous execution: blocks until the report is finalized."""
    task_id = str(uuid4())
    store = get_task_store()
    store.create(task_id, payload.user_request, payload.metadata)
    try:
        response = run_task(
            payload.user_request,
            payload.metadata,
            task_id,
            session_id=payload.session_id,
            quick=payload.quick,
        )
    except Exception as exc:  # noqa: BLE001
        store.set_status(task_id, "failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Orchestrator failed: {exc}") from exc
    store.store_result(task_id, response.model_dump())
    return response


@router.post("/agent/stream", tags=["orchestrator"])
def stream_agent(payload: RunRequest):
    """Server-Sent-Events endpoint for the ChatGPT-style chat.

    Emits ``status`` (pipeline stage), ``chunk`` (streamed quick-answer text)
    and ``answer`` events; the client renders them live.
    """

    import json
    import time
    from concurrent.futures import ThreadPoolExecutor

    from fastapi.responses import StreamingResponse

    from app.services.progress import emit, register, release, snapshot
    from app.services.runner import (
        conversation_context,
        persist_conversation,
        quick_answer_generator,
    )

    task_id = str(uuid4())
    store = get_task_store()
    history = (
        store.get_conversation(payload.session_id) if payload.session_id else []
    )
    context = conversation_context(history)
    use_quick = payload.quick and bool(history)

    def _sse(event: dict) -> str:
        return "data: " + json.dumps(event, default=str) + "\n\n"

    def _generate():
        try:
            if use_quick:
                pieces: list[str] = []
                for event in quick_answer_generator(
                    payload.user_request, context, task_id
                ):
                    if event.get("type") == "chunk":
                        pieces.append(event["text"])
                    yield _sse(event)
                report = "".join(pieces)
                persist_conversation(
                    payload.session_id, task_id, payload.user_request, report
                )
                yield _sse(
                    {
                        "type": "answer",
                        "data": {
                            "task_id": task_id,
                            "status": "completed",
                            "quick": True,
                            "report": report,
                        },
                    }
                )
                yield _sse({"type": "done"})
                return

            register(task_id)

            def _job():
                return run_task(
                    payload.user_request,
                    payload.metadata,
                    task_id,
                    session_id=payload.session_id,
                    history=history,
                    quick=False,
                    on_event=lambda **event: emit(task_id, **event),
                )

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_job)
                seen = 0
                while not future.done():
                    for event in snapshot(task_id)[seen:]:
                        seen += 1
                        yield _sse(event)
                    time.sleep(0.1)
                for event in snapshot(task_id)[seen:]:
                    seen += 1
                    yield _sse(event)
                response = future.result()
                yield _sse({"type": "answer", "data": response.model_dump()})
                yield _sse({"type": "done"})
        finally:
            release(task_id)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/webhook/agent", response_model=TaskAcceptedResponse, status_code=202, tags=["n8n"])
def webhook_agent(payload: WebhookRequest):
    """Async entry point for n8n: enqueues the request, returns a task_id to poll."""
    task_id = submit_task(payload.user_request, payload.metadata)
    return TaskAcceptedResponse(task_id=task_id)


@router.post("/webhook", response_model=TaskAcceptedResponse, status_code=202, tags=["n8n"])
def webhook_generic(payload: WebhookRequest):
    """Generic webhook alias; accepts any n8n payload shape with user_request."""
    task_id = submit_task(payload.user_request, payload.metadata)
    return TaskAcceptedResponse(task_id=task_id)


@router.get("/tasks", tags=["orchestrator"])
def list_tasks(limit: int = 50):
    if not (0 < limit <= 200):
        raise HTTPException(status_code=422, detail="limit must be between 1 and 200")
    return get_task_store().list_tasks(limit=limit)


@router.get("/tasks/{task_id}", response_model=AgentRunResponse, tags=["orchestrator"])
def get_task(task_id: str):
    task = get_task_store().get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return _build_response(
        status=task["status"],
        task_id=task_id,
        payload=task.get("payload"),
        error=task.get("error"),
    )


@router.post("/analyze", response_model=AnalyzeResponse, tags=["sandbox"])
def analyze_code(payload: AnalyzeRequest):
    """Focused analyst sandbox: run arbitrary analysis code, get JSON + charts."""
    result = run_code(payload.code)
    return AnalyzeResponse(
        ok=result["ok"],
        stdout=result["stdout"],
        stderr=result["stderr"],
        result=result["result"],
        charts=result["charts"],
        error=result["error"],
    )


@router.post("/index", tags=["vectorstore"])
def rebuild_index(force: bool = False):
    """(Re)build the FAISS index from data/raw_docs; returns the chunk count."""
    from app.services.vectorstore import build_index, load_documents, reset_index_cache

    reset_index_cache()
    db = build_index(force=force)
    if db is None:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed: set OPENAI_API_KEY and add files to data/raw_docs.",
        )
    return {"indexed_documents": len(load_documents())}


@router.get("/health", tags=["meta"])
def health():
    settings = get_settings()
    return {
        "app": settings.app_name,
        "status": "ok",
        "llm_configured": settings.has_llm,
        "web_search_configured": settings.has_web_search,
        "provider": settings.llm_provider,
        "gemini_configured": settings.has_gemini,
    }


@router.get("/settings", tags=["meta"])
def get_settings_status():
    """Masked view of the currently active provider/keys (in-app key vault)."""
    from app.services.env_overrides import status as vault_status

    return vault_status()


@router.post("/settings", tags=["meta"])
def save_settings(fields: dict):
    """Save provider/API keys from the in-app Settings panel.

    Empty strings clear an override. Keys are masked in every response.
    """
    from app.services.env_overrides import update

    allowed = {"llm_provider", "openai_api_key", "gemini_api_key", "tavily_api_key"}
    clean = {k: v for k, v in fields.items() if k in allowed}
    if "llm_provider" in clean and clean["llm_provider"] not in {"openai", "gemini"}:
        raise HTTPException(status_code=422, detail="provider must be openai or gemini")
    return update(clean)
