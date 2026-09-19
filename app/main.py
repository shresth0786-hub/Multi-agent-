# NOTE: router imports `from app.api.schemas import ...` — safe at import time.
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import get_settings

settings = get_settings()
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Multi-Agent Research & Execution System",
    description=(
        "Stateful, event-driven orchestrator (LangGraph) over specialized "
        "research / analysis / writing agents, exposed as the n8n bridge."
    ),
    version="0.1.0",
)

app.include_router(router, prefix=settings.api_prefix)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
