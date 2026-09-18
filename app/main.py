# NOTE: router imports `from app.api.schemas import ...` — safe at import time.
from fastapi import FastAPI

from app.api.routes import router
from app.config import get_settings

settings = get_settings()

app = FastAPI(
    title="Multi-Agent Research & Execution System",
    description=(
        "Stateful, event-driven orchestrator (LangGraph) over specialized "
        "research / analysis / writing agents, exposed as the n8n bridge."
    ),
    version="0.1.0",
)

app.include_router(router, prefix=settings.api_prefix)
