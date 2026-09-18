"""Background task queue: webhook submits, a worker pool executes, and the
SQLite store holds status + results for clients to poll.
"""

import atexit
import concurrent.futures
import threading
from uuid import uuid4

from app.config import get_settings
from app.services.task_store import get_task_store

_executor: concurrent.futures.ThreadPoolExecutor | None = None
_worker_lock = threading.Lock()


def _get_executor() -> concurrent.futures.ThreadPoolExecutor:
    global _executor
    if _executor is None:
        with _worker_lock:
            if _executor is None:
                _executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=get_settings().worker_count,
                    thread_name_prefix="agent-worker",
                )
                atexit.register(_executor.shutdown, wait=False)
    return _executor


def submit(user_request: str, metadata: dict | None = None) -> str:
    """Enqueue a task; returns the task_id to poll with GET /tasks/{id}."""
    task_id = str(uuid4())
    store = get_task_store()
    store.create(task_id, user_request, metadata)

    def _job() -> None:
        store.set_status(task_id, "running")
        try:
            from app.services.runner import run_task

            response = run_task(user_request, metadata, task_id)
            response.status = "completed"
            store.store_result(task_id, response.model_dump(), status="completed")
        except Exception as exc:  # noqa: BLE001
            store.set_status(task_id, "failed", error=str(exc))

    _get_executor().submit(_job)
    return task_id
