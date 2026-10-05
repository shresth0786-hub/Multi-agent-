"""Thread-safe progress hub for streaming agent runs over SSE.

Each task_id owns a list of events; long-lived workers append while the
SSE generator polls for fresh events and forwards them to the browser.
"""

import threading
import time
from typing import Any

_lock = threading.Lock()
_buckets: dict[str, list[dict[str, Any]]] = {}
_MAX_EVENTS = 2000

_local = threading.local()

STAGE_LABELS = {
    "plan": "Planning sub-tasks…",
    "research": "Researching (local knowledge base + web)…",
    "coverage": "Scoring evidence coverage…",
    "analyze": "Running statistical analysis…",
    "report": "Writing the report…",
    "critique": "Critiquing the draft…",
    "finalize": "Finalizing the answer…",
}


def set_active(task_id: str | None) -> None:
    """Pin the task being executed by the current thread (used by graph nodes)."""
    _local.active_task = task_id


def active() -> str | None:
    return getattr(_local, "active_task", None)


def stage(name: str) -> None:
    """Emit a live "stage" event for the task currently running on this thread."""
    task_id = active()
    label = STAGE_LABELS.get(name, name)
    if task_id:
        emit(task_id, type="status", text=label)


def register(task_id: str) -> None:
    with _lock:
        _buckets[task_id] = []


def emit(task_id: str, **event: Any) -> None:
    event.setdefault("ts", time.time())
    with _lock:
        bucket = _buckets.get(task_id)
        if bucket is None:
            return
        if len(bucket) >= _MAX_EVENTS:
            bucket.pop(0)
        bucket.append(event)


def snapshot(task_id: str) -> list[dict[str, Any]]:
    with _lock:
        return list(_buckets.get(task_id) or [])


def release(task_id: str) -> None:
    with _lock:
        _buckets.pop(task_id, None)
