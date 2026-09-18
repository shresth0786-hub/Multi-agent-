"""Durable SQLite task store. Replaces the in-memory task dict so results
survive restarts; also serves as the source of truth for n8n polling.
"""

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id     TEXT PRIMARY KEY,
    status      TEXT NOT NULL,
    user_request TEXT NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}',
    payload     TEXT,
    error       TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
"""

_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute(_SCHEMA)
    return conn


class _TaskStore:
    def __init__(self) -> None:
        self._path = get_settings().task_db_path

    def create(self, task_id: str, user_request: str, metadata: dict | None = None) -> None:
        with _lock, _connect(self._path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO tasks (task_id, status, user_request, metadata, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    task_id,
                    "queued",
                    user_request,
                    json.dumps(metadata or {}),
                    _now(),
                    _now(),
                ),
            )

    def set_status(self, task_id: str, status: str, error: str | None = None) -> None:
        with _lock, _connect(self._path) as conn:
            conn.execute(
                "UPDATE tasks SET status = ?, error = ?, updated_at = ? WHERE task_id = ?",
                (status, error, _now(), task_id),
            )

    def store_result(
        self, task_id: str, payload: dict[str, Any], status: str = "completed"
    ) -> None:
        with _lock, _connect(self._path) as conn:
            conn.execute(
                "UPDATE tasks SET status = ?, payload = ?, updated_at = ? WHERE task_id = ?",
                (status, json.dumps(payload), _now(), task_id),
            )

    def get(self, task_id: str) -> dict | None:
        with _connect(self._path) as conn:
            row = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["metadata"] = json.loads(result.get("metadata") or "{}")
        result["payload"] = json.loads(result["payload"]) if result.get("payload") else None
        return result

    def list(self, limit: int = 50) -> list[dict]:
        with _connect(self._path) as conn:
            rows = conn.execute(
                "SELECT task_id, status, user_request, created_at, updated_at, error "
                "FROM tasks ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]


_task_store = _TaskStore()


def get_task_store() -> _TaskStore:
    return _task_store
