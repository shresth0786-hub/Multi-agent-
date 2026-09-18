"""LangGraph checkpoint persistence (SQLite).

Each graph invocation gets its own connection so concurrent background tasks
never share a sqlite connection across threads.
"""

import sqlite3
from contextlib import contextmanager

from app.config import get_settings


@contextmanager
def sqlite_checkpointer():
    """Yield a SQLite-backed checkpointer for graph compilation."""
    settings = get_settings()
    settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    db_path = settings.checkpoint_dir / "checkpoints.sqlite"

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        saver = SqliteSaver(conn)
        saver.setup()
        yield saver
    finally:
        conn.close()
