import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch, tmp_path):
    """Isolate tests from the user's persisted state.

    Points the key vault, task DB, and checkpoint dir at throwaway per-test
    paths and clears in-app keys, so tests never read, write, or delete the
    real settings file, task history, or checkpoints.
    """
    from app.config import get_settings
    from app.services.env_overrides import reset

    monkeypatch.setenv("KEY_VAULT_FILE", str(tmp_path / "vault.json"))
    monkeypatch.setenv("TASK_DB_PATH", str(tmp_path / "tasks.sqlite"))
    monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path / "checkpoints"))
    get_settings.cache_clear()
    reset()
    yield
    get_settings.cache_clear()
    reset()
