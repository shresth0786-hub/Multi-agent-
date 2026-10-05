"""Runtime key vault for the application.

Lets the web UI manage provider/API keys *inside the application* instead of
requiring hand-editing ``.env``. Values are persisted in the app's own data
folder (gitignored) and applied to the live :class:`Settings` object so health,
model factories, and the pipeline pick them up immediately.

The presence of a configured key does not make this a secure vault: values are
stored in plaintext on local disk, like ``.env`` would be.
"""

import json
from pathlib import Path
from typing import Any

from app.config import get_settings

_ATTRS = ("llm_provider", "openai_api_key", "gemini_api_key", "tavily_api_key")


def _path() -> Path:
    return get_settings().key_vault_file


def _read() -> dict[str, str]:
    path = _path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: str(v) for k, v in data.items() if k in _ATTRS and v}


def _write(data: dict[str, str]) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def apply() -> None:
    """Apply persisted overrides onto the live settings (call at startup)."""
    settings = get_settings()
    for key, value in _read().items():
        setattr(settings, key, value)


def update(fields: dict[str, Any]) -> dict[str, str]:
    """Merge *fields* into the store; empty string removes an override."""
    data = _read()
    for key in _ATTRS:
        if key in fields:
            value = (fields.get(key) or "").strip()
            if value:
                data[key] = value
            else:
                data.pop(key, None)
    _write(data)
    apply()
    return status()


def status() -> dict[str, Any]:
    settings = get_settings()
    return {
        "provider": settings.llm_provider,
        "llm_configured": settings.has_llm,
        "web_search_configured": settings.has_web_search,
        "gemini_configured": settings.has_gemini,
        "openai_api_key": _masked(settings.openai_api_key),
        "gemini_api_key": _masked(settings.gemini_api_key),
        "tavily_api_key": _masked(settings.tavily_api_key),
    }


def reset() -> None:
    """Clear overrides from the live settings (used by tests)."""
    settings = get_settings()
    for key in _ATTRS:
        setattr(settings, key, "" if key != "llm_provider" else "openai")


def _masked(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 10:
        return "•••set"
    return f"{value[:5]}…{value[-4:]}"
