import threading
import time
from typing import Any

from app.config import get_settings, usable_key

_rate_lock = threading.Lock()
_last_request_at = 0.0


def active_provider() -> str:
    """Configured LLM provider: 'openai' or 'gemini'."""
    return _provider()


def throttle() -> None:
    """Space out Gemini calls to respect its per-minute free-tier limit.

    A full pipeline issues several calls back-to-back, which trips the
    "5 requests per minute per model" limit and makes later stages fail.
    """
    settings = get_settings()
    if active_provider() != "gemini":
        return
    interval = float(getattr(settings, "gemini_min_request_interval_seconds", 0.0) or 0.0)
    if interval <= 0:
        return
    global _last_request_at
    with _rate_lock:
        wait = interval - (time.monotonic() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()


class _ThrottledChatModel:
    """Chat model proxy that paces each request for rate-limited providers."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        throttle()
        return self._model.invoke(*args, **kwargs)

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        throttle()
        return self._model.stream(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


def _provider() -> str:
    return (get_settings().llm_provider or "openai").strip().lower() or "openai"


def gemini_available() -> bool:
    """True when the Gemini provider is selected AND a real key is present."""
    settings = get_settings()
    return usable_key(settings.gemini_api_key) and _provider() == "gemini"


def openai_available() -> bool:
    settings = get_settings()
    return usable_key(settings.openai_api_key)


def get_chat_model(model: str | None = None, temperature: float | None = None) -> Any:
    """Return the configured chat model (OpenAI or Gemini), or None without keys.

    Also returns None when the Gemini extras are not installed so the rest of
    the app degrades gracefully instead of crashing.
    """
    settings = get_settings()
    temperature = (
        settings.llm_temperature if temperature is None else temperature
    )

    if _provider() == "gemini":
        if not usable_key(settings.gemini_api_key):
            return None
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            return _ThrottledChatModel(
                ChatGoogleGenerativeAI(
                    model=model or settings.gemini_model,
                    temperature=temperature,
                    api_key=settings.gemini_api_key,
                    # Fail fast on 429/quota errors instead of the SDK's long
                    # backoff, which would stall a streaming request for minutes.
                    max_retries=settings.llm_max_retries,
                )
            )
        except ImportError:
            return None

    if not usable_key(settings.openai_api_key):
        return None
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model or settings.openai_model,
        temperature=temperature,
        api_key=settings.openai_api_key,
    )


def get_embeddings() -> Any:
    """Return an embeddings model (OpenAI preferred, Gemini as fallback)."""
    settings = get_settings()
    if usable_key(settings.openai_api_key):
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )
    if usable_key(settings.gemini_api_key):
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            return GoogleGenerativeAIEmbeddings(
                model=settings.gemini_embedding_model,
                google_api_key=settings.gemini_api_key,
            )
        except ImportError:
            return None
    return None
