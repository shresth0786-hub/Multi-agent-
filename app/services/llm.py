from typing import Any

from app.config import get_settings


def get_chat_model(model: str | None = None, temperature: float | None = None) -> Any:
    """Return a ChatOpenAI instance, or None when no API key is configured."""
    settings = get_settings()
    if not settings.openai_api_key:
        return None
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model or settings.openai_model,
        temperature=settings.llm_temperature if temperature is None else temperature,
        api_key=settings.openai_api_key,
    )


def get_embeddings() -> Any:
    """Return an OpenAIEmbeddings instance, or None when no API key is configured."""
    settings = get_settings()
    if not settings.openai_api_key:
        return None
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
    )
