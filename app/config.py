from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Central runtime configuration, overridable via env vars or .env."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM provider: "openai" | "gemini" | "groq" | "ollama" (auto free-first)
    llm_provider: str = "openai"

    # LLM (OpenAI)
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    llm_temperature: float = 0.2
    llm_max_retries: int = 1

    # LLM (Gemini - free 1500 req/day, 5/min; use 3.5-flash for stability)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash"
    gemini_embedding_model: str = "models/gemini-embedding-001"
    gemini_min_request_interval_seconds: float = 15.0

    # LLM (Groq - free 14k req/day, fastest)
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"

    # LLM (Ollama - fully local, no key)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    ollama_embedding_model: str = "nomic-embed-text"

    # Embeddings (HF local = free, no key)
    embedding_provider: str = "huggingface"
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Research
    tavily_api_key: str = ""
    enable_web_search: bool = True
    web_search_max_results: int = 5
    research_coverage_threshold: float = 0.6
    max_research_depth: int = 2

    # Vectorstore (FAISS)
    data_dir: Path = PROJECT_ROOT / "data" / "raw_docs"
    index_dir: Path = PROJECT_ROOT / "data" / "faiss_index"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    vectorstore_top_k: int = 5

    # Sandboxed analyst
    sandbox_timeout_seconds: int = 60
    sandbox_max_output_bytes: int = 200_000
    python_executable: str | None = None

    # State & durability
    task_db_path: Path = PROJECT_ROOT / "data" / "tasks.sqlite"
    checkpoint_dir: Path = PROJECT_ROOT / "data" / "checkpoints"
    worker_count: int = 2
    task_timeout_seconds: int = 300

    # In-app key vault (Settings panel). Tests point this at a temp file.
    key_vault_file: Path = PROJECT_ROOT / "data" / "settings_overrides.json"

    # Writer / Critiquer loop
    max_critique_iterations: int = 3
    critique_threshold: float = 0.85
    report_max_words: int = 2500

    # Style training (few-shot examples + fine-tuning source data)
    style_examples_file: Path = (
        PROJECT_ROOT / "data" / "examples" / "style_examples.jsonl"
    )

    # Chat UX (streaming answer mode, conversation memory, quick follow-ups)
    max_conversation_turns: int = 8
    quick_answer_max_words: int = 200
    stream_chunk_delay: float = 0.02

    # App
    app_name: str = "multi-agent-research"
    api_prefix: str = "/api/v1"

    @property
    def has_llm(self) -> bool:
        p = (self.llm_provider or "openai").lower()
        if p == "gemini":
            return usable_key(self.gemini_api_key)
        if p == "groq":
            return usable_key(self.groq_api_key)
        if p == "ollama":
            return True
        return usable_key(self.openai_api_key)

    @property
    def has_gemini(self) -> bool:
        return usable_key(self.gemini_api_key)

    @property
    def has_groq(self) -> bool:
        return usable_key(self.groq_api_key)

    @property
    def has_web_search(self) -> bool:
        return self.enable_web_search and usable_key(self.tavily_api_key)


_PLACEHOLDER_KEYS = {"", "sk-", "tvly-", "sk-...", "tvly-..."}


def usable_key(key: str) -> bool:
    """True only for a real-looking secret, never for .env placeholders."""
    stripped = key.strip()
    if stripped in _PLACEHOLDER_KEYS or stripped.endswith("..."):
        return False
    return bool(stripped)


@lru_cache
def get_settings() -> Settings:
    return Settings()
