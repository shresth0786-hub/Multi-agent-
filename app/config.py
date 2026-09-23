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

    # LLM (OpenAI)
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    llm_temperature: float = 0.2

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

    # Writer / Critiquer loop
    max_critique_iterations: int = 3
    critique_threshold: float = 0.85
    report_max_words: int = 2500

    # Style training (few-shot examples + fine-tuning source data)
    style_examples_file: Path = (
        PROJECT_ROOT / "data" / "examples" / "style_examples.jsonl"
    )

    # App
    app_name: str = "multi-agent-research"
    api_prefix: str = "/api/v1"

    @property
    def has_llm(self) -> bool:
        return usable_key(self.openai_api_key)

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
