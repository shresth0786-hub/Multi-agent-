import os
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

# Free-first provider chain:
# LLM_PROVIDER=grog|gemini|ollama|openai  (auto-detect if not set)
# GROQ_API_KEY (free at console.groq.com) -> llama-3.1-8b-instant (~14k free req/day)
# GOOGLE_API_KEY (free at aistudio.google.com) -> gemini-1.5-flash (1500 req/day free)
# Ollama (fully local, no key) -> llama3.1 via http://localhost:11434
# OpenAI last fallback (paid)

def _detect_provider() -> str:
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit in {"groq", "gemini", "ollama", "openai", "huggingface"}:
        return explicit
    # auto-detect: prefer free providers if their key/env exists
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    if os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    if os.getenv("OLLAMA_BASE_URL") or os.getenv("USE_OLLAMA", "").lower() in {"1", "true"}:
        return "ollama"
    if os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY") != "sk-...":
        return "openai"
    # default to groq-free shape but will error with clear message if no key -> falls back to ollama/hf
    return os.getenv("DEFAULT_FREE_PROVIDER", "groq")


@lru_cache(maxsize=1)
def get_llm():
    """Singleton LLM - free-first, cached."""
    provider = _detect_provider()

    # Common params
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", os.getenv("LLM_MAX_TOKENS", "2000")))

    if provider == "groq":
        try:
            from langchain_groq import ChatGroq
        except ImportError as e:
            raise ImportError("Groq provider selected but langchain-groq not installed. Run: pip install langchain-groq") from e
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY not set. Get free key at https://console.groq.com/keys and set LLM_PROVIDER=groq")
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=3,
        )

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:
            raise ImportError("Gemini provider selected but langchain-google-genai not installed. Run: pip install langchain-google-genai") from e
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY not set. Get free key at https://aistudio.google.com/app/apikey and set LLM_PROVIDER=gemini")
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
            temperature=temperature,
            max_output_tokens=max_tokens,
            max_retries=3,
        )

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            from langchain_community.chat_models import ChatOllama  # fallback
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.1"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=temperature,
            num_predict=max_tokens,
        )

    # openai fallback (paid)
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=temperature,
        max_retries=3,
        request_timeout=30,
        max_tokens=max_tokens,
    )


@lru_cache(maxsize=1)
def get_embeddings():
    """Singleton embeddings - free-first (HF local), no API cost by default."""
    # EMBEDDING_PROVIDER=huggingface|openai|google|ollama (auto)
    provider = os.getenv("EMBEDDING_PROVIDER", "").strip().lower()
    if not provider:
        # Default to free local HF to avoid any API quota/billing
        provider = os.getenv("DEFAULT_EMBED_PROVIDER", "huggingface")

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
            chunk_size=1000,
            max_retries=3,
            request_timeout=30,
        )

    if provider == "google":
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except ImportError as e:
            raise ImportError("pip install langchain-google-genai for Google embeddings") from e
        return GoogleGenerativeAIEmbeddings(model=os.getenv("GEMINI_EMBED_MODEL", "models/embedding-001"))

    if provider == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError:
            from langchain_community.embeddings import OllamaEmbeddings

        return OllamaEmbeddings(
            model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

    # huggingface - fully free, local, no API key (downloads ~80MB on first run)
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        cache_folder=os.getenv("HF_CACHE_DIR", None),
    )


class _LazyProxy:
    def __init__(self, factory):
        self._factory = factory
        self._instance = None

    def _get(self):
        if self._instance is None:
            self._instance = self._factory()
        return self._instance

    def __getattr__(self, name):
        return getattr(self._get(), name)

    def invoke(self, *a, **kw):
        return self._get().invoke(*a, **kw)

    async def ainvoke(self, *a, **kw):
        return await self._get().ainvoke(*a, **kw)

    def batch(self, *a, **kw):
        return self._get().batch(*a, **kw)


llm = _LazyProxy(get_llm)
embeddings = _LazyProxy(get_embeddings)

