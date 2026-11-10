from pathlib import Path

from app.config import get_settings
from app.services.llm import get_embeddings

_SUPPORTED_SUFFIXES = {".md", ".txt", ".csv", ".json", ".html"}

_index = None


def load_documents(data_dir: Path | None = None) -> list[str]:
    """Read every supported text file under *data_dir* into a flat text list."""
    settings = get_settings()
    directory = data_dir or settings.data_dir
    if not directory.exists():
        return []
    docs: list[str] = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in _SUPPORTED_SUFFIXES and path.is_file():
            try:
                docs.append(path.read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                continue
    return docs


def build_index(force: bool = False) -> object | None:
    """Create (or rebuild) the local FAISS index from documents in data_dir."""
    settings = get_settings()
    embeddings = get_embeddings()
    if embeddings is None:
        return None

    texts = load_documents()
    if not texts:
        return None

    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_text("\n\n".join(texts))
    if not chunks:
        return None

    if force and settings.index_dir.exists():
        for f in settings.index_dir.glob("*"):
            if f.is_file():
                f.unlink()

    db = FAISS.from_texts(chunks, embeddings)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    db.save_local(str(settings.index_dir))
    return db


def get_index() -> object | None:
    """Load the FAISS index from disk, building it on first use if needed."""
    global _index
    if _index is not None:
        return _index

    settings = get_settings()
    embeddings = get_embeddings()
    if embeddings is None:
        return None

    if (settings.index_dir / "index.faiss").exists():
        from langchain_community.vectorstores import FAISS

        _index = FAISS.load_local(
            str(settings.index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        _index = build_index()
    return _index


def get_retriever():
    """Return a similarity-search retriever over local documents (or None)."""
    index = get_index()
    if index is None:
        return None
    return index.as_retriever(search_kwargs={"k": get_settings().vectorstore_top_k})


def search_local_documents(query: str, top_k: int | None = None) -> list[str]:
    """Raw similarity search over local documents, returning text chunks."""
    retriever = get_retriever()
    if retriever is None:
        return []
    k = top_k or get_settings().vectorstore_top_k
    docs = retriever.invoke(query)
    return [d.page_content for d in docs[:k]]


def reset_index_cache() -> None:
    global _index
    _index = None
