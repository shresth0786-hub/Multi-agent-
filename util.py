import os
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

@lru_cache(maxsize=1)
def get_vectorstore():
    """Singleton Chroma vectorstore with persistence - uses free HF embeddings by default."""
    # Prefer langchain-chroma (new), fallback to community
    try:
        from langchain_chroma import Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma

    from config import get_embeddings

    embeddings = get_embeddings()
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    return Chroma(
        collection_name="research_papers",
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )


class _VectorProxy:
    def __getattr__(self, name):
        return getattr(get_vectorstore(), name)

    def add_documents(self, *a, **kw):
        return get_vectorstore().add_documents(*a, **kw)

    def as_retriever(self, *a, **kw):
        return get_vectorstore().as_retriever(*a, **kw)

    def similarity_search(self, *a, **kw):
        return get_vectorstore().similarity_search(*a, **kw)


vectorstore = _VectorProxy()
__all__ = ["get_vectorstore", "vectorstore"]
