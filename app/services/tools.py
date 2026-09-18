from langchain_core.tools import tool

from app.config import get_settings


def build_web_search_tool():
    """Tavily web search tool, or None when not configured/enabled."""
    settings = get_settings()
    if not settings.has_web_search:
        return None
    from langchain_community.tools.tavily_search import TavilySearchResults

    return TavilySearchResults(
        max_results=settings.web_search_max_results,
        tavily_api_key=settings.tavily_api_key,
    )


def build_retrieve_tool():
    """Tool that retrieves relevant chunks from the local FAISS store."""
    from app.services.vectorstore import get_retriever

    retriever = get_retriever()
    if retriever is None:
        return None

    @tool("retrieve_local_documents")
    def retrieve(query: str) -> str:
        """Search the organisation's local documents for information relevant to the query.

        Args:
            query: a natural language retrieval query.
        """
        docs = retriever.invoke(query)
        if not docs:
            return "No local documents matched the query."
        return "\n\n---\n\n".join(d.page_content for d in docs)

    return retrieve


def build_tools() -> list:
    """All tools available to the Researcher agent."""
    return [t for t in (build_web_search_tool(), build_retrieve_tool()) if t is not None]
