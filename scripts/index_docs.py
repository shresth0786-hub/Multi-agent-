"""Build the local FAISS index from data/raw_docs.

Requires OPENAI_API_KEY (embeddings). Run: python -m scripts.index_docs
"""

from app.config import get_settings
from app.services.vectorstore import build_index, load_documents, reset_index_cache


def main() -> int:
    settings = get_settings()
    reset_index_cache()

    docs = load_documents()
    if not docs:
        print(f"No documents found in {settings.data_dir.resolve()}.")
        print("Seed the sample corpus first:  python -m scripts.seed_docs")
        return 1
    if not settings.openai_api_key:
        print("OPENAI_API_KEY is required to generate embeddings.")
        return 1

    db = build_index(force=True)
    if db is None:
        print("Index build failed.")
        return 1
    print(f"Indexed {len(docs)} document(s) into {settings.index_dir.resolve()}.")
    print("Held in memory for the running process; reload on next invocation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
