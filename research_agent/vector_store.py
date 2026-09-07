"""Vector store module for Academic Research Agent.
Manages ChromaDB local persistence, embeddings, and similarity search.
"""

import os
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "db"

_vector_store_instance = None


def _embedding_profile() -> str:
    """Returns a stable profile name for the active embedding backend."""
    if os.getenv("GOOGLE_API_KEY"):
        return f"google:{os.getenv('GEMINI_EMBED_MODEL', 'models/gemini-embedding-001')}"
    return "local:all-MiniLM-L6-v2-or-dummy"


def _collection_name() -> str:
    """Build a Chroma-safe collection name isolated by embedding dimensions."""
    profile_hash = hashlib.sha256(_embedding_profile().encode("utf-8")).hexdigest()[:12]
    return f"academic_papers_{profile_hash}"


def get_embedding_function():
    """Initializes and returns the embedding function.
    Prefers GoogleGenerativeAIEmbeddings if GOOGLE_API_KEY is available.
    Falls back to a lightweight local embedding for offline testing.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        embed_model = os.getenv("GEMINI_EMBED_MODEL", "models/gemini-embedding-001")
        return GoogleGenerativeAIEmbeddings(
            model=embed_model,
            google_api_key=api_key,
        )
    else:
        # Fallback to HuggingFace or basic local embeddings if key is missing
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        except Exception:
            # Fake/mock embedding function for graceful testing without credentials
            from langchain_core.embeddings import Embeddings

            class DummyEmbeddings(Embeddings):
                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    return [[0.0] * 384 for _ in texts]

                def embed_query(self, text: str) -> List[float]:
                    return [0.0] * 384

            return DummyEmbeddings()


def get_vector_store():
    """Returns singleton Chroma vector store instance."""
    global _vector_store_instance
    if _vector_store_instance is None:
        from langchain_chroma import Chroma

        DB_DIR.mkdir(parents=True, exist_ok=True)
        embeddings = get_embedding_function()
        _vector_store_instance = Chroma(
            collection_name=_collection_name(),
            embedding_function=embeddings,
            persist_directory=str(DB_DIR),
        )
    return _vector_store_instance


def reset_vector_store():
    """Resets the singleton vector store instance (useful for tests or reloads)."""
    global _vector_store_instance
    _vector_store_instance = None


def add_paper_to_db(doi: str, chunks: List[str]) -> bool:
    """Stores text chunks in ChromaDB with metadata {"doi": doi}.

    Args:
        doi: Digital Object Identifier of the paper.
        chunks: List of text chunk strings to index.

    Returns:
        bool: True if chunks were successfully added, False otherwise.
    """
    if not chunks:
        return False

    try:
        from langchain_core.documents import Document

        store = get_vector_store()
        documents = [
            Document(
                page_content=chunk,
                metadata={"doi": doi.strip(), "chunk_id": idx}
            )
            for idx, chunk in enumerate(chunks)
        ]
        store.add_documents(documents)
        return True
    except Exception as e:
        print(f"[vector_store] Error adding paper {doi} to ChromaDB: {e}")
        return False


def query_db(query: str, k: int = 5, doi_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Performs similarity search in ChromaDB, optionally filtering by DOI.

    Args:
        query: User search query.
        k: Maximum number of relevant chunks to retrieve.
        doi_filter: Optional DOI string to filter chunks belonging to a specific paper.

    Returns:
        List of dicts containing text, doi, and metadata.
    """
    try:
        store = get_vector_store()
        filter_dict = {"doi": doi_filter.strip()} if doi_filter else None

        docs = store.similarity_search(query, k=k, filter=filter_dict)
        results = []
        for doc in docs:
            results.append({
                "text": doc.page_content,
                "doi": doc.metadata.get("doi", doi_filter or "unknown"),
                "chunk_id": doc.metadata.get("chunk_id", -1)
            })
        return results
    except Exception as e:
        print(f"[vector_store] Error querying ChromaDB: {e}")
        return []
