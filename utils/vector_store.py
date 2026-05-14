"""
vector_store.py
Manages ChromaDB vector database operations.
Handles embedding generation, storage, and similarity search for RAG retrieval.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

# Try multiple import paths for Chroma to handle different package versions
try:
    # Primary: langchain-chroma package (recommended for 2026)
    from langchain_chroma import Chroma
except ImportError:
    try:
        # Fallback: langchain_community vectorstores
        from langchain_community.vectorstores import Chroma
    except ImportError:
        # Last resort: direct chromadb with LangChain wrapper
        import chromadb
        from chromadb.config import Settings

        class Chroma:
            """Minimal compatibility wrapper if langchain-chroma fails."""
            pass


# Configuration constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RESULTS = 5
CHROMA_PERSIST_DIR = Path("chroma_db")


def _select_embedding_model() -> str:
    """Choose a supported embedding model name.

    Priority:
    1) Explicit env override
    2) First available embedContent-capable model from API
    """
    env_model = os.getenv("GOOGLE_EMBEDDING_MODEL")
    if env_model:
        return env_model

    try:
        import google.generativeai as genai

        models = genai.list_models()
        for model in models:
            name = getattr(model, "name", "")
            methods = set(getattr(model, "supported_generation_methods", []) or [])
            if "embedContent" in methods and "embedding" in name:
                return name
    except Exception:
        pass

    raise RuntimeError(
        "No embedding model available. Set GOOGLE_EMBEDDING_MODEL in .env "
        "to a model that supports embedContent."
    )


def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """Initialize Google's Gemini embedding model."""
    return GoogleGenerativeAIEmbeddings(
        model=_select_embedding_model(),
        task_type="retrieval_document"
    )


def create_vector_store(
    documents: List[Document],
    collection_name: str = "docquery_docs"
) -> Chroma:
    """
    Create a new ChromaDB vector store from documents.
    Steps: chunk → embed → store in persistent ChromaDB.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    embeddings = get_embedding_model()

    # Split documents into overlapping chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    # Create vector store with persistent storage
    # Note: persist_directory must be a string, not Path object
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_PERSIST_DIR),
        collection_name=collection_name
    )

    return vector_store


def get_vector_store(collection_name: str = "docquery_docs") -> Optional[Chroma]:
    """Load an existing ChromaDB vector store from disk."""
    embeddings = get_embedding_model()

    try:
        return Chroma(
            persist_directory=str(CHROMA_PERSIST_DIR),
            embedding_function=embeddings,
            collection_name=collection_name
        )
    except Exception:
        return None


def similarity_search(
    vector_store: Chroma,
    query: str,
    k: int = TOP_K_RESULTS
) -> List[Document]:
    """Retrieve top-k most relevant document chunks for a query."""
    results = vector_store.similarity_search(query, k=k)
    return results


def clear_vector_store() -> None:
    """Delete all data from the vector store to start fresh."""
    if CHROMA_PERSIST_DIR.exists():
        try:
            shutil.rmtree(CHROMA_PERSIST_DIR)
        except OSError:
            # Fallback in case of lingering files/locks during rapid reruns
            shutil.rmtree(CHROMA_PERSIST_DIR, ignore_errors=True)
        CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
