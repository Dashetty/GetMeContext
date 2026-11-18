"""Vector store management.

Pipeline: split documents into overlapping chunks → embed with Gemini →
persist in ChromaDB. The app keeps a single collection that is rebuilt
whenever the set of uploaded PDFs changes.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.config import Settings, get_settings


class GeminiEmbeddings(Embeddings):
    """Asymmetric Gemini embeddings.

    Gemini optimises retrieval when documents and queries use different task
    types. Chroma only accepts a single embedding function, so this wrapper
    routes each call to the matching task type.
    """

    def __init__(
        self,
        model: str,
        document_embeddings: Embeddings | None = None,
        query_embeddings: Embeddings | None = None,
    ):
        self._document = document_embeddings or GoogleGenerativeAIEmbeddings(
            model=model, task_type="retrieval_document"
        )
        self._query = query_embeddings or GoogleGenerativeAIEmbeddings(
            model=model, task_type="retrieval_query"
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._document.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._query.embed_query(text)


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    """Initialise the Gemini embedding model."""
    settings = settings or get_settings()
    return GeminiEmbeddings(settings.embedding_model)


def split_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    """Split documents into overlapping chunks, preserving page metadata."""
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vector_store(
    documents: List[Document],
    persist_dir: Optional[str | Path] = None,
    collection_name: Optional[str] = None,
) -> Tuple[Chroma, List[Document]]:
    """Chunk, embed, and persist documents in ChromaDB.

    Returns:
        A tuple of ``(vector_store, chunks)`` so callers can report how many
        chunks were indexed.
    """
    settings = get_settings()
    if not documents:
        raise ValueError("No documents to index.")

    chunks = split_documents(documents)
    if not chunks:
        raise ValueError("No text chunks could be created from the uploaded PDFs.")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(settings),
        persist_directory=str(persist_dir or settings.persist_dir),
        collection_name=collection_name or settings.collection_name,
    )
    return vector_store, chunks


def clear_vector_store(
    persist_dir: Optional[str | Path] = None,
    collection_name: Optional[str] = None,
) -> None:
    """Delete all persisted vectors so the next upload starts clean."""
    settings = get_settings()
    path = Path(persist_dir or settings.persist_dir)
    collection = collection_name or settings.collection_name

    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return

    # Best-effort collection drop first; this keeps the SQLite catalogue tidy
    # even if a file lock prevents removing the directory in the same call.
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(path))
        try:
            client.delete_collection(name=collection)
        except Exception:  # noqa: BLE001 - missing collection is fine
            pass
    except Exception:  # noqa: BLE001 - chromadb unavailable/corrupt
        pass

    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
