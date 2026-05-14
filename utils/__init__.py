"""
utils package for GetMeContext.
Contains modular components for PDF loading, vector storage, and RAG chain.
"""

from .pdf_loader import load_pdf
from .vector_store import (
    create_vector_store,
    get_vector_store,
    similarity_search,
    clear_vector_store,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RESULTS,
)
from .rag_chain import (
    get_llm,
    create_rag_chain,
    answer_question,
    format_docs,
)

__all__ = [
    "load_pdf",
    "create_vector_store",
    "get_vector_store",
    "similarity_search",
    "clear_vector_store",
    "get_llm",
    "create_rag_chain",
    "answer_question",
    "format_docs",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "TOP_K_RESULTS",
]
