"""Central configuration for GetMeContext.

All tunables live here so the app, the RAG pipeline, and the tests share a
single source of truth. Values are resolved in this order:

1. Environment variables (local development via ``.env``)
2. Streamlit secrets (deployment on Streamlit Community Cloud)
3. Defaults defined below
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Load .env once, at import time. Safe to call in every entry point.
load_dotenv()

DEFAULT_LLM_MODEL = "gemini-3.5-flash-lite"
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"


def _streamlit_secret(name: str) -> str | None:
    """Read a secret from Streamlit's secrets store when running on Cloud.

    Imported lazily so the utils package stays usable outside Streamlit
    (tests, scripts, CI) where ``st.secrets`` raises on access.
    """
    try:
        import streamlit as st

        value = st.secrets.get(name)
    except Exception:
        return None
    return str(value) if value else None


def get_setting(name: str, default: str | None = None) -> str | None:
    """Resolve a setting from environment, then Streamlit secrets, then default."""
    value = os.getenv(name)
    if value:
        return value

    secret = _streamlit_secret(name)
    if secret:
        # Mirror into os.environ so libraries reading env vars directly
        # (e.g. langchain-google-genai's GOOGLE_API_KEY lookup) still see it.
        os.environ[name] = secret
        return secret

    return default


def _get_int(name: str, default: int) -> int:
    raw = get_setting(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    raw = get_setting(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the application configuration."""

    google_api_key: str | None
    llm_model: str
    embedding_model: str
    temperature: float
    chunk_size: int
    chunk_overlap: int
    top_k: int
    conversation_window: int
    persist_dir: Path
    collection_name: str
    max_upload_mb: int

    @property
    def has_api_key(self) -> bool:
        key = (self.google_api_key or "").strip()
        return bool(key) and key != "your_google_api_key_here"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings.

    Call ``get_settings.cache_clear()`` in tests after changing environment
    variables.
    """
    return Settings(
        google_api_key=get_setting("GOOGLE_API_KEY"),
        llm_model=get_setting("GOOGLE_LLM_MODEL", DEFAULT_LLM_MODEL) or DEFAULT_LLM_MODEL,
        embedding_model=(
            get_setting("GOOGLE_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
            or DEFAULT_EMBEDDING_MODEL
        ),
        temperature=_get_float("LLM_TEMPERATURE", 0.1),
        chunk_size=_get_int("CHUNK_SIZE", 1000),
        chunk_overlap=_get_int("CHUNK_OVERLAP", 200),
        top_k=_get_int("TOP_K", 5),
        conversation_window=_get_int("CONVERSATION_WINDOW", 3),
        persist_dir=Path(get_setting("CHROMA_PERSIST_DIR", "chroma_db") or "chroma_db"),
        collection_name=get_setting("CHROMA_COLLECTION", "docquery_docs") or "docquery_docs",
        max_upload_mb=_get_int("MAX_UPLOAD_MB", 50),
    )
