"""Small, testable helpers shared by the Streamlit UI."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping, Sequence


def upload_signature(files: Iterable[Any]) -> str:
    """Return a stable fingerprint for a set of uploaded files.

    Uses name + size instead of bytes so re-runs triggered by unrelated UI
    events do not re-embed documents.
    """
    parts = sorted(f"{getattr(f, 'name', '')}:{getattr(f, 'size', 0)}" for f in files)
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def explain_api_error(exc: Exception) -> str:
    """Turn a provider/library exception into an actionable message."""
    text = f"{type(exc).__name__}: {exc}".lower()

    if any(token in text for token in ("resource_exhausted", "429", "quota", "rate limit")):
        return (
            "The model's free-tier quota is exhausted (HTTP 429). Wait for the "
            "per-minute window to reset, or switch GOOGLE_LLM_MODEL to a higher-quota "
            "model such as gemini-3.5-flash-lite. Daily quotas reset at midnight Pacific."
        )
    if any(
        token in text
        for token in ("api key", "api_key_invalid", "unauthenticated", "permission_denied")
    ):
        return (
            "The Google API key is missing, invalid, or lacks access to the selected "
            "model. Check GOOGLE_API_KEY in .env or Streamlit secrets."
        )
    if any(token in text for token in ("404", "not found", "not_found")):
        return (
            "The configured model is unavailable. Set GOOGLE_LLM_MODEL to a current "
            "model such as gemini-3.5-flash-lite or gemini-4-hydrogen."
        )
    return "The AI service request failed. Check the error details and try again."


def format_transcript(messages: Sequence[Mapping[str, Any]]) -> str:
    """Render the conversation as Markdown for the download button."""
    lines = ["# GetMeContext — conversation", ""]
    for message in messages:
        if message.get("role") == "user":
            lines.extend(["## Question", "", str(message.get("content", "")), ""])
        else:
            lines.extend(["### Answer", "", str(message.get("content", "")), ""])
            sources = message.get("sources") or []
            if sources:
                lines.append("**Sources**")
                lines.append("")
                for source in sources:
                    page = source.get("page") or "?"
                    lines.append(f"- {source.get('document', 'document')} — page {page}")
                lines.append("")
    return "\n".join(lines).strip() + "\n"


def document_summary(files: Sequence[Mapping[str, Any]]) -> str:
    """Pluralised summary of indexed files, e.g. ``3 documents · 42 pages``."""
    pages = sum(int(f.get("pages", 0)) for f in files)
    count = len(files)
    noun = "document" if count == 1 else "documents"
    return f"{count} {noun} · {pages} pages"
