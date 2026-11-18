"""GetMeContext — Streamlit frontend for RAG-powered PDF Q&A."""

from __future__ import annotations

import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import streamlit as st

from utils.config import Settings, get_settings
from utils.helpers import document_summary, format_transcript, upload_signature
from utils.pdf_loader import PDFLoadError, load_pdf
from utils.rag_chain import RAGPipeline, Source, create_rag_pipeline
from utils.vector_store import build_vector_store, clear_vector_store

st.set_page_config(
    page_title="GetMeContext — PDF Q&A",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
  .block-container { padding-top: 2.2rem; padding-bottom: 8rem; max-width: 860px; }
  .gmc-title {
      font-size: 2.6rem; font-weight: 800; letter-spacing: -0.02em;
      background: linear-gradient(90deg, #0f766e 0%, #1d4ed8 100%);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      margin-bottom: 0.15rem;
  }
  .gmc-subtitle { color: #5b6572; font-size: 1.05rem; margin-bottom: 1.6rem; }
  .gmc-step { border: 1px solid #e5e7eb; border-radius: 12px; padding: 1rem 1.1rem; height: 100%; }
  .gmc-step h4 { margin: 0 0 0.25rem 0; font-size: 0.95rem; }
  .gmc-step p { margin: 0; color: #6b7280; font-size: 0.85rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────
def init_state() -> None:
    defaults: dict[str, Any] = {
        "messages": [],
        "pipeline": None,
        "indexed_files": [],
        "chunk_count": 0,
        "upload_signature": None,
        "index_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_index(clear_store: bool = True, clear_uploads: bool = False) -> None:
    """Forget the current index and conversation."""
    if clear_store:
        clear_vector_store()
    st.session_state.pipeline = None
    st.session_state.indexed_files = []
    st.session_state.chunk_count = 0
    st.session_state.upload_signature = None
    st.session_state.index_error = None
    st.session_state.messages = []
    if clear_uploads and "pdf_uploader" in st.session_state:
        del st.session_state["pdf_uploader"]


def reset_app() -> None:
    """Widget callback: wipe the index, conversation, and uploaded files."""
    reset_index(clear_uploads=True)


# ──────────────────────────────────────────────
# Indexing
# ──────────────────────────────────────────────
def index_uploads(
    files: Sequence[Any], max_upload_mb: int
) -> tuple[list[dict[str, Any]], int, list[str]]:
    """Extract, chunk, embed, and index a set of uploaded PDFs.

    Returns ``(summaries, chunk_count, warnings)`` and raises when nothing
    could be indexed. UI feedback is left to the caller.
    """
    clear_vector_store()

    documents = []
    summaries: list[dict[str, Any]] = []
    errors: list[str] = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for position, uploaded in enumerate(files):
            safe_name = Path(uploaded.name).name or f"document-{position + 1}.pdf"
            size_mb = getattr(uploaded, "size", 0) / (1024 * 1024)
            if size_mb > max_upload_mb:
                errors.append(f"{safe_name} is larger than {max_upload_mb} MB — skipped.")
                continue

            path = Path(temp_dir) / f"{position}-{safe_name}"
            path.write_bytes(uploaded.getvalue())
            try:
                pages = load_pdf(path, display_name=safe_name)
            except (PDFLoadError, OSError) as exc:
                errors.append(str(exc))
                continue

            documents.extend(pages)
            summaries.append({"name": safe_name, "pages": len(pages)})

    if not documents:
        detail = " ".join(errors)
        raise PDFLoadError(
            "None of the uploaded files could be read." + (f" {detail}" if detail else "")
        )

    vector_store, chunks = build_vector_store(documents)
    st.session_state.pipeline = create_rag_pipeline(vector_store)
    st.session_state.indexed_files = summaries
    st.session_state.chunk_count = len(chunks)
    st.session_state.messages = []

    return summaries, len(chunks), errors


# ──────────────────────────────────────────────
# Chat
# ──────────────────────────────────────────────
def render_sources(sources: Sequence[Source | Mapping[str, Any]]) -> None:
    """Render the citation list under an answer."""
    if not sources:
        return

    with st.expander(f"Sources ({len(sources)})"):
        for source in sources:
            if isinstance(source, Source):
                index, document, page, snippet = (
                    source.index,
                    source.document,
                    source.page,
                    source.snippet,
                )
            else:
                index = source.get("index", "?")
                document = source.get("document", "document")
                page = source.get("page", 0)
                snippet = source.get("snippet", "")

            page_label = f"page {page}" if page else "page unknown"
            st.markdown(f"**[{index}] {document}** — {page_label}")
            if snippet:
                st.caption(snippet)


def render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources") or [])


def answer_question(question: str, pipeline: RAGPipeline) -> None:
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        parts: list[str] = []
        sources: list[dict[str, Any]] = []

        try:
            prepared = pipeline.prepare(question, history)
            for token in pipeline.stream(prepared):
                parts.append(token)
                placeholder.markdown("".join(parts) + "▌")

            answer = "".join(parts).strip()
            if not answer:
                raise ValueError("The model returned an empty answer.")

            placeholder.markdown(answer)
            render_sources(prepared.sources)
            sources = [asdict(source) for source in prepared.sources]
        except Exception as exc:  # noqa: BLE001 - show a friendly failure in the UI
            answer = (
                "Something went wrong while answering. Check that your Google API key "
                "is valid and your quota is not exhausted, then try again."
            )
            placeholder.error(answer)
            with st.expander("Error details"):
                st.code(f"{type(exc).__name__}: {exc}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
def render_sidebar(settings: Settings) -> None:
    with st.sidebar:
        st.markdown("### GetMeContext")
        st.caption("RAG-powered PDF Q&A")

        if settings.has_api_key:
            st.success("Google API key configured", icon="✅")
        else:
            st.error("Google API key missing", icon="🔑")

        st.divider()

        if st.session_state.indexed_files:
            st.markdown("**Indexed documents**")
            for item in st.session_state.indexed_files:
                pages = item["pages"]
                noun = "page" if pages == 1 else "pages"
                st.markdown(f"- 📄 {item['name']} · {pages} {noun}")

            left, right = st.columns(2)
            left.metric("Pages", sum(f["pages"] for f in st.session_state.indexed_files))
            right.metric("Chunks", st.session_state.chunk_count)

            st.button("Reset app", on_click=reset_app, use_container_width=True)
        else:
            st.caption("No documents indexed yet.")

        if st.session_state.messages:
            st.download_button(
                "Download conversation",
                data=format_transcript(st.session_state.messages),
                file_name="getmecontext-conversation.md",
                mime="text/markdown",
                use_container_width=True,
            )

        st.divider()
        st.caption(f"LLM: `{settings.llm_model}`")
        st.caption(f"Embeddings: `{settings.embedding_model}`")
        st.caption(
            "Uploaded PDFs are deleted after indexing; the searchable text index "
            "is local and rebuilt for each new upload."
        )


# ──────────────────────────────────────────────
# Page
# ──────────────────────────────────────────────
def render_missing_key() -> None:
    st.error("**Google API key not configured.** The app cannot run without one.")
    st.markdown(
        "**Local development** — add your key to `.env`:\n\n"
        "```bash\nGOOGLE_API_KEY=your_key_here\n```\n\n"
        "**Streamlit Community Cloud** — add it under "
        "*App settings → Secrets*:\n\n"
        '```toml\nGOOGLE_API_KEY = "your_key_here"\n```'
    )
    st.info("Get a free key at https://aistudio.google.com/app/apikey, then reload the app.")


def render_empty_state() -> None:
    st.info("Upload one or more PDFs to get started.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="gmc-step"><h4>1 · Upload</h4>'
            "<p>One or many PDFs, up to 50 MB each.</p></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="gmc-step"><h4>2 · Index</h4>'
            "<p>Text is chunked and embedded with Gemini.</p></div>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="gmc-step"><h4>3 · Ask</h4>'
            "<p>Grounded answers with page-level citations.</p></div>",
            unsafe_allow_html=True,
        )


def main() -> None:
    settings = get_settings()
    init_state()

    render_sidebar(settings)
    st.markdown('<div class="gmc-title">GetMeContext</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="gmc-subtitle">Ask questions across your PDFs — answers grounded '
        "in the source, with page-level citations.</div>",
        unsafe_allow_html=True,
    )

    if not settings.has_api_key:
        render_missing_key()
        st.stop()

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
        help=f"Up to {settings.max_upload_mb} MB per file.",
    )

    if not uploaded_files:
        if st.session_state.upload_signature is not None or st.session_state.index_error:
            reset_index()
        render_empty_state()
        return

    signature = upload_signature(uploaded_files)
    if signature != st.session_state.upload_signature:
        try:
            with st.spinner("Indexing documents… this can take a moment"):
                summaries, chunk_count, warnings = index_uploads(
                    uploaded_files, settings.max_upload_mb
                )
        except Exception as exc:  # noqa: BLE001 - surface any ingestion failure
            reset_index()
            st.session_state.upload_signature = signature
            st.session_state.index_error = str(exc)
            st.error(f"Could not index the uploads: {exc}")
            if st.button("Retry indexing"):
                reset_index()
                st.rerun()
            return

        st.session_state.upload_signature = signature
        st.session_state.index_error = None
        for warning in warnings:
            st.warning(warning)
        st.success(
            f"Indexed {document_summary(summaries)} into {chunk_count} searchable chunks."
        )
    elif st.session_state.index_error:
        st.error(f"Could not index the uploads: {st.session_state.index_error}")
        if st.button("Retry indexing"):
            reset_index()
            st.rerun()
        return

    st.divider()
    render_history()

    question = st.chat_input("Ask something about your documents…")
    if question:
        answer_question(question, st.session_state.pipeline)


main()
