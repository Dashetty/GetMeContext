"""
app.py — GetMeContext Main Application
Streamlit frontend for the RAG-powered PDF Q&A system.
"""

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Handle potential import issues gracefully
try:
    from utils.pdf_loader import load_pdf
    from utils.vector_store import create_vector_store, clear_vector_store
    from utils.rag_chain import create_rag_chain
    IMPORTS_OK = True
except ImportError as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)


# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="GetMeContext",
    page_icon="G",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
    <style>
    .main-title { font-size: 2.5rem; font-weight: 700; color: #1f77b4; margin-bottom: 0.5rem; }
    .subtitle { font-size: 1.1rem; color: #666; margin-bottom: 2rem; }
    .success-box { background-color: #d4edda; color: #155724; padding: 1rem; border-radius: 8px; border-left: 4px solid #28a745; }
    .info-box { background-color: #e7f3ff; color: #004085; padding: 1rem; border-radius: 8px; border-left: 4px solid #1f77b4; }
    .error-box { background-color: #f8d7da; color: #721c24; padding: 1rem; border-radius: 8px; border-left: 4px solid #dc3545; }
    </style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### GetMeContext")
    st.markdown("---")
    st.markdown("""
    **RAG-Powered PDF Q&A**

    Upload any PDF and ask questions based only on its content.

    ---
    **Examples:**
    - "What font size for subheadings?"
    - "Summarize Module 3"
    - "What is the submission deadline?"

    ---
    **Stack:** Streamlit • LangChain • ChromaDB • Gemini
    """)

    st.markdown("---")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_google_api_key_here":
        st.error("GOOGLE_API_KEY not configured.")
        st.info("Add your key to `.env` and restart.")
    else:
        st.success("API key configured")


# ──────────────────────────────────────────────
# Main Content
# ──────────────────────────────────────────────
st.markdown('<div class="main-title">GetMeContext</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask anything about your PDF - get grounded answers</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Import Error Banner (if any)
# ──────────────────────────────────────────────
if not IMPORTS_OK:
    st.markdown(
        f'<div class="error-box">'
        f'<b>Import Error:</b> {IMPORT_ERROR}<br><br>'
        f'Try these fixes:<br>'
        f'1. <code>pip install --upgrade langchain-chroma chromadb email-validator</code><br>'
        f'2. Restart your Python kernel / terminal<br>'
        f'3. If on Python 3.13, ensure you have the latest chromadb (&gt;=0.5.0)'
        f'</div>',
        unsafe_allow_html=True
    )
    st.stop()


# ──────────────────────────────────────────────
# Session State
# ──────────────────────────────────────────────
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "current_pdf_name" not in st.session_state:
    st.session_state.current_pdf_name = None


# ──────────────────────────────────────────────
# PDF Upload
# ──────────────────────────────────────────────
st.markdown("---")
st.subheader("1. Upload Your Document")

uploaded_file = st.file_uploader(
    "Drag and drop a PDF file here",
    type=["pdf"],
    accept_multiple_files=False,
)

if uploaded_file is not None:
    is_new_file = st.session_state.current_pdf_name != uploaded_file.name

    if is_new_file:
        st.session_state.chat_history = []
        st.session_state.pdf_processed = False
        st.session_state.current_pdf_name = uploaded_file.name
        clear_vector_store()
        st.session_state.vector_store = None
        st.session_state.rag_chain = None

    if not st.session_state.pdf_processed:
        with st.spinner("Processing PDF... Please wait"):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                documents = load_pdf(tmp_path)
                vector_store = create_vector_store(documents)
                st.session_state.vector_store = vector_store

                retriever = vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5}
                )
                rag_chain = create_rag_chain(retriever)
                st.session_state.rag_chain = rag_chain

                st.session_state.pdf_processed = True
                os.unlink(tmp_path)

                st.markdown(
                    f'<div class="success-box">'
                    f'<b>{uploaded_file.name}</b> processed.<br>'
                    f'{len(documents)} pages indexed'
                    f'</div>',
                    unsafe_allow_html=True
                )

            except Exception as e:
                st.error(f"Processing failed: {e}")
                st.info("Ensure PDF is not corrupted or password-protected.")
    else:
        st.markdown(
            f'<div class="info-box">'
            f'<b>{uploaded_file.name}</b> loaded. Ask questions below.'
            f'</div>',
            unsafe_allow_html=True
        )


# ──────────────────────────────────────────────
# Q&A Section
# ──────────────────────────────────────────────
if st.session_state.pdf_processed and st.session_state.rag_chain is not None:
    st.markdown("---")
    st.subheader("2. Ask Your Questions")

    if st.session_state.chat_history:
        st.markdown("##### Conversation")
        for i, (question, answer) in enumerate(st.session_state.chat_history, 1):
            with st.chat_message("user"):
                st.markdown(f"**Q{i}:** {question}")
            with st.chat_message("assistant"):
                st.markdown(answer)
        st.markdown("---")

    question = st.chat_input("Ask something about your document...")

    if question:
        with st.chat_message("user"):
            st.markdown(f"**Q:** {question}")

        with st.spinner("Thinking..."):
            try:
                answer = st.session_state.rag_chain.invoke(question)
            except Exception as e:
                answer = f"Error: {e}"

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.chat_history.append((question, answer))
        st.rerun()

if not st.session_state.pdf_processed:
    st.markdown("---")
    st.info("Upload a PDF to get started.")
    col1, col2, col3 = st.columns(3)
    with col1: st.markdown("**Upload**"); st.caption("Any PDF")
    with col2: st.markdown("**Retrieve**"); st.caption("Smart matching")
    with col3: st.markdown("**Answer**"); st.caption("Grounded in context")
