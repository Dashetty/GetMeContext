# GetMeContext - RAG-Powered PDF Q&A

A simple, polished full-stack RAG (Retrieval-Augmented Generation) application that lets you upload any PDF and ask questions based **only** on its content.

## Use Cases

- **Students**: Query lecture notes, ask "Summarize Module 3" or "What is the submission deadline?"
- **Professionals**: Quickly understand lengthy reports, guidelines, or contracts
- **Researchers**: Extract specific details without reading the entire document

## Architecture

```
project/
├── app.py                  # Streamlit frontend
├── requirements.txt        # Dependencies
├── .env                    # API keys (not tracked)
├── .env.example            # Template for env vars
├── utils/
│   ├── pdf_loader.py       # PDF text extraction
│   ├── vector_store.py     # ChromaDB embedding storage
│   └── rag_chain.py        # Retrieval + LLM pipeline
├── data/                   # Uploaded PDFs
├── chroma_db/              # Persistent vector database
└── README.md
```

## Quick Start

### 1. Clone & Setup

```bash
git clone <repo-url>
cd GetMeContext
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure API Key

```bash
cp .env.example .env
# Edit .env and add your Google API Key:
# GOOGLE_API_KEY=your_key_here
# Optional model overrides:
# GOOGLE_EMBEDDING_MODEL=models/gemini-embedding-001
# GOOGLE_LLM_MODEL=models/gemini-2.0-flash
```

Get your key free at: [Google AI Studio](https://aistudio.google.com/app/apikey)

### 3. Run the App

```bash
streamlit run app.py
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| PDF Parsing | PyPDF2 |
| Text Splitting | RecursiveCharacterTextSplitter |
| Embeddings | Gemini embeddings (configurable) |
| Vector DB | ChromaDB (persistent) |
| LLM | Gemini (configurable) |
| Orchestration | LangChain |

## Features

- **PDF Upload**: Drag-and-drop any PDF document
- **Smart Retrieval**: Similarity search with top-k chunk retrieval
- **Grounded Answers**: Responses strictly from uploaded document context
- **Safety First**: "The answer is not available in the uploaded document" when context is insufficient
- **Multi-turn Q&A**: Ask unlimited follow-up questions after upload
- **Loading States**: Visual feedback during processing
- **Clean UI**: Minimal, focused interface with sidebar info

## Example Questions

- "What font size should subheadings use?"
- "Summarize Module 3"
- "What is the submission deadline?"
- "List all the key findings from section 2"

## Security Notes

- Never commit `.env` files — they are `.gitignore`d
- API keys are loaded at runtime, not hardcoded
- ChromaDB runs locally; no data leaves your machine except to Google's API

## Learning Resources

This project is designed to be **interview-ready** and **beginner-friendly**:
- Modular code structure for easy testing
- Clear comments explaining each step
- Separation of concerns: UI ↔ logic ↔ data

---

Built with LangChain + Gemini + ChromaDB + Streamlit
