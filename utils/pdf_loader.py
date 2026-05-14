"""
pdf_loader.py
Handles PDF text extraction using PyPDF2.
Provides a clean interface to load and extract text from uploaded PDF files.
"""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file_path: str | Path) -> List:
    """
    Load a PDF file and extract all text as LangChain Document objects.

    Args:
        file_path: Path to the PDF file (string or Path object)

    Returns:
        List of Document objects, one per page, containing:
            - page_content: Extracted text
            - metadata: Page number, source file, etc.

    Raises:
        FileNotFoundError: If PDF file does not exist
        Exception: If PDF parsing fails (corrupted file, encrypted, etc.)
    """
    # Convert to Path object for consistent handling
    path = Path(file_path)

    # Validate file exists before attempting to load
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    # Use LangChain's PyPDFLoader for robust text extraction
    # It handles multi-page PDFs and returns metadata-rich Document objects
    loader = PyPDFLoader(str(path))

    # load() returns a list of Document objects (one per page)
    documents = loader.load()

    return documents
