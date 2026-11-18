"""PDF loading utilities.

Uses pypdf directly (no deprecated community loader) and normalises metadata
so every chunk can be traced back to the file and page it came from.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader


class PDFLoadError(ValueError):
    """Raised when a PDF cannot be read or contains no extractable text."""


def load_pdf(file_path: str | Path, display_name: str | None = None) -> list[Document]:
    """Load a PDF and return one LangChain ``Document`` per page.

    Args:
        file_path: Path to the PDF on disk.
        display_name: Name shown in citations. Defaults to the file name.

    Returns:
        Documents with ``metadata["source"]`` set to ``display_name`` and
        ``metadata["page"]`` set to the zero-based page index.

    Raises:
        FileNotFoundError: If the file does not exist.
        PDFLoadError: If parsing fails or no text can be extracted.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    name = display_name or path.name

    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            # Many PDFs are encrypted with an empty owner password; try that
            # first before giving up on the file.
            try:
                reader.decrypt("")
            except Exception as exc:  # noqa: BLE001 - normalise to our error type
                raise PDFLoadError(f"{name} is password-protected and cannot be read.") from exc
        pages = [page.extract_text() or "" for page in reader.pages]
    except PDFLoadError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface a friendly, typed error
        raise PDFLoadError(f"Could not read {name}: {exc}") from exc

    if not pages:
        raise PDFLoadError(f"{name} contains no readable pages.")

    documents = [
        Document(page_content=text, metadata={"source": name, "page": index})
        for index, text in enumerate(pages)
    ]

    if not any(document.page_content.strip() for document in documents):
        raise PDFLoadError(
            f"{name} contains no extractable text. Scanned PDFs need OCR before upload."
        )

    return documents
