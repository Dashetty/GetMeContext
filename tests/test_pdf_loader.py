import pytest
from langchain_core.documents import Document

from utils.pdf_loader import PDFLoadError, load_pdf


class FakePage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


class FakeReader:
    is_encrypted = False

    def __init__(self, pages=None, error=None):
        self._pages = pages or []
        self._error = error

    def __call__(self, _path):
        if self._error:
            raise self._error
        return self

    @property
    def pages(self):
        return self._pages

    def decrypt(self, _password):
        return 1


def patch_reader(monkeypatch, reader):
    monkeypatch.setattr("utils.pdf_loader.PdfReader", reader)


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pdf(tmp_path / "missing.pdf")


def test_metadata_uses_display_name_and_page_index(monkeypatch, tmp_path):
    path = tmp_path / "upload.pdf"
    path.write_bytes(b"%PDF-1.4 fake")
    patch_reader(monkeypatch, FakeReader([FakePage("page one"), FakePage("page two")]))

    pages = load_pdf(path, display_name="lecture-notes.pdf")

    assert len(pages) == 2
    assert pages[0].metadata["source"] == "lecture-notes.pdf"
    assert pages[0].metadata["page"] == 0
    assert pages[1].metadata["page"] == 1
    assert pages[0].page_content == "page one"


def test_reader_failure_is_wrapped(monkeypatch, tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"not a pdf")
    patch_reader(monkeypatch, FakeReader(error=ValueError("bad header")))

    with pytest.raises(PDFLoadError, match="Could not read broken.pdf"):
        load_pdf(path)


def test_pdf_without_text_is_rejected(monkeypatch, tmp_path):
    path = tmp_path / "scanned.pdf"
    path.write_bytes(b"%PDF-1.4 fake")
    patch_reader(monkeypatch, FakeReader([FakePage("   ")]))

    with pytest.raises(PDFLoadError, match="no extractable text"):
        load_pdf(path)


def test_empty_pdf_is_rejected(monkeypatch, tmp_path):
    path = tmp_path / "empty.pdf"
    path.write_bytes(b"%PDF-1.4 fake")
    patch_reader(monkeypatch, FakeReader(pages=[]))

    with pytest.raises(PDFLoadError, match="no readable pages"):
        load_pdf(path)


def test_documents_have_expected_shape(monkeypatch, tmp_path):
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.4 fake")
    patch_reader(monkeypatch, FakeReader([FakePage("hello")]))

    pages = load_pdf(path)

    assert all(isinstance(page, Document) for page in pages)
