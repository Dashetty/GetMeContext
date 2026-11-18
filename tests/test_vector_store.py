import hashlib

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from utils.vector_store import (
    GeminiEmbeddings,
    build_vector_store,
    clear_vector_store,
    split_documents,
)


class HashEmbeddings(Embeddings):
    """Deterministic local embeddings so tests never call the Gemini API."""

    def __init__(self, size: int = 16):
        self.size = size

    def _embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [byte / 255 for byte in digest[: self.size]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def test_split_documents_preserves_metadata_and_overlaps():
    text = "Sentence about retrieval. " * 60
    document = Document(page_content=text, metadata={"source": "notes.pdf", "page": 4})

    chunks = split_documents([document], chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == "notes.pdf" for chunk in chunks)
    assert all(chunk.metadata["page"] == 4 for chunk in chunks)
    assert all(len(chunk.page_content) <= 220 for chunk in chunks)


def test_gemini_embeddings_route_documents_and_queries_separately():
    class RecordingEmbeddings(Embeddings):
        def __init__(self, value):
            self.value = value
            self.seen = []

        def embed_documents(self, texts):
            self.seen.extend(texts)
            return [[self.value] for _ in texts]

        def embed_query(self, text):
            self.seen.append(text)
            return [self.value]

    documents = RecordingEmbeddings(1.0)
    queries = RecordingEmbeddings(2.0)

    embeddings = GeminiEmbeddings(
        "test-model", document_embeddings=documents, query_embeddings=queries
    )

    assert embeddings.embed_documents(["a", "b"]) == [[1.0], [1.0]]
    assert embeddings.embed_query("q") == [2.0]
    assert documents.seen == ["a", "b"]
    assert queries.seen == ["q"]


def test_build_and_clear_vector_store(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "utils.vector_store.get_embeddings",
        lambda settings=None: HashEmbeddings(),
    )
    persist_dir = tmp_path / "chroma"
    documents = [
        Document(
            page_content="ChromaDB stores embeddings persistently on disk.",
            metadata={"source": "a.pdf", "page": 0},
        ),
        Document(
            page_content="Streamlit renders a conversational interface.",
            metadata={"source": "a.pdf", "page": 1},
        ),
    ]

    store, chunks = build_vector_store(
        documents, persist_dir=persist_dir, collection_name="test"
    )

    assert len(chunks) == 2
    assert store.similarity_search("embeddings", k=1)

    clear_vector_store(persist_dir=persist_dir, collection_name="test")

    assert persist_dir.exists()
    assert not (persist_dir / "chroma.sqlite3").exists()
