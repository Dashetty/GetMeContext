import pytest

from utils.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_LLM_MODEL, get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_defaults_when_nothing_is_configured(monkeypatch):
    for name in [
        "GOOGLE_API_KEY",
        "GOOGLE_LLM_MODEL",
        "GOOGLE_EMBEDDING_MODEL",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "TOP_K",
        "CONVERSATION_WINDOW",
        "CHROMA_PERSIST_DIR",
        "CHROMA_COLLECTION",
    ]:
        monkeypatch.delenv(name, raising=False)

    settings = get_settings()

    assert settings.llm_model == DEFAULT_LLM_MODEL
    assert settings.embedding_model == DEFAULT_EMBEDDING_MODEL
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 200
    assert settings.top_k == 5
    assert settings.conversation_window == 3
    assert settings.has_api_key is False


def test_environment_overrides(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_LLM_MODEL", "models/gemini-test")
    monkeypatch.setenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-test")
    monkeypatch.setenv("TOP_K", "9")
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("CONVERSATION_WINDOW", "0")

    settings = get_settings()

    assert settings.llm_model == "models/gemini-test"
    assert settings.embedding_model == "models/embedding-test"
    assert settings.top_k == 9
    assert settings.chunk_size == 500
    assert settings.conversation_window == 0
    assert settings.has_api_key is True


def test_placeholder_api_key_is_not_treated_as_configured(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "your_google_api_key_here")

    assert get_settings().has_api_key is False


def test_invalid_integer_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("TOP_K", "not-a-number")

    assert get_settings().top_k == 5
