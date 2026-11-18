from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from utils.config import get_settings

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_app_renders_uploader_when_key_is_configured(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    at = AppTest.from_file(str(APP), default_timeout=60).run()

    assert not at.exception
    assert at.get("file_uploader")


def test_missing_key_shows_setup_screen(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    at = AppTest.from_file(str(APP), default_timeout=60).run()

    assert not at.exception
    assert any("Google API key not configured" in error.value for error in at.error)
