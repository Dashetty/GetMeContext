from pathlib import Path

from streamlit.testing.v1 import AppTest

from utils.config import get_settings

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_app_renders_without_exceptions():
    at = AppTest.from_file(str(APP), default_timeout=60).run()

    assert not at.exception
    assert at.get("file_uploader")


def test_missing_key_shows_setup_screen(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    get_settings.cache_clear()
    try:
        at = AppTest.from_file(str(APP), default_timeout=60).run()

        assert not at.exception
        assert any("Google API key not configured" in error.value for error in at.error)
    finally:
        get_settings.cache_clear()
