from types import SimpleNamespace

from utils.helpers import (
    document_summary,
    explain_api_error,
    format_transcript,
    upload_signature,
)


def test_upload_signature_is_order_independent():
    a = SimpleNamespace(name="a.pdf", size=10)
    b = SimpleNamespace(name="b.pdf", size=20)

    assert upload_signature([a, b]) == upload_signature([b, a])


def test_upload_signature_changes_with_content():
    a = SimpleNamespace(name="a.pdf", size=10)

    assert upload_signature([a]) != upload_signature([SimpleNamespace(name="a.pdf", size=11)])


def test_format_transcript_includes_questions_answers_and_sources():
    messages = [
        {"role": "user", "content": "What is the deadline?"},
        {
            "role": "assistant",
            "content": "It is 15 May [1].",
            "sources": [{"document": "a.pdf", "page": 3}],
        },
    ]

    transcript = format_transcript(messages)

    assert "What is the deadline?" in transcript
    assert "It is 15 May [1]." in transcript
    assert "a.pdf — page 3" in transcript


def test_document_summary_pluralises_and_sums_pages():
    one = [{"name": "a.pdf", "pages": 4}]
    many = [{"name": "a.pdf", "pages": 4}, {"name": "b.pdf", "pages": 6}]

    assert document_summary(one) == "1 document · 4 pages"
    assert document_summary(many) == "2 documents · 10 pages"


def test_explain_api_error_detects_quota_errors():
    message = explain_api_error(RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded"))

    assert "quota" in message.lower()
    assert "gemini-3.5-flash-lite" in message


def test_explain_api_error_detects_invalid_keys():
    message = explain_api_error(RuntimeError("API_KEY_INVALID: API key not valid"))

    assert "API key" in message
    assert "GOOGLE_API_KEY" in message


def test_explain_api_error_detects_retired_models():
    message = explain_api_error(RuntimeError("404 model not found"))

    assert "gemini-3.5-flash-lite" in message
    assert "gemini-4-hydrogen" in message


def test_explain_api_error_falls_back_to_generic_message():
    message = explain_api_error(ValueError("something unexpected"))

    assert "request failed" in message
