import os

from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def test_app_starts_without_errors(monkeypatch):
    # Point at an address with nothing listening so the test never needs Ollama
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "PDF Q&A with Llama 3.1"


def test_question_before_upload_shows_warning(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("What is this about?").run()
    assert not at.exception
    assert any("upload a PDF" in w.value for w in at.warning)
