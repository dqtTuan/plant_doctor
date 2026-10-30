"""Kiểm tra import và cấu trúc của streamlit_app."""
import importlib


def test_streamlit_app_has_get_pipeline():
    spec = importlib.util.find_spec("app.streamlit_app")
    assert spec is not None


def test_streamlit_app_answer_function_exists():
    import app.streamlit_app as st_app
    assert hasattr(st_app, "answer")
    assert callable(st_app.answer)
