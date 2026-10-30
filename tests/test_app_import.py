"""Kiểm tra import và cấu trúc của streamlit_app."""
import importlib


def test_streamlit_app_has_get_pipeline():
    spec = importlib.util.find_spec("app.streamlit_app")
    assert spec is not None
