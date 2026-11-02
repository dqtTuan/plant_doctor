"""Kiểm tra cấu trúc và các hàm cốt lõi của streamlit_app."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_streamlit_app_has_get_pipeline():
    src = (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    func_names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    assert "get_pipeline" in func_names


def test_streamlit_app_answer_function_exists():
    src = (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    func_names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    assert "answer" in func_names
