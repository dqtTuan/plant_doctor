import importlib
import warnings
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]


def test_load_llm_uses_dtype_kwarg(tmp_path, monkeypatch):
    import plant_doctor.pipeline as pl

    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setattr(pl, "LLM_MERGED_DIR", tmp_path)
    fake_model = MagicMock()
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: MagicMock())
    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", fake_model)
    pl.PlantDoctorPipeline(load_models=False).load_llm()
    kwargs = fake_model.call_args.kwargs
    assert "dtype" in kwargs and "torch_dtype" not in kwargs


def test_no_torch_dtype_or_use_container_width_in_sources():
    files = list((ROOT / "src").rglob("*.py")) + [ROOT / "app" / "streamlit_app.py"]
    offenders = [
        f.name
        for f in files
        if "torch_dtype=" in f.read_text(encoding="utf-8")
        or "use_container_width" in f.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_train_embedding_imports_without_deprecation_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        importlib.reload(importlib.import_module("plant_doctor.training.train_embedding"))


def test_build_train_dataset_columns():
    from plant_doctor.training.train_embedding import build_train_dataset

    ds = build_train_dataset([("q1", "p1"), ("q2", "p2")])
    assert ds.column_names == ["anchor", "positive"]
    assert len(ds) == 2
