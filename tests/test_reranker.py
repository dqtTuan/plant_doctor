from plant_doctor.rag.reranker import load_reranker


def test_missing_model_returns_none(tmp_path):
    assert load_reranker(tmp_path) is None


def test_pipeline_works_without_reranker(monkeypatch, tmp_path):
    import plant_doctor.pipeline as pl

    monkeypatch.setattr(pl, "RERANKER_DIR", tmp_path)
    p = pl.PlantDoctorPipeline(load_models=False)
    assert p.load_reranker() is None
