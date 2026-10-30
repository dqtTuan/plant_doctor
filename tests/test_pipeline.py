"""Kiểm thử cho lớp PlantDoctorPipeline trong core service engine."""
from unittest.mock import MagicMock
from plant_doctor.pipeline import PlantDoctorPipeline


def test_pipeline_fallback_answer():
    pipeline = PlantDoctorPipeline(load_models=False)
    fake_entry = {
        "name_vi": "Gỉ sắt", "name_en": "Rust", "crop_vi": "Cà phê", "crop_en": "Coffee",
        "pathogen": "Nấm Hemileia vastatrix", "symptoms": "Đốm vàng cam", "causes": "Mưa ẩm",
        "bio_treatment": "Phun dung dịch Bordeaux", "chemical_note": "Hạn chế", "prevention": "Tỉa cành thông thoáng"
    }
    diagnosis = [("Rust", 0.95, fake_entry)]
    res = pipeline.fallback_answer(diagnosis, [])
    assert "Gỉ sắt" in res
    assert "Phác đồ sinh học" in res


def test_pipeline_retrieve_without_store():
    pipeline = PlantDoctorPipeline(load_models=False)
    docs = pipeline.retrieve_context("cách trị", ["coffee_leaf_rust"])
    assert len(docs) > 0
    doc_str = docs[0] if isinstance(docs[0], str) else docs[0].page_content
    assert "coffee" in doc_str.lower() or "cà phê" in doc_str.lower()


def test_pipeline_default_optimization_constants():
    from plant_doctor.config import DEFAULT_MAX_CONTEXT_DOCS, DEFAULT_MAX_NEW_TOKENS
    assert DEFAULT_MAX_CONTEXT_DOCS == 2
    assert DEFAULT_MAX_NEW_TOKENS == 256


def test_get_cached_answer_for_standard_query():
    pipeline = PlantDoctorPipeline(load_models=False)
    fake_entry = {
        "name_vi": "Gỉ sắt", "name_en": "Rust", "crop_vi": "Cà phê", "crop_en": "Coffee",
        "pathogen": "Nấm", "symptoms": "Bột vàng cam", "causes": "Mưa ẩm",
        "bio_treatment": "Phun Bordeaux", "chemical_note": "Hạn chế", "prevention": "Tỉa cành"
    }
    diagnosis = [("Rust", 0.92, fake_entry)]
    auto_query = ("Mô hình YOLOv8 vừa phát hiện lá cây bị bệnh: Rust. Hãy giải thích "
                  "triệu chứng, nguyên nhân gây bệnh và cách điều trị sinh học/phòng ngừa.")
    cached = pipeline.get_cached_answer(auto_query, diagnosis)
    assert cached is not None
    assert "Gỉ sắt" in cached
    assert "Phác đồ sinh học" in cached


def test_get_cached_answer_returns_none_for_free_question():
    pipeline = PlantDoctorPipeline(load_models=False)
    diagnosis = [("Rust", 0.92, None)]
    cached = pipeline.get_cached_answer("Hôm nay trời đẹp không?", diagnosis)
    assert cached is None
