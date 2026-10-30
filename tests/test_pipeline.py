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
