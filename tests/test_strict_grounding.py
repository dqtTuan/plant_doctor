import pytest
from plant_doctor.pipeline import PlantDoctorPipeline
from plant_doctor.prompts import SYSTEM_PROMPT


def test_system_prompt_enforces_strict_grounding():
    prompt_lower = SYSTEM_PROMPT.lower()
    assert "chỉ sử dụng thông tin trong tài liệu" in prompt_lower or "căn cứ vào tài liệu" in prompt_lower
    assert "tuyệt đối không" in prompt_lower or "không tự suy diễn" in prompt_lower or "từ chối" in prompt_lower


def test_fallback_answer_when_no_user_documents():
    pipeline = PlantDoctorPipeline(load_models=False)
    diag = [("Coffee___Rust", 0.9, {"display_name": "Bệnh gỉ sắt cà phê"})]
    ans = pipeline.fallback_answer(diag, texts=[])
    assert "chưa có tài liệu" in ans.lower() or "vui lòng nạp thêm" in ans.lower()
