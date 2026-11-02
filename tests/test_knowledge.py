"""Kiểm thử nhanh phần không cần GPU/mô hình: kho tri thức và prompt (không phụ thuộc JSON)."""
from plant_doctor.knowledge import entry_to_answer, entry_to_text, is_healthy, load_kb, lookup, normalize
from plant_doctor.prompts import SYSTEM_PROMPT, build_messages, format_context


def test_normalize():
    assert normalize("Corn_(maize)___Common_rust_") == "corn_maize_common_rust"


def test_kb_keys_safe_without_json():
    keys = [e["key"] for e in load_kb()]
    assert len(keys) == len(set(keys))


def test_lookup_exact_and_healthy():
    assert lookup("Tomato___Late_blight")["key"] == "tomato_late_blight"
    assert is_healthy("Apple___healthy")
    assert lookup("Apple___healthy")["key"] == "apple_healthy"
    assert lookup("Apple___healthy")["is_healthy"] is True


def test_lookup_yolo_classes():
    # Nhãn của dataset YOLO bất kỳ (cà phê, lúa, v.v.) đều phân tích thành công
    for name in ["Cercospora", "Miner", "Phoma", "Rust", "Coffee___Rust"]:
        assert lookup(name) is not None, name
        assert "display_name" in lookup(name)


def test_formatting():
    e = lookup("Tomato___Early_blight")
    text = entry_to_text(e)
    assert "Cà chua" in text
    ans = entry_to_answer(e)
    assert "Cà chua" in ans


def test_build_messages():
    msgs = build_messages(format_context(["doc"]), "hỏi?", [("q1", "a1")])
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert "[1] doc" in msgs[-1]["content"]


def test_system_prompt_general_and_farming():
    assert "chuyên gia bệnh học thực vật" in SYSTEM_PROMPT.lower()
    assert "chào hỏi" in SYSTEM_PROMPT.lower() or "đời sống" in SYSTEM_PROMPT.lower()
