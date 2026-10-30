"""Kiểm thử nhanh phần không cần GPU/mô hình: kho tri thức và prompt."""
from plant_doctor.knowledge import entry_to_answer, entry_to_text, is_healthy, load_kb, lookup, normalize
from plant_doctor.prompts import SYSTEM_PROMPT, build_messages, format_context


def test_normalize():
    assert normalize("Corn_(maize)___Common_rust_") == "corn_maize_common_rust"


def test_kb_keys_unique():
    keys = [e["key"] for e in load_kb()]
    assert len(keys) == len(set(keys))


def test_lookup_exact_and_healthy():
    assert lookup("Tomato___Late_blight")["key"] == "tomato_late_blight"
    assert is_healthy("Apple___healthy")
    assert lookup("Apple___healthy")["key"] == "healthy"


def test_lookup_yolo_classes():
    # Nhãn của dataset YOLO hiện tại (BRACOL – cà phê) phải tra được trong kho tri thức
    for name in ["Cercospora", "Miner", "Phoma", "Rust"]:
        assert lookup(name) is not None, name


def test_formatting():
    e = load_kb()[1]
    assert e["name_vi"] in entry_to_text(e)
    assert "**Phòng ngừa:**" in entry_to_answer(e)


def test_build_messages():
    msgs = build_messages(format_context(["doc"]), "hỏi?", [("q1", "a1")])
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert "[1] doc" in msgs[-1]["content"]


def test_system_prompt_general_and_farming():
    assert "chuyên gia bệnh học thực vật" in SYSTEM_PROMPT.lower()
    assert "chào hỏi" in SYSTEM_PROMPT.lower() or "đời sống" in SYSTEM_PROMPT.lower()
