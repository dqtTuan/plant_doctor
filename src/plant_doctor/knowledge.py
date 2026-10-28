"""Kho tri thức: tra cứu bệnh theo nhãn YOLO và định dạng văn bản từ diseases.json."""
import difflib
import json
import re
from functools import lru_cache

from plant_doctor.config import KB_JSON

SECTIONS = [
    ("symptoms", "Triệu chứng"),
    ("causes", "Nguyên nhân & điều kiện phát sinh"),
    ("bio_treatment", "Điều trị sinh học / thân thiện môi trường"),
    ("chemical_note", "Lưu ý về thuốc hóa học"),
    ("prevention", "Phòng ngừa"),
]


# ---------------------------------------------------------------- Kho tri thức
def normalize(name: str) -> str:
    """'Corn_(maize)___Common_rust_' -> 'corn_maize_common_rust'."""
    s = re.sub(r"[^0-9a-z]+", "_", name.lower())
    return re.sub(r"_+", "_", s).strip("_")


@lru_cache(maxsize=1)
def load_kb() -> list:
    return json.loads(KB_JSON.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _alias_index() -> dict:
    index = {}
    for e in load_kb():
        for alias in [e["key"], e["name_en"], *e.get("aliases", [])]:
            index.setdefault(normalize(alias), e)
    return index


def is_healthy(class_name: str) -> bool:
    return "healthy" in normalize(class_name).split("_")


def lookup(class_name: str):
    """Ánh xạ nhãn YOLO -> mục trong kho tri thức (khớp chính xác, rồi khớp gần đúng)."""
    n = normalize(class_name)
    index = _alias_index()
    if is_healthy(n):
        return index.get("healthy")
    if n in index:
        return index[n]
    match = difflib.get_close_matches(n, list(index), n=1, cutoff=0.75)
    return index[match[0]] if match else None


# ---------------------------------------------------------------- Định dạng
def title(e: dict) -> str:
    return f"{e['name_vi']} ({e['name_en']}) – {e['crop_vi']} ({e['crop_en']})"


def section_text(e: dict, field: str) -> str:
    label = dict(SECTIONS)[field]
    return f"{title(e)}\n{label}: {e[field]}"


def entry_to_text(e: dict) -> str:
    """Văn bản đầy đủ của một bệnh – dùng làm tài liệu trong FAISS."""
    lines = [f"Bệnh: {title(e)}", f"Tác nhân: {e['pathogen']}"]
    lines += [f"{label}: {e[field]}" for field, label in SECTIONS]
    return "\n".join(lines)


def entry_to_answer(e: dict, focus: str = "full") -> str:
    """Câu trả lời mẫu có cấu trúc (dùng làm nhãn huấn luyện LLM và chế độ dự phòng)."""
    parts = []
    if focus in ("full", "symptoms"):
        parts.append(f"**Triệu chứng:** {e['symptoms']}")
    if focus in ("full", "causes"):
        parts.append(f"**Nguyên nhân:** {e['pathogen']}. {e['causes']}")
    if focus in ("full", "treatment"):
        parts.append(f"**Phác đồ sinh học:** {e['bio_treatment']}")
        parts.append(f"**Thuốc hóa học (chỉ khi thật cần thiết):** {e['chemical_note']}")
    if focus in ("full", "treatment", "prevention"):
        parts.append(f"**Phòng ngừa:** {e['prevention']}")
    return f"**{title(e)}**\n\n" + "\n\n".join(parts)

