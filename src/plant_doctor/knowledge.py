"""Kho tri thức và chuẩn hóa nhãn bệnh cây trồng (hoàn toàn không phụ thuộc vào file JSON).

Tự động phân tích nhãn YOLO và định dạng văn bản phục vụ hiển thị và suy luận.
"""
import re
from plant_doctor.rag.query_expansion import parse_label, normalize_label

SECTIONS = [
    ("symptoms", "Triệu chứng"),
    ("causes", "Nguyên nhân & điều kiện phát sinh"),
    ("bio_treatment", "Điều trị sinh học / thân thiện môi trường"),
    ("chemical_note", "Lưu ý về thuốc hóa học"),
    ("prevention", "Phòng ngừa"),
]


def normalize(name: str) -> str:
    """'Corn_(maize)___Common_rust_' -> 'corn_maize_common_rust'."""
    return normalize_label(name)


def load_kb() -> list:
    """Trả về danh sách rỗng vì tri thức đã được chuyển đổi hoàn toàn sang tài liệu Markdown/PDF."""
    return []


def is_healthy(class_name: str) -> bool:
    """Kiểm tra xem nhãn nhận diện có phải là cây khỏe mạnh hay không."""
    return "healthy" in normalize(class_name).split("_")


def lookup(class_name: str) -> dict:
    """Ánh xạ nhãn YOLO -> thực thể bệnh học bằng thuật toán phân tích nhãn động."""
    info = parse_label(class_name)
    # Tương thích với các khóa cũ (name_vi, name_en)
    info["name_vi"] = info["disease_vi"]
    info["name_en"] = info["disease_en"]
    return info


def title(e: dict) -> str:
    """Tạo tiêu đề chuẩn hóa cho bệnh cây."""
    name_vi = e.get("name_vi") or e.get("disease_vi", "")
    name_en = e.get("name_en") or e.get("disease_en", "")
    crop_vi = e.get("crop_vi", "")
    crop_en = e.get("crop_en", "")
    return f"{name_vi} ({name_en}) – {crop_vi} ({crop_en})"


def section_text(e: dict, field: str) -> str:
    label = dict(SECTIONS).get(field, field)
    val = e.get(field, "Xem chi tiết trong cẩm nang hướng dẫn.")
    return f"{title(e)}\n{label}: {val}"


def entry_to_text(e: dict) -> str:
    """Văn bản tổng quan của một bệnh."""
    lines = [f"Bệnh: {title(e)}", f"Tác nhân: {e.get('pathogen', 'Nấm hoặc vi khuẩn gây hại')}"]
    for field, label in SECTIONS:
        if field in e:
            lines.append(f"{label}: {e[field]}")
    return "\n".join(lines)


def entry_to_answer(e: dict, focus: str = "full") -> str:
    """Câu trả lời mẫu có cấu trúc."""
    parts = []
    if "symptoms" in e and focus in ("full", "symptoms"):
        parts.append(f"**Triệu chứng:** {e['symptoms']}")
    if "causes" in e and focus in ("full", "causes"):
        parts.append(f"**Nguyên nhân:** {e.get('pathogen', '')}. {e['causes']}")
    if "bio_treatment" in e and focus in ("full", "treatment"):
        parts.append(f"**Phác đồ sinh học:** {e['bio_treatment']}")
    if "chemical_note" in e and focus in ("full", "treatment"):
        parts.append(f"**Thuốc hóa học (chỉ khi thật cần thiết):** {e['chemical_note']}")
    if "prevention" in e and focus in ("full", "treatment", "prevention"):
        parts.append(f"**Phòng ngừa:** {e['prevention']}")

    if not parts:
        parts.append(f"Thông tin chi tiết về phác đồ điều trị của {title(e)} được truy xuất tự động từ cẩm nang tài liệu chuyên ngành.")

    return f"**{title(e)}**\n\n" + "\n\n".join(parts)
