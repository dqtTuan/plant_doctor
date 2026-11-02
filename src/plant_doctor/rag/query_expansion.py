"""Module phân tích nhãn nhận diện YOLO và mở rộng truy vấn ngữ nghĩa (Query Expansion).

Cho phép hệ thống chuyển đổi nhãn phát hiện từ YOLO thành câu hỏi ngữ nghĩa nông nghiệp tự nhiên
để truy xuất trực tiếp từ Vector Database (FAISS / PostgreSQL pgvector) mà không phụ thuộc vào file JSON.
"""
import re

CROP_TRANSLATIONS = {
    "apple": "Táo",
    "blueberry": "Việt quất",
    "cherry": "Anh đào (Cherry)",
    "corn": "Ngô (Bắp)",
    "grape": "Nho",
    "orange": "Cam",
    "peach": "Đào",
    "pepper": "Ớt chuông",
    "potato": "Khoai tây",
    "raspberry": "Mâm xôi",
    "soybean": "Đậu nành",
    "squash": "Bí",
    "strawberry": "Dâu tây",
    "tomato": "Cà chua",
    "coffee": "Cà phê",
    "rice": "Lúa",
    "banana": "Chuối",
    "mango": "Xoài",
    "avocado": "Bơ",
    "dragonfruit": "Thanh long",
    "citrus": "Cây có múi",
    "general": "Cây trồng nói chung",
}

DISEASE_TRANSLATIONS = {
    "apple_scab": "Bệnh ghẻ táo",
    "black_rot": "Bệnh thối đen",
    "cedar_apple_rust": "Bệnh gỉ sắt táo",
    "cercospora_leaf_spot": "Bệnh đốm lá Cercospora",
    "gray_leaf_spot": "Bệnh đốm lá xám",
    "common_rust": "Bệnh gỉ sắt thông thường",
    "northern_leaf_blight": "Bệnh cháy lá viền xám",
    "black_measles": "Bệnh sởi đen",
    "leaf_blight": "Bệnh cháy lá",
    "isariopsis_leaf_spot": "Bệnh đốm lá Isariopsis",
    "haunglongbing": "Bệnh vàng lá Greening",
    "citrus_greening": "Bệnh vàng lá Greening",
    "bacterial_spot": "Bệnh đốm đốm vi khuẩn",
    "early_blight": "Bệnh đốm vòng sớm",
    "late_blight": "Bệnh mốc sương / sương mai",
    "powdery_mildew": "Bệnh phấn trắng",
    "leaf_scorch": "Bệnh cháy xém phiến lá",
    "target_spot": "Bệnh đốm mắt chim",
    "mosaic_virus": "Bệnh virus khảm lá",
    "yellow_leaf_curl_virus": "Bệnh xoăn vàng lá do virus",
    "spider_mites": "Bệnh nhện đỏ hại lá",
    "leaf_mold": "Bệnh đốm mốc lá",
    "septoria_leaf_spot": "Bệnh đốm lá Septoria",
    "anthracnose": "Bệnh thán thư",
    "blast": "Bệnh đạo ôn",
    "brown_spot": "Bệnh đốm nâu",
    "stem_canker": "Bệnh thối thân / loét cành",
    "rust": "Bệnh gỉ sắt",
    "healthy": "Lá khỏe mạnh",
}


def normalize_label(name: str) -> str:
    """'Tomato___Early_blight' -> 'tomato_early_blight'."""
    s = re.sub(r"[^0-9a-z]+", "_", name.lower())
    return re.sub(r"_+", "_", s).strip("_")


def parse_label(class_name: str) -> dict:
    """Phân tích nhãn YOLO bất kỳ thành thực thể cây trồng và bệnh học."""
    raw = class_name.strip()
    norm = normalize_label(raw)
    is_healthy = "healthy" in norm.split("_")

    crop_en = "General"
    disease_en = "Disease"

    if "___" in raw:
        parts = raw.split("___", 1)
        crop_en = parts[0].replace("_", " ").strip()
        crop_en = re.sub(r"\s*\(.*?\)", "", crop_en).strip()
        disease_en = parts[1].replace("_", " ").strip()
    elif "_" in raw:
        tokens = raw.split("_")
        crop_en = tokens[0].capitalize()
        disease_en = " ".join(tokens[1:]).capitalize()
    else:
        crop_en = raw.capitalize()

    # Ánh xạ tên tiếng Việt của cây
    crop_lower = crop_en.lower()
    crop_vi = CROP_TRANSLATIONS.get(crop_lower, crop_en)

    # Ánh xạ tên tiếng Việt của bệnh
    disease_norm = normalize_label(disease_en)
    if is_healthy:
        disease_vi = "Lá khỏe mạnh"
        display_name = f"Lá khỏe mạnh (Healthy) trên cây {crop_vi} ({crop_en})"
    else:
        disease_vi = DISEASE_TRANSLATIONS.get(disease_norm)
        if not disease_vi:
            # Tìm kiếm mờ từng phần
            for k, v in DISEASE_TRANSLATIONS.items():
                if k in disease_norm:
                    disease_vi = v
                    break
            if not disease_vi:
                disease_vi = f"Bệnh {disease_en}"
        display_name = f"{disease_vi} ({disease_en}) trên cây {crop_vi} ({crop_en})"

    return {
        "key": norm,
        "crop_en": crop_en.capitalize(),
        "crop_vi": crop_vi,
        "disease_en": disease_en.capitalize(),
        "disease_vi": disease_vi,
        "is_healthy": is_healthy,
        "display_name": display_name,
        "pathogen": "Xem tài liệu chuyên môn",
    }


def label_to_query(class_name: str, aspect: str | None = None) -> str:
    """Sinh câu truy vấn ngữ nghĩa phong phú cho RAG từ nhãn YOLO."""
    info = parse_label(class_name)
    if info["is_healthy"]:
        return f"Cách chăm sóc và duy trì cây {info['crop_vi']} ({info['crop_en']}) khỏe mạnh, phòng ngừa nấm bệnh"

    base_query = f"{info['disease_vi']} ({info['disease_en']}) trên cây {info['crop_vi']} ({info['crop_en']})"
    if aspect == "symptoms":
        return f"Triệu chứng nhận biết và dấu hiệu gây hại của {base_query}"
    elif aspect == "treatment":
        return f"Phác đồ điều trị sinh học, thuốc phòng trừ và cách khắc phục {base_query}"
    elif aspect == "prevention":
        return f"Biện pháp canh tác và kỹ thuật phòng ngừa {base_query}"

    return (
        f"{base_query}: triệu chứng nhận biết, tác nhân gây bệnh, "
        f"phác đồ điều trị sinh học và biện pháp phòng ngừa canh tác"
    )


def yolo_detections_to_queries(detections: list[tuple[str, float]]) -> list[str]:
    """Chuyển đổi danh sách nhận diện YOLO thành các truy vấn RAG tương ứng."""
    return [label_to_query(cls_name) for cls_name, _ in detections if cls_name]
