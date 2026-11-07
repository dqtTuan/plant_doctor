"""Sinh mẫu huấn luyện/đánh giá dạng RAG cho LLM: (ngữ cảnh, câu hỏi) -> câu trả lời.

- Mẫu dương ("answer")  : ngữ cảnh chứa đúng bệnh (+1 tài liệu gây nhiễu) -> trả lời có cấu trúc
- Mẫu âm  ("refusal")   : ngữ cảnh KHÔNG chứa bệnh được hỏi -> từ chối (chống ảo giác)
Mỗi mẫu có trường `group` (khoá bệnh) để chia train/val/test theo bệnh, tránh rò rỉ dữ liệu.
Bổ sung tuỳ chọn: data/training/llm_extra.jsonl {"context", "question", "answer"}.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from plant_doctor.config import KB_DIR, SEED, TRAIN_DATA_DIR
from plant_doctor.inference.knowledge import entry_to_answer, entry_to_text
from plant_doctor.inference.prompts import format_context

TEMPLATES = {
    "full": [
        "Mô hình YOLOv8 vừa phát hiện lá cây bị bệnh: {cls}. Hãy giải thích triệu chứng, "
        "nguyên nhân gây bệnh và cách điều trị sinh học/phòng ngừa.",
        "Cây {crop} nhà tôi bị {name}, tôi phải làm sao?",
        "Cho tôi phác đồ đầy đủ về {name} trên cây {crop}.",
    ],
    "symptoms": ["{name} trên {crop} có triệu chứng gì?", "Làm sao nhận biết {name} trên lá {crop}?"],
    "causes": ["Nguyên nhân nào gây ra {name} trên {crop}?", "Vì sao cây {crop} bị {name}?"],
    "treatment": [
        "Có cách nào chữa {name} trên {crop} mà hạn chế thuốc hóa học không?",
        "Phác đồ điều trị sinh học cho {name} trên {crop} là gì?",
    ],
    "prevention": ["Làm thế nào để phòng ngừa {name} cho cây {crop} ở vụ sau?"],
}

REFUSAL = (
    "Tài liệu tham khảo được cung cấp không chứa thông tin về {name} trên cây {crop}. "
    "Để tránh tư vấn sai, tôi không đưa ra phác đồ cụ thể. Bạn nên chụp thêm ảnh rõ nét hơn "
    "hoặc liên hệ cán bộ khuyến nông / trạm bảo vệ thực vật địa phương để được hỗ trợ."
)


def fill(tpl: str, e: dict) -> str:
    cls = e["aliases"][0] if e.get("aliases") else f"{e['crop_en']} {e['name_en']}"
    return tpl.format(cls=cls, crop=e["crop_vi"].lower(), name=e["name_vi"].lower())


def entries_from_knowledge_base(kb_dir: Path | str = KB_DIR) -> list[dict]:
    """Trích danh sách mục bệnh từ tài liệu (.md theo mục '##', .pdf/.txt theo trang)."""
    from plant_doctor.rag.loaders import load_directory

    entries = []
    for md_file in sorted(Path(kb_dir).glob("*.md")):
        if md_file.name.lower().startswith("readme"):
            continue
        try:
            content = md_file.read_text(encoding="utf-8")
        except Exception:
            continue
        for block in re.split(r"\n##\s+", content)[1:]:
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if not lines:
                continue
            title_line = lines[0]
            name_part = title_line.split("–")[0].strip() if "–" in title_line else title_line
            crop_part = title_line.split("–")[1].strip() if "–" in title_line else "Cây trồng"
            entries.append({
                "key": re.sub(r"[^0-9a-z]+", "_", name_part.lower()).strip("_"),
                "name_vi": name_part.split("(")[0].strip(),
                "name_en": name_part.split("(")[1].rstrip(")") if "(" in name_part else name_part,
                "crop_vi": crop_part.split(":")[1].split("(")[0].strip() if ":" in crop_part else crop_part,
                "crop_en": crop_part.split("(")[1].rstrip(")") if "(" in crop_part else crop_part,
                "symptoms": block,
                "pathogen": "Nấm hoặc vi khuẩn gây hại",
                "causes": "Thời tiết ẩm ướt kéo dài.",
                "bio_treatment": "Sử dụng chế phẩm sinh học Trichoderma hoặc Bacillus.",
                "chemical_note": "Chỉ dùng khi cần thiết và cách ly an toàn.",
                "prevention": "Vệ sinh đồng ruộng, tỉa cành thông thoáng.",
                "aliases": [name_part],
            })

    for doc in load_directory(kb_dir):
        source = doc.metadata.get("source", "")
        if source.lower().endswith(".md"):
            continue
        page = doc.metadata.get("page")
        content = doc.page_content.strip()
        if len(content) < 50:
            continue
        stem = Path(source).stem
        entries.append({
            "key": re.sub(r"[^0-9a-z]+", "_", f"{stem}_{page or ''}".lower()).strip("_"),
            "name_vi": f"{stem}{f' (trang {page})' if page else ''}",
            "name_en": stem,
            "crop_vi": "Cây trồng",
            "crop_en": "Crops",
            "symptoms": content[:500],
            "pathogen": f"Thông tin chuyên ngành từ {source}",
            "causes": "Điều kiện canh tác thực tế.",
            "bio_treatment": content,
            "chemical_note": "Tuân thủ khuyến cáo bảo vệ thực vật.",
            "prevention": "Áp dụng biện pháp canh tác bền vững.",
            "aliases": [stem],
        })
    return entries


def build_samples(kb: list[dict] | None = None, rng: random.Random | None = None,
                  n_distractors: int = 1) -> list[dict]:
    """Sinh mẫu RAG. Mỗi mẫu: context, question, answer, group (khoá bệnh), kind (answer|refusal)."""
    rng = rng or random.Random(SEED)
    kb = kb or entries_from_knowledge_base()
    diseases = [e for e in kb if e["key"] != "healthy"]
    healthy = next((e for e in kb if e["key"] == "healthy"), None)
    samples: list[dict] = []
    for e in diseases:
        others = [o for o in diseases if o["key"] != e["key"]]
        for focus, tpls in TEMPLATES.items():
            for tpl in tpls:
                ctx = [e, *rng.sample(others, min(n_distractors, len(others)))]
                rng.shuffle(ctx)
                samples.append({
                    "context": format_context([entry_to_text(c) for c in ctx]),
                    "question": fill(tpl, e),
                    "answer": entry_to_answer(e, focus),
                    "group": e["key"], "kind": "answer",
                })
        if others:  # mẫu âm: ngữ cảnh không liên quan -> phải từ chối
            for tpl in (TEMPLATES["full"][1], TEMPLATES["treatment"][0]):
                ctx = rng.sample(others, min(2, len(others)))
                samples.append({
                    "context": format_context([entry_to_text(c) for c in ctx]),
                    "question": fill(tpl, e),
                    "answer": REFUSAL.format(name=e["name_vi"].lower(), crop=e["crop_vi"].lower()),
                    "group": e["key"], "kind": "refusal",
                })
    if healthy:
        for q in ("Mô hình YOLOv8 phát hiện lá khỏe mạnh (healthy). Tôi cần làm gì để giữ cây khỏe?",
                  "Lá cây của tôi không có bệnh, có cần phun thuốc phòng không?"):
            samples.append({"context": format_context([entry_to_text(healthy)]), "question": q,
                            "answer": entry_to_answer(healthy, "full"), "group": "healthy", "kind": "answer"})

    extra = TRAIN_DATA_DIR / "llm_extra.jsonl"
    if extra.exists():
        for line in extra.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                r.setdefault("group", f"extra:{r['question']}")
                r.setdefault("kind", "answer")
                samples.append(r)
    rng.shuffle(samples)
    return samples

