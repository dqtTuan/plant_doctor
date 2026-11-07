"""Sinh cặp (truy vấn, đoạn tài liệu) để tinh chỉnh và đánh giá mô hình embedding.

Nguồn: tài liệu .md/.pdf/.txt trong data/knowledge_base/ (+ tuỳ chọn data/training/embedding_pairs.jsonl
dạng {"query": "...", "passage": "..."}).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from plant_doctor.config import KB_DIR, TRAIN_DATA_DIR
from plant_doctor.inference.knowledge import entry_to_text, section_text

Pair = tuple[str, str]


def queries_for(e: dict) -> tuple[set[str], list[tuple[str, str]]]:
    """Truy vấn mẫu cho một mục bệnh có cấu trúc: (truy vấn theo tên, [(truy vấn, trường)])."""
    crop = e["crop_vi"].lower()
    name_q = {
        e["name_en"],
        e["name_vi"],
        e["key"].replace("_", " "),
        f"{e['crop_en']} {e['name_en']}",
        f"{e['name_vi']} trên cây {crop}",
        f"Mô hình YOLOv8 phát hiện {e['crop_en']} {e['name_en']}",
        e["pathogen"],
        *e.get("aliases", []),
    }
    section_q = [
        (f"Triệu chứng của {e['name_vi']} trên {crop} là gì?", "symptoms"),
        (f"What are the symptoms of {e['name_en']} on {e['crop_en']}?", "symptoms"),
        (f"Nguyên nhân gây {e['name_vi']} trên {crop}", "causes"),
        (f"Cách chữa {e['name_vi']} cho {crop} bằng biện pháp sinh học", "bio_treatment"),
        (f"How to treat {e['crop_en']} {e['name_en']} organically", "bio_treatment"),
        (f"Phòng ngừa {e['name_vi']} cho cây {crop}", "prevention"),
    ]
    return name_q, section_q


def pairs_from_entries(kb: list[dict]) -> list[Pair]:
    pairs: list[Pair] = []
    for e in kb:
        full = entry_to_text(e)
        name_q, section_q = queries_for(e)
        pairs += [(q, full) for q in sorted(name_q)]
        for q, field in section_q:
            pairs.append((q, full))
            pairs.append((q, section_text(e, field)))
    return pairs


def pairs_from_knowledge_base(kb_dir: Path | str = KB_DIR) -> list[Pair]:
    """Trích cặp từ toàn bộ tài liệu trong kho tri thức (.md theo mục '##', .pdf/.txt theo đoạn)."""
    from plant_doctor.rag.loaders import load_directory

    pairs: list[Pair] = []
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
            pairs.append((f"Cách phòng trị và triệu chứng của {title_line}", block))
            pairs.append((title_line, block))

    for doc in load_directory(kb_dir):
        source = doc.metadata.get("source", "")
        if source.lower().endswith(".md"):
            continue
        page = doc.metadata.get("page")
        content = doc.page_content.strip()
        if len(content) < 30:
            continue
        page_str = f" trang {page}" if page else ""
        paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 30] or [content]
        for p in paragraphs:
            first_sentence = p.split("\n")[0].split(".")[0].strip()
            if len(first_sentence) > 10:
                pairs.append((first_sentence, p))
            pairs.append((f"Thông tin tài liệu {source}{page_str}", p))
    return pairs


def extra_pairs(path: Path | str = TRAIN_DATA_DIR / "embedding_pairs.jsonl") -> list[Pair]:
    path = Path(path)
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [(r["query"], r["passage"]) for r in rows]


def build_pairs(kb: list[dict] | None = None) -> list[Pair]:
    """Toàn bộ cặp huấn luyện: từ mục có cấu trúc (nếu có) hoặc từ tài liệu, cộng dữ liệu bổ sung."""
    pairs = pairs_from_entries(kb) if kb else pairs_from_knowledge_base()
    return pairs + extra_pairs()

