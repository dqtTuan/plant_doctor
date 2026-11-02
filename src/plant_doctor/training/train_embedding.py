"""Bước 2: Tinh chỉnh mô hình nhúng (sentence-transformers) trên dữ liệu bệnh cây.

Cặp huấn luyện (truy vấn -> đoạn tài liệu) được sinh tự động từ Markdown trong kho tri thức
và có thể bổ sung trong data/training/embedding_pairs.jsonl: {"query": "...", "passage": "..."}.

Hàm mất mát: MultipleNegativesRankingLoss (InfoNCE với negative trong batch).
"""
import argparse
import json
import random
from pathlib import Path

from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.training_args import BatchSamplers
import torch

from plant_doctor.config import EMBED_BASE, EMBED_DIR, MODELS_DIR, SEED, TRAIN_DATA_DIR
from plant_doctor.knowledge import entry_to_text, load_kb, section_text


def queries_for(e):
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


def extract_pairs_from_knowledge_base() -> list[tuple[str, str]]:
    """Trích xuất cặp câu hỏi - tài liệu từ toàn bộ file trong kho tri thức (.md, .pdf, .txt)."""
    import re
    from plant_doctor.config import KB_DIR
    from plant_doctor.rag.loaders import load_directory

    pairs = []
    # 1. Trích xuất từ Markdown
    for md_file in Path(KB_DIR).glob("*.md"):
        if md_file.name.lower().startswith("readme"):
            continue
        try:
            content = md_file.read_text(encoding="utf-8")
        except Exception:
            continue
        blocks = re.split(r"\n##\s+", content)
        for block in blocks[1:]:
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if not lines:
                continue
            title_line = lines[0]
            pairs.append((f"Cách phòng trị và triệu chứng của {title_line}", block))
            pairs.append((title_line, block))

    # 2. Trích xuất từ các định dạng khác như PDF và TXT
    all_docs = load_directory(KB_DIR)
    for doc in all_docs:
        source = doc.metadata.get("source", "")
        if source.lower().endswith(".md"):
            continue
        page = doc.metadata.get("page")
        content = doc.page_content.strip()
        if not content or len(content) < 30:
            continue
        page_str = f" trang {page}" if page else ""
        paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 30]
        if not paragraphs:
            paragraphs = [content]
        for p in paragraphs:
            first_sentence = p.split("\n")[0].split(".")[0].strip()
            if len(first_sentence) > 10:
                pairs.append((first_sentence, p))
            pairs.append((f"Thông tin tài liệu {source}{page_str}", p))

    return pairs

extract_pairs_from_markdown = extract_pairs_from_knowledge_base


def build_pairs(kb=None):
    pairs = []
    if kb:
        for e in kb:
            full = entry_to_text(e)
            name_q, section_q = queries_for(e)
            pairs += [(q, full) for q in name_q]
            for q, field in section_q:
                pairs.append((q, full))  # khớp với tài liệu đầy đủ trong FAISS
                pairs.append((q, section_text(e, field)))  # và với đúng phần nội dung liên quan
    else:
        pairs += extract_pairs_from_markdown()

    extra = TRAIN_DATA_DIR / "embedding_pairs.jsonl"
    if extra.exists():
        for line in extra.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                pairs.append((r["query"], r["passage"]))
    return pairs


def build_train_dataset(pairs: list[tuple[str, str]]) -> Dataset:
    """Chuyển đổi danh sách cặp (anchor, positive) thành HuggingFace Dataset."""
    anchors = [q for q, _ in pairs]
    positives = [p for _, p in pairs]
    return Dataset.from_dict({"anchor": anchors, "positive": positives})


@torch.no_grad()
def top1_accuracy(model, kb=None):
    if kb:
        docs = [entry_to_text(e) for e in kb]
        if not docs:
            return 1.0
        doc_emb = model.encode(docs, normalize_embeddings=True, convert_to_tensor=True)
        hits = total = 0
        for i, e in enumerate(kb):
            name_q, section_q = queries_for(e)
            qs = list(name_q) + [q for q, _ in section_q]
            if not qs:
                continue
            q_emb = model.encode(qs, normalize_embeddings=True, convert_to_tensor=True)
            pred = (q_emb @ doc_emb.T).argmax(dim=1)
            hits += int((pred == i).sum())
            total += len(qs)
        return (hits / total) if total > 0 else 1.0

    pairs = extract_pairs_from_markdown()
    if not pairs:
        return 1.0
    sample_pairs = pairs[:min(30, len(pairs))]
    qs = [q for q, _ in sample_pairs]
    passages = [p for _, p in sample_pairs]
    q_emb = model.encode(qs, normalize_embeddings=True, convert_to_tensor=True)
    p_emb = model.encode(passages, normalize_embeddings=True, convert_to_tensor=True)
    sim = q_emb @ p_emb.T
    target = torch.arange(sim.shape[0], device=sim.device)
    hits = int((sim.argmax(dim=1) == target).sum())
    return hits / sim.shape[0]


def train_custom_embedding(
    base_model: str = EMBED_BASE,
    output_dir: Path | str = EMBED_DIR,
    epochs: int = 20,
    batch: int = 16,
) -> float:
    """Huấn luyện mô hình Embedding bằng MultipleNegativesRankingLoss và trả về top1 accuracy."""
    random.seed(SEED)
    torch.manual_seed(SEED)
    kb = load_kb()
    pairs = build_pairs(kb)
    if not pairs:
        print("⚠️ Không tìm thấy cặp dữ liệu nào để huấn luyện Embedding.")
        print("Vui lòng đặt tài liệu (.md, .pdf, .txt) vào data/knowledge_base/.")
        return 1.0
    print(f"Số cặp huấn luyện: {len(pairs)}")

    model = SentenceTransformer(base_model)
    acc_before = top1_accuracy(model, kb)
    print(f"Top-1 retrieval TRƯỚC khi fine-tune: {acc_before:.3f}")

    train_dataset = build_train_dataset(pairs)
    loss = MultipleNegativesRankingLoss(model)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(MODELS_DIR / "embedding_checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        seed=SEED,
        save_strategy="no",
        report_to="none",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
    )
    trainer.train()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    acc_after = top1_accuracy(model, kb)
    print(f"Top-1 retrieval SAU khi fine-tune: {acc_after:.3f}")
    print(f"Đã lưu mô hình Embedding tự train tại: {out_path}")
    return acc_after


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=EMBED_BASE)
    ap.add_argument("--output", default=str(EMBED_DIR))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    train_custom_embedding(
        base_model=args.base,
        output_dir=args.output,
        epochs=args.epochs,
        batch=args.batch,
    )


if __name__ == "__main__":
    main()
