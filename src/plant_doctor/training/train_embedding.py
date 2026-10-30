"""Bước 2: Tinh chỉnh mô hình nhúng (sentence-transformers) trên dữ liệu bệnh cây.

Cặp huấn luyện (truy vấn -> đoạn tài liệu) được sinh tự động từ data/knowledge_base/diseases.json
và có thể bổ sung trong data/training/embedding_pairs.jsonl: {"query": "...", "passage": "..."}.

Hàm mất mát: MultipleNegativesRankingLoss (InfoNCE với negative trong batch).
"""
import argparse
import json
import random
from pathlib import Path

import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.datasets import NoDuplicatesDataLoader

from plant_doctor.config import EMBED_BASE, EMBED_DIR, SEED, TRAIN_DATA_DIR
from plant_doctor.knowledge import entry_to_text, load_kb, section_text


def queries_for(e):
    crop = e["crop_vi"].lower()
    name_q = {
        e["name_en"], e["name_vi"], e["key"].replace("_", " "),
        f"{e['crop_en']} {e['name_en']}", f"{e['name_vi']} trên cây {crop}",
        f"Mô hình YOLOv8 phát hiện {e['crop_en']} {e['name_en']}", e["pathogen"],
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


def build_pairs(kb):
    pairs = []
    for e in kb:
        full = entry_to_text(e)
        name_q, section_q = queries_for(e)
        pairs += [(q, full) for q in name_q]
        for q, field in section_q:
            pairs.append((q, full))                    # khớp với tài liệu đầy đủ trong FAISS
            pairs.append((q, section_text(e, field)))  # và với đúng phần nội dung liên quan
    extra = TRAIN_DATA_DIR / "embedding_pairs.jsonl"
    if extra.exists():
        for line in extra.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                pairs.append((r["query"], r["passage"]))
    return pairs


@torch.no_grad()
def top1_accuracy(model, kb):
    docs = [entry_to_text(e) for e in kb]
    doc_emb = model.encode(docs, normalize_embeddings=True, convert_to_tensor=True)
    hits = total = 0
    for i, e in enumerate(kb):
        name_q, section_q = queries_for(e)
        qs = list(name_q) + [q for q, _ in section_q]
        q_emb = model.encode(qs, normalize_embeddings=True, convert_to_tensor=True)
        pred = (q_emb @ doc_emb.T).argmax(dim=1)
        hits += int((pred == i).sum())
        total += len(qs)
    return hits / total


def train_custom_embedding(base_model: str = EMBED_BASE,
                           output_dir: Path | str = EMBED_DIR,
                           epochs: int = 20,
                           batch: int = 16) -> float:
    """Huấn luyện mô hình Embedding bằng MultipleNegativesRankingLoss và trả về top1 accuracy."""
    random.seed(SEED)
    torch.manual_seed(SEED)
    kb = load_kb()
    pairs = build_pairs(kb)
    print(f"Số cặp huấn luyện: {len(pairs)}")

    model = SentenceTransformer(base_model)
    acc_before = top1_accuracy(model, kb)
    print(f"Top-1 retrieval TRƯỚC khi fine-tune: {acc_before:.3f}")

    examples = [InputExample(texts=[q, p]) for q, p in pairs]
    loader = NoDuplicatesDataLoader(examples, batch_size=batch)
    loss = losses.MultipleNegativesRankingLoss(model)
    steps = len(loader) * epochs
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        warmup_steps=int(0.1 * steps),
        show_progress_bar=True,
    )
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
