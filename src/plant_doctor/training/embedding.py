"""Tinh chỉnh mô hình embedding (MultipleNegativesRankingLoss) + đánh giá truy xuất held-out.

    pd train embedding                   # cấu hình: configs/embedding.yaml
    pd train embedding train.epochs=5

Dữ liệu được chia train/val/test theo đoạn tài liệu (data/splits.py). Mô hình gốc và mô hình
sau tinh chỉnh được đánh giá trên cùng tập test để thấy rõ mức cải thiện.
"""
from __future__ import annotations

from pathlib import Path

from datasets import Dataset

from plant_doctor.data.embedding_pairs import Pair, build_pairs
from plant_doctor.data.splits import group_split
from plant_doctor.utils.experiment import Run, set_seed


def build_train_dataset(pairs: list[Pair]) -> Dataset:
    """Chuyển danh sách cặp (anchor, positive) thành HuggingFace Dataset."""
    return Dataset.from_dict({"anchor": [q for q, _ in pairs], "positive": [p for _, p in pairs]})


def split_pairs(pairs: list[Pair], cfg: dict) -> dict[str, list[Pair]]:
    d = cfg["data"]
    return group_split(pairs, group_fn=lambda pr: pr[1], val=d["val_ratio"], test=d["test_ratio"],
                       seed=cfg["seed"])


def _eval_split(model, split: dict[str, list[Pair]], name: str, ks) -> dict:
    from plant_doctor.evaluation.embedding import build_ir_dataset, evaluate_embedding

    corpus_pairs = split["train"] + split["val"] + split["test"]
    queries, relevant, corpus = build_ir_dataset(split[name], corpus_pairs)
    return evaluate_embedding(model, queries, relevant, corpus, ks=ks)


def _report(rows: dict[str, dict], sizes: dict[str, int]) -> str:
    keys = [k for k in next(iter(rows.values()), {}) if "@" in k]
    lines = ["# Đánh giá mô hình Embedding (truy xuất held-out)", "",
             "Số cặp: " + ", ".join(f"{k}={v}" for k, v in sizes.items()), "",
             "| Mô hình / tập | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for name, m in rows.items():
        lines.append(f"| {name} | " + " | ".join(f"{m.get(k, 0):.4f}" for k in keys) + " |")
    return "\n".join(lines) + "\n"


def train_embedding(cfg: dict) -> dict:
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments)
    from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
    from sentence_transformers.sentence_transformer.training_args import BatchSamplers

    set_seed(cfg["seed"])
    pairs = build_pairs()
    if not pairs:
        print("⚠️ Không có cặp dữ liệu nào. Hãy đặt tài liệu (.md/.pdf/.txt) vào data/knowledge_base/.")
        return {}
    split = split_pairs(pairs, cfg)
    sizes = {k: len(v) for k, v in split.items()}
    print(f"Số cặp: {sizes}")
    ks = cfg["eval"]["ks"]

    run = Run.create("embedding", cfg)
    model = SentenceTransformer(cfg["model"]["base"])
    rows = {}
    if split["test"]:
        rows["base / test"] = _eval_split(model, split, "test", ks)
        run.log_metrics(rows["base / test"], prefix="base_test")

    t = cfg["train"]
    args = SentenceTransformerTrainingArguments(
        output_dir=str(run.dir / "checkpoints"), num_train_epochs=t["epochs"],
        per_device_train_batch_size=t["batch"], learning_rate=t["lr"],
        warmup_steps=t["warmup_ratio"],  # transformers>=5: số thực < 1 = tỉ lệ (thay warmup_ratio)
        batch_sampler=BatchSamplers.NO_DUPLICATES, seed=cfg["seed"], save_strategy="no", report_to="none",
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=build_train_dataset(split["train"]),
                                         loss=MultipleNegativesRankingLoss(model))
    train_out = trainer.train()
    run.log_metrics({"train_loss": train_out.training_loss})

    out = Path(cfg["model"]["output"])
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    print(f"Đã lưu mô hình Embedding tại: {out}")

    for name in ("val", "test"):
        if split[name]:
            rows[f"fine-tuned / {name}"] = _eval_split(model, split, name, ks)
            run.log_metrics(rows[f"fine-tuned / {name}"], prefix=name)
    run.write_report(_report(rows, sizes))
    run.finish()
    for name, m in rows.items():
        print(f"{name:20s} acc@1={m.get('accuracy@1', 0):.4f}  recall@5={m.get('recall@5', 0):.4f}  "
              f"mrr={m.get(f'mrr@{max(ks)}', 0):.4f}")
    return rows.get("fine-tuned / test", {})


def evaluate_embedding_run(cfg: dict) -> dict:
    """Chỉ đánh giá mô hình model.output (và mô hình gốc để so sánh) trên tập test."""
    from sentence_transformers import SentenceTransformer

    set_seed(cfg["seed"])
    split = split_pairs(build_pairs(), cfg)
    if not split["test"]:
        print("⚠️ Tập test rỗng – cần thêm tài liệu vào data/knowledge_base/.")
        return {}
    ks = cfg["eval"]["ks"]
    run = Run.create("embedding_eval", cfg)
    rows = {"base / test": _eval_split(SentenceTransformer(cfg["model"]["base"]), split, "test", ks),
            "fine-tuned / test": _eval_split(SentenceTransformer(str(cfg["model"]["output"])), split, "test", ks)}
    run.log_metrics(rows["base / test"], prefix="base_test")
    run.log_metrics(rows["fine-tuned / test"], prefix="test")
    report = _report(rows, {k: len(v) for k, v in split.items()})
    run.write_report(report)
    run.finish()
    print(report)
    return rows["fine-tuned / test"]
