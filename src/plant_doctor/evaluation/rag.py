"""Đánh giá truy xuất RAG end-to-end trên tập câu hỏi vàng (hit@k, MRR, no-answer accuracy).

    pd eval rag            # cấu hình: configs/rag.yaml
    pd eval rag k=6
"""
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Callable

from langchain_core.documents import Document


@dataclass
class GoldenItem:
    question: str
    labels: list[str] = field(default_factory=list)
    expected: list[tuple[str, int | None]] = field(default_factory=list)


def load_golden(path: Path | str) -> list[GoldenItem]:
    """Đọc tệp JSONL chứa tập câu hỏi vàng kiểm thử RAG."""
    items = []
    p = Path(path)
    if not p.exists():
        return []

    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        raw = json.loads(line)
        expected_raw = raw.get("expected", [])
        expected = []
        for e in expected_raw:
            if isinstance(e, (list, tuple)):
                expected.append((e[0], e[1] if len(e) > 1 else None))
            elif isinstance(e, dict):
                expected.append((e.get("source", ""), e.get("page")))
        items.append(
            GoldenItem(
                question=raw["question"],
                labels=raw.get("labels", []),
                expected=expected,
            )
        )
    return items


def evaluate(
    retrieve: Callable[[str, list[str]], list[Document]],
    items: list[GoldenItem],
    k: int = 4,
) -> dict:
    """Tính toán hit@k, MRR và độ chính xác từ chối (no-answer accuracy)."""
    hits = 0
    mrr_sum = 0.0
    no_answer_correct = 0
    n_answerable = 0
    n_unanswerable = 0

    for item in items:
        docs = retrieve(item.question, item.labels)[:k]
        if not item.expected:
            n_unanswerable += 1
            if len(docs) == 0:
                no_answer_correct += 1
        else:
            n_answerable += 1
            found_rank = None
            for rank, doc in enumerate(docs, start=1):
                src = doc.metadata.get("source") if hasattr(doc, "metadata") else None
                page = doc.metadata.get("page") if hasattr(doc, "metadata") else None
                for exp_src, exp_page in item.expected:
                    if src == exp_src and (exp_page is None or page == exp_page):
                        found_rank = rank
                        break
                if found_rank is not None:
                    break
            if found_rank is not None:
                hits += 1
                mrr_sum += 1.0 / found_rank

    return {
        "hit_at_k": round(hits / n_answerable, 4) if n_answerable > 0 else 0.0,
        "mrr": round(mrr_sum / n_answerable, 4) if n_answerable > 0 else 0.0,
        "no_answer_acc": round(no_answer_correct / n_unanswerable, 4) if n_unanswerable > 0 else 1.0,
        "n_answerable": n_answerable,
        "n_unanswerable": n_unanswerable,
    }


def run_rag_eval(cfg: dict) -> dict:
    """Đánh giá pipeline truy xuất thật trên golden set, lưu kết quả vào runs/rag/<timestamp>/."""
    from plant_doctor.utils.experiment import Run, metrics_table

    k = int(cfg.get("k", 4))
    items = load_golden(cfg["golden"])
    if not items:
        print(f"Chưa có dữ liệu kiểm thử tại: {cfg['golden']} (xem data/eval/README.md)")
        return {}

    from plant_doctor.inference.pipeline import PlantDoctorPipeline

    run = Run.create("rag", cfg)
    p = PlantDoctorPipeline(load_models=True)
    metrics = evaluate(lambda q, labels: p.retrieve_context(q, labels, max_docs=k), items, k=k)
    run.log_metrics(metrics)
    run.write_report(f"# Đánh giá truy xuất RAG (k={k})\n\n"
                     f"{len(items)} câu hỏi ({metrics['n_answerable']} có đáp án, "
                     f"{metrics['n_unanswerable']} cần từ chối)\n\n{metrics_table(metrics)}\n")
    run.finish()

    print("=" * 45)
    print(f"Hit@{k}        : {metrics['hit_at_k']:.2%}")
    print(f"MRR           : {metrics['mrr']:.4f}")
    print(f"No-Answer Acc : {metrics['no_answer_acc']:.2%}")
    print("=" * 45)
    return metrics
