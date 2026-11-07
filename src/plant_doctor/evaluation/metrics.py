"""Các hàm đo lường thuần Python (không phụ thuộc torch) – dễ kiểm thử.

- Truy xuất (embedding, RAG): accuracy@1, recall@k, MRR@k, NDCG@k
- Sinh văn bản (LLM): token-F1, ROUGE-L (F1 theo dãy con chung dài nhất), khớp chính xác
- Chống ảo giác: phát hiện câu từ chối và đo precision/recall của hành vi từ chối
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence


# ---------------------------------------------------------------- Truy xuất
def retrieval_metrics(ranked: Sequence[Sequence], relevant: Sequence[set],
                      ks: Sequence[int] = (1, 3, 5, 10)) -> dict:
    """Tính chỉ số truy xuất trung bình trên nhiều truy vấn.

    ranked[i]   : danh sách id tài liệu đã xếp hạng cho truy vấn i (tốt nhất trước)
    relevant[i] : tập id tài liệu đúng của truy vấn i
    """
    if len(ranked) != len(relevant):
        raise ValueError("ranked và relevant phải cùng độ dài")
    n = len(ranked)
    if n == 0:
        return {}
    max_k = max(ks)
    out: dict[str, float] = {}
    recall = {k: 0.0 for k in ks}
    ndcg = {k: 0.0 for k in ks}
    mrr = 0.0
    for docs, rel in zip(ranked, relevant):
        if not rel:
            continue
        first = next((r for r, d in enumerate(docs[:max_k], 1) if d in rel), None)
        if first is not None:
            mrr += 1.0 / first
        for k in ks:
            top = list(docs[:k])
            recall[k] += len(rel.intersection(top)) / len(rel)
            dcg = sum(1.0 / math.log2(r + 1) for r, d in enumerate(top, 1) if d in rel)
            idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(rel), k) + 1))
            ndcg[k] += dcg / idcg if idcg else 0.0
    out["accuracy@1"] = sum(1 for docs, rel in zip(ranked, relevant) if docs and docs[0] in rel) / n
    for k in ks:
        out[f"recall@{k}"] = recall[k] / n
        out[f"ndcg@{k}"] = ndcg[k] / n
    out[f"mrr@{max_k}"] = mrr / n
    return out


# ---------------------------------------------------------------- Sinh văn bản
def normalize_text(text: str) -> str:
    """Chữ thường, chuẩn Unicode NFC, bỏ markdown/dấu câu, gộp khoảng trắng."""
    text = unicodedata.normalize("NFC", text.lower())
    text = re.sub(r"[*_#`>\[\]()|]", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return normalize_text(text).split()


def token_f1(prediction: str, reference: str) -> float:
    """F1 trên túi từ (kiểu SQuAD)."""
    pred, ref = tokenize(prediction), tokenize(reference)
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    common = sum((Counter(pred) & Counter(ref)).values())
    if common == 0:
        return 0.0
    p, r = common / len(pred), common / len(ref)
    return 2 * p * r / (p + r)


def _lcs_len(a: list[str], b: list[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1 ở mức từ (không cần thư viện rouge-score)."""
    pred, ref = tokenize(prediction), tokenize(reference)
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    lcs = _lcs_len(pred, ref)
    if lcs == 0:
        return 0.0
    p, r = lcs / len(pred), lcs / len(ref)
    return 2 * p * r / (p + r)


# ---------------------------------------------------------------- Từ chối (chống ảo giác)
REFUSAL_MARKERS = (
    "không chứa thông tin",
    "không có thông tin",
    "không tìm thấy thông tin",
    "không đưa ra phác đồ",
    "tránh tư vấn sai",
    "tài liệu không đề cập",
)


def is_refusal(text: str) -> bool:
    t = unicodedata.normalize("NFC", text.lower())
    return any(m in t for m in REFUSAL_MARKERS)


def refusal_metrics(should_refuse: Sequence[bool], did_refuse: Sequence[bool]) -> dict:
    """Precision/recall/F1 của hành vi từ chối + tỉ lệ từ chối nhầm câu có đáp án."""
    tp = sum(1 for s, d in zip(should_refuse, did_refuse) if s and d)
    fp = sum(1 for s, d in zip(should_refuse, did_refuse) if not s and d)
    fn = sum(1 for s, d in zip(should_refuse, did_refuse) if s and not d)
    n_pos = sum(1 for s in should_refuse if s)
    n_neg = len(should_refuse) - n_pos
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "refusal_precision": precision,
        "refusal_recall": recall,
        "refusal_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_refusal_rate": fp / n_neg if n_neg else 0.0,
        "n_should_refuse": n_pos,
        "n_should_answer": n_neg,
    }

