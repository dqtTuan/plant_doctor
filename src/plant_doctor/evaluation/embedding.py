"""Đánh giá mô hình embedding như một bài toán truy xuất thông tin (IR) trên tập held-out.

Truy vấn lấy từ tập test (đoạn tài liệu của chúng KHÔNG xuất hiện khi huấn luyện), kho tài liệu
gồm toàn bộ đoạn -> mô phỏng truy xuất thật. Chỉ số: accuracy@1, recall@k, NDCG@k, MRR@k.
"""
from __future__ import annotations

from collections.abc import Sequence

from plant_doctor.evaluation.metrics import retrieval_metrics

Pair = tuple[str, str]


def build_ir_dataset(eval_pairs: Sequence[Pair], corpus_pairs: Sequence[Pair]
                     ) -> tuple[list[str], list[set[int]], list[str]]:
    """-> (queries, relevant_ids, corpus). Một truy vấn có thể có nhiều đoạn đúng."""
    corpus: list[str] = []
    index: dict[str, int] = {}
    for _, p in list(corpus_pairs) + list(eval_pairs):
        if p not in index:
            index[p] = len(corpus)
            corpus.append(p)
    rel: dict[str, set[int]] = {}
    for q, p in eval_pairs:
        rel.setdefault(q, set()).add(index[p])
    queries = list(rel)
    return queries, [rel[q] for q in queries], corpus


def evaluate_embedding(model, queries: list[str], relevant: list[set[int]], corpus: list[str],
                       ks: Sequence[int] = (1, 3, 5, 10), batch_size: int = 64) -> dict:
    """model: SentenceTransformer hoặc đường dẫn tới mô hình."""
    import torch

    if not queries or not corpus:
        return {}
    if isinstance(model, str) or hasattr(model, "__fspath__"):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(str(model))
    with torch.no_grad():
        q = model.encode(queries, batch_size=batch_size, normalize_embeddings=True, convert_to_tensor=True)
        c = model.encode(corpus, batch_size=batch_size, normalize_embeddings=True, convert_to_tensor=True)
        top = (q @ c.T).topk(k=min(max(ks), len(corpus)), dim=1).indices.cpu().tolist()
    metrics = retrieval_metrics(top, relevant, ks=ks)
    metrics.update({"n_queries": len(queries), "corpus_size": len(corpus)})
    return metrics

