"""Thuật toán Semantic Chunking thuần túy cho văn bản tiếng Việt & nông nghiệp.

Cơ chế:
1. Tách văn bản thành các câu đơn lẻ.
2. Mã hóa vector embedding cho từng câu.
3. Tính khoảng cách Cosine Distance d_i = 1 - cos(v_i, v_{i+1}) giữa các câu kế tiếp.
4. Xác định điểm ngắt (breakpoint) bằng phân vị (percentile_threshold).
5. Gom các câu có độ tương đồng cao thành các chunk hoàn chỉnh về mặt ngữ nghĩa.
"""
import re
from typing import Callable

import numpy as np
from langchain_core.documents import Document

from plant_doctor.rag.build_index import tag_disease


def split_sentences(text: str) -> list[str]:
    """Tách văn bản thành danh sách các câu đơn lẻ, xử lý dấu chấm câu thông minh."""
    clean = text.strip()
    if not clean:
        return []

    # Tách theo ngắt dòng kép hoặc các dấu kết thúc câu (. ? !)
    pattern = r'(?<=[.?!])\s+(?=[A-ZÀ-Ỹ0-9])|\n\n+'
    raw_sents = re.split(pattern, clean)

    sentences = []
    for s in raw_sents:
        s_clean = s.strip()
        if s_clean:
            sentences.append(s_clean)
    return sentences if sentences else [clean]


def semantic_chunking(text: str,
                      embed_fn: Callable[[list[str]], np.ndarray],
                      percentile_threshold: float = 85.0,
                      min_chunk_len: int = 150) -> list[str]:
    """Phân đoạn văn bản dựa trên khoảng cách ngữ nghĩa giữa các câu kế tiếp."""
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return sentences

    # 1. Tạo embedding cho từng câu
    embeddings = np.array(embed_fn(sentences))
    if len(embeddings.shape) != 2 or len(embeddings) != len(sentences):
        return sentences

    # Chuẩn hóa L2 vectors
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    norm_embeddings = embeddings / norms

    # 2. Tính khoảng cách Cosine Distance giữa các câu liên tiếp
    # cos_sim = u . v
    similarities = np.sum(norm_embeddings[:-1] * norm_embeddings[1:], axis=1)
    distances = 1.0 - similarities

    # 3. Xác định ngưỡng cắt (breakpoint threshold)
    threshold = np.percentile(distances, percentile_threshold)

    # 4. Gom cụm các câu thành các chunk
    chunks = []
    current_chunk = [sentences[0]]

    for i, dist in enumerate(distances):
        next_sent = sentences[i + 1]
        current_text_len = sum(len(s) for s in current_chunk)

        # Ngắt chunk nếu khoảng cách LỚN HƠN ngưỡng (vượt qua percentile) VÀ độ dài đạt mức tối thiểu
        if dist > threshold and current_text_len >= min_chunk_len:
            chunks.append(" ".join(current_chunk))
            current_chunk = [next_sent]
        else:
            current_chunk.append(next_sent)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def split_documents_semantically(docs: list[Document],
                                 embed_fn: Callable[[list[str]], np.ndarray],
                                 percentile_threshold: float = 85.0,
                                 min_chunk_len: int = 150,
                                 kb: list | None = None) -> list[Document]:
    """Áp dụng Semantic Chunking cho danh sách Documents và bảo toàn metadata."""
    semantic_docs = []
    for doc in docs:
        chunks = semantic_chunking(
            doc.page_content,
            embed_fn=embed_fn,
            percentile_threshold=percentile_threshold,
            min_chunk_len=min_chunk_len,
        )
        for i, chunk_text in enumerate(chunks):
            meta = dict(doc.metadata)
            meta["chunk_id"] = i
            c_doc = Document(page_content=chunk_text, metadata=meta)
            if kb:
                tag_disease(c_doc, kb)
            semantic_docs.append(c_doc)
    return semantic_docs
