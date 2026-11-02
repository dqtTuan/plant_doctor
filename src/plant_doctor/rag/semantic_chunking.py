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

from langchain_core.documents import Document
import numpy as np


def tag_disease(doc: Document, kb: list) -> None:
    """Gắn metadata key cho đoạn tài liệu nếu có nhắc tên cây và tên bệnh."""
    text = doc.page_content.lower()
    for e in kb:
        if e["key"] == "healthy":
            continue
        crop_hit = e["crop_vi"].lower() in text or e["crop_en"].lower() in text
        name_hit = e["name_vi"].lower() in text or e["name_en"].lower() in text
        if crop_hit and name_hit:
            doc.metadata["key"] = e["key"]
            return
    doc.metadata["key"] = ""


def split_sentences(text: str) -> list[str]:
    """Tách văn bản thành danh sách các câu đơn lẻ, xử lý dấu chấm câu thông minh."""
    clean = text.strip()
    if not clean:
        return []

    # Tách theo ngắt dòng kép hoặc các dấu kết thúc câu (. ? !)
    pattern = r"(?<=[.?!])\s+(?=[A-ZÀ-Ỹ0-9])|\n\n+"
    raw_sents = re.split(pattern, clean)

    sentences = []
    for s in raw_sents:
        s_clean = s.strip()
        if s_clean:
            sentences.append(s_clean)
    return sentences if sentences else [clean]


def split_markdown_sections(doc: Document) -> list[Document]:
    """Tách tài liệu Markdown theo các tiêu đề H1, H2, H3 và tạo breadcrumb."""
    source = doc.metadata.get("source", "").lower()
    if not source.endswith(".md"):
        return [doc]

    from langchain_text_splitters import MarkdownHeaderTextSplitter

    headers_to_split = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split, strip_headers=True)
    sub_docs = splitter.split_text(doc.page_content)
    if not sub_docs:
        return [doc]

    results = []
    for sd in sub_docs:
        meta = dict(doc.metadata)
        breadcrumbs = [
            sd.metadata.get(k, "").strip()
            for k in ("h1", "h2", "h3")
            if sd.metadata.get(k, "").strip()
        ]
        breadcrumb = " › ".join(breadcrumbs)
        if breadcrumb:
            meta["breadcrumb"] = breadcrumb
        for k, v in sd.metadata.items():
            meta[k] = v
        results.append(Document(page_content=sd.page_content, metadata=meta))
    return results


def embedded_text(doc: Document) -> str:
    """Trả về văn bản đầy đủ kèm breadcrumb ngữ cảnh phục vụ sinh vector và hiển thị."""
    breadcrumb = doc.metadata.get("breadcrumb")
    content = doc.page_content
    if breadcrumb:
        prefix = f"{breadcrumb}\n"
        if not content.startswith(prefix):
            return f"{prefix}{content}"
    return content


def cap_chunks(
    chunks: list[Document],
    max_tokens: int,
    count_tokens: Callable[[str], int],
) -> list[Document]:
    """Giới hạn độ dài token của chunk để không vượt quá giới hạn của mô hình embedding."""
    capped_docs = []
    for doc in chunks:
        full_text = embedded_text(doc)
        if count_tokens(full_text) <= max_tokens:
            capped_docs.append(doc)
            continue

        breadcrumb = doc.metadata.get("breadcrumb", "")
        b_prefix = f"{breadcrumb}\n" if breadcrumb else ""

        sentences = split_sentences(doc.page_content)
        expanded_sentences = []
        for s in sentences:
            test_s = f"{b_prefix}{s}" if b_prefix else s
            if count_tokens(test_s) <= max_tokens:
                expanded_sentences.append(s)
            else:
                words = s.split()
                buf = []
                for w in words:
                    cand = (" ".join(buf + [w])).strip()
                    test_cand = f"{b_prefix}{cand}" if b_prefix else cand
                    if count_tokens(test_cand) <= max_tokens:
                        buf.append(w)
                    else:
                        if buf:
                            expanded_sentences.append(" ".join(buf))
                            buf = [w]
                        else:
                            expanded_sentences.append(w)
                            buf = []
                if buf:
                    expanded_sentences.append(" ".join(buf))

        current_group = []
        for s in expanded_sentences:
            cand_text = " ".join(current_group + [s]).strip()
            test_cand = f"{b_prefix}{cand_text}" if b_prefix else cand_text
            if count_tokens(test_cand) <= max_tokens:
                current_group.append(s)
            else:
                if current_group:
                    piece_text = " ".join(current_group).strip()
                    capped_docs.append(Document(page_content=piece_text, metadata=dict(doc.metadata)))
                    current_group = [s]
                else:
                    capped_docs.append(Document(page_content=s, metadata=dict(doc.metadata)))
                    current_group = []
        if current_group:
            piece_text = " ".join(current_group).strip()
            capped_docs.append(Document(page_content=piece_text, metadata=dict(doc.metadata)))
    return capped_docs


def semantic_chunking(
    text: str,
    embed_fn: Callable[[list[str]], np.ndarray],
    percentile_threshold: float = 85.0,
    min_chunk_len: int = 150,
) -> list[str]:
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

        if dist > threshold and current_text_len >= min_chunk_len:
            chunks.append(" ".join(current_chunk))
            current_chunk = [next_sent]
        else:
            current_chunk.append(next_sent)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def split_documents_semantically(
    docs: list[Document],
    embed_fn: Callable[[list[str]], np.ndarray],
    percentile_threshold: float = 85.0,
    min_chunk_len: int = 150,
    kb: list | None = None,
) -> list[Document]:
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
