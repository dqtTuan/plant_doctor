"""Kiểm thử thuật toán Semantic Chunking thuần túy."""
import numpy as np
from langchain_core.documents import Document
from plant_doctor.rag.semantic_chunking import (semantic_chunking,
                                                split_documents_semantically,
                                                split_sentences)


def test_split_sentences():
    text = "Cây cà phê bị bệnh gỉ sắt. Lá xuất hiện đốm vàng? Cần tỉa cành ngay!"
    sentences = split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "Cây cà phê bị bệnh gỉ sắt."


def test_semantic_chunking_with_mock_embedding():
    # 4 câu: 2 câu đầu chủ đề A [1, 0], 2 câu sau chủ đề B [0, 1]
    text = "Câu một về cây. Câu hai về cây. Chuyển đề tài về thuốc trừ sâu. Thuốc này rất độc hại."
    def mock_embed(sentences):
        return np.array([
            [1.0, 0.0],
            [0.9, 0.1],
            [0.1, 0.9],
            [0.0, 1.0],
        ])
    chunks = semantic_chunking(text, embed_fn=mock_embed, percentile_threshold=50.0, min_chunk_len=10)
    assert len(chunks) == 2
    assert "Câu một" in chunks[0]
    assert "thuốc trừ sâu" in chunks[1].lower()


def test_split_documents_semantically_with_metadata():
    doc = Document(page_content="Cây cà phê bị bệnh gỉ sắt. Vết bệnh màu cam. Cần phun Bordeaux phòng bệnh.",
                   metadata={"source": "manual.txt", "page": 1})
    fake_kb = [{"key": "coffee_leaf_rust", "crop_vi": "Cà phê", "crop_en": "Coffee",
                "name_vi": "Bệnh gỉ sắt", "name_en": "Rust"}]
    mock_embed = lambda sents: np.ones((len(sents), 4))
    res = split_documents_semantically([doc], embed_fn=mock_embed, kb=fake_kb)
    assert len(res) >= 1
    assert res[0].metadata["source"] == "manual.txt"
    assert res[0].metadata.get("key") == "coffee_leaf_rust"
