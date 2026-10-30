import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from plant_doctor.rag.build_index import load_documents, chunk_documents_by_type
from plant_doctor.config import KB_DIR


def test_load_documents_from_markdown():
    docs = load_documents(KB_DIR)
    assert len(docs) >= 1
    # Kiểm tra đã nạp cẩm nang markdown
    assert any("cam_nang" in d.metadata.get("source", "").lower() for d in docs)
    # Nội dung có chứa tri thức
    sample = [d for d in docs if "cam_nang" in d.metadata.get("source", "").lower()][0]
    assert "CẨM NANG" in sample.page_content


def test_chunk_documents_without_any_json():
    doc = Document(
        page_content="Bệnh đốm vòng cà chua do nấm Alternaria solani. Cần dùng chế phẩm Trichoderma để kiểm soát.",
        metadata={"source": "manual.md"}
    )
    chunks = chunk_documents_by_type([doc], chunker_type="recursive", chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 1
    assert chunks[0].metadata["source"] == "manual.md"
