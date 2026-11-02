"""Kiểm thử kỹ thuật Chunking và nạp tài liệu trong RAG."""
from langchain_core.documents import Document
from plant_doctor.rag.build_index import chunk_documents, load_documents


def test_chunk_documents_overlap_and_tagging():
    sample_text = ("Cây cà phê bị bệnh gỉ sắt xuất hiện vết bột màu vàng cam. " * 30)
    doc = Document(page_content=sample_text, metadata={"source": "test_guide.txt"})
    fake_kb = [{
        "key": "coffee_leaf_rust",
        "crop_vi": "Cà phê",
        "crop_en": "Coffee",
        "name_vi": "Bệnh gỉ sắt",
        "name_en": "Rust"
    }]
    chunks = chunk_documents([doc], chunk_size=300, chunk_overlap=50, kb=fake_kb)
    assert len(chunks) > 1
    assert all(c.metadata.get("key") == "coffee_leaf_rust" for c in chunks)
