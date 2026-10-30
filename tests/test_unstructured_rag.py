import pytest
from langchain_core.documents import Document
from plant_doctor.rag.build_index import load_documents, chunk_documents_by_type


def test_load_documents_from_user_directory(tmp_path):
    user_file = tmp_path / "tai_lieu_lua.md"
    user_file.write_text("# Cẩm nang bệnh lúa\n\nBệnh đạo ôn lúa do nấm Pyricularia oryzae gây ra.", encoding="utf-8")
    
    docs = load_documents(tmp_path)
    assert len(docs) >= 1
    assert docs[0].metadata["source"] == "tai_lieu_lua.md"
    assert "đạo ôn lúa" in docs[0].page_content


def test_chunk_documents_without_any_json():
    doc = Document(
        page_content="Bệnh đốm vòng cà chua do nấm Alternaria solani. Cần dùng chế phẩm Trichoderma để kiểm soát.",
        metadata={"source": "manual.md"}
    )
    chunks = chunk_documents_by_type([doc], chunker_type="recursive", chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 1
    assert chunks[0].metadata["source"] == "manual.md"
