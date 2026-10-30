import pytest
from pathlib import Path
from plant_doctor.rag.document_manager import DocumentManager


def test_save_uploaded_file(tmp_path):
    dm = DocumentManager(kb_dir=tmp_path)
    saved_path = dm.save_uploaded_file("so_tay_ca_chua.md", b"# Huong dan ca chua")
    assert saved_path.exists()
    assert saved_path.name == "so_tay_ca_chua.md"
    assert saved_path.read_text(encoding="utf-8") == "# Huong dan ca chua"


def test_list_user_documents(tmp_path):
    dm = DocumentManager(kb_dir=tmp_path)
    dm.save_uploaded_file("doc1.txt", b"Noi dung 1")
    dm.save_uploaded_file("doc2.md", b"Noi dung 2")
    docs = dm.list_user_documents()
    assert len(docs) == 2
    names = [d["name"] for d in docs]
    assert "doc1.txt" in names
    assert "doc2.md" in names
    assert all("size_kb" in d for d in docs)


def test_delete_document(tmp_path):
    dm = DocumentManager(kb_dir=tmp_path)
    dm.save_uploaded_file("test.txt", b"xoa toi di")
    assert dm.delete_document("test.txt") is True
    assert len(dm.list_user_documents()) == 0
    assert dm.delete_document("not_found.txt") is False


def test_load_file_documents(tmp_path):
    dm = DocumentManager(kb_dir=tmp_path)
    dm.save_uploaded_file("doc.txt", b"Benh dom la ca chua. Can cat tia.")
    langchain_docs = dm.load_file_documents("doc.txt")
    assert len(langchain_docs) == 1
    assert "dom la ca chua" in langchain_docs[0].page_content
    assert langchain_docs[0].metadata["source"] == "doc.txt"
