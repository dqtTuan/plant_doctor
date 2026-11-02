from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from plant_doctor.rag.document_manager import DocumentManager
from plant_doctor.rag.index_store import FaissIndexManager


def chunk(t, src="a.md", uid=None):
    return Document(page_content=t, metadata={"source": src, "chunk_uid": uid or t})


def test_add_persists_across_reload(tmp_path, fake_emb):
    FaissIndexManager(tmp_path, fake_emb).add([chunk("x"), chunk("y")])
    assert len(FaissIndexManager(tmp_path, fake_emb).all_documents()) == 2


def test_add_is_idempotent(tmp_path, fake_emb):
    m = FaissIndexManager(tmp_path, fake_emb)
    assert m.add([chunk("x")]) == 1
    assert m.add([chunk("x")]) == 0


def test_delete_source_removes_only_that_file(tmp_path, fake_emb):
    m = FaissIndexManager(tmp_path, fake_emb)
    m.add([chunk("x", "a.md"), chunk("y", "b.md")])
    assert m.delete_source("a.md") == 1
    assert [d.metadata["source"] for d in FaissIndexManager(tmp_path, fake_emb).all_documents()] == ["b.md"]


def test_delete_source_on_legacy_index(tmp_path, fake_emb):
    FAISS.from_documents([Document(page_content="old", metadata={"source": "a.md"})], fake_emb).save_local(
        str(tmp_path)
    )
    assert FaissIndexManager(tmp_path, fake_emb).delete_source("a.md") == 1


def test_reupload_changed_file_replaces_chunks(tmp_path, fake_emb):
    kb, idx = tmp_path / "kb", FaissIndexManager(tmp_path / "idx", fake_emb)
    dm = DocumentManager(kb_dir=kb)
    dm.save_uploaded_file("a.txt", "Phiên bản một.".encode("utf-8"))
    dm.ingest_file("a.txt", idx, chunker_type="recursive")
    dm.save_uploaded_file("a.txt", "Phiên bản hai.".encode("utf-8"))
    dm.ingest_file("a.txt", idx, chunker_type="recursive")
    texts = [d.page_content for d in idx.all_documents()]
    assert texts == ["Phiên bản hai."]


def test_delete_document_removes_vectors(tmp_path, fake_emb):
    idx = FaissIndexManager(tmp_path / "idx", fake_emb)
    dm = DocumentManager(kb_dir=tmp_path / "kb")
    dm.save_uploaded_file("a.txt", "Nội dung.".encode("utf-8"))
    dm.ingest_file("a.txt", idx, chunker_type="recursive")
    assert dm.delete_document("a.txt", index=idx) is True
    assert idx.all_documents() == []
