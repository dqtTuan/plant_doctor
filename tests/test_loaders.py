from langchain_core.documents import Document
from plant_doctor.rag.loaders import assign_chunk_uids, load_directory, load_file


def test_load_file_md_has_source_and_doc_id(tmp_path):
    p = tmp_path / "a.md"
    p.write_text("# Lúa\nĐạo ôn", encoding="utf-8")
    [d] = load_file(p)
    assert d.metadata["source"] == "a.md"
    assert len(d.metadata["doc_id"]) == 16


def test_doc_id_changes_with_content(tmp_path):
    p = tmp_path / "a.md"
    p.write_text("v1", encoding="utf-8")
    id1 = load_file(p)[0].metadata["doc_id"]
    p.write_text("v2", encoding="utf-8")
    assert load_file(p)[0].metadata["doc_id"] != id1


def test_load_directory_skips_readme_and_unsupported(tmp_path):
    (tmp_path / "README.md").write_text("x", encoding="utf-8")
    (tmp_path / "a.txt").write_text("y", encoding="utf-8")
    (tmp_path / "b.docx").write_bytes(b"z")
    assert [d.metadata["source"] for d in load_directory(tmp_path)] == ["a.txt"]


def test_chunk_uids_are_stable_and_unique():
    def mk():
        return [Document(page_content=t, metadata={"doc_id": "d1", "page": 1}) for t in ("a", "b")]

    a = assign_chunk_uids(mk())
    b = assign_chunk_uids(mk())
    assert [c.metadata["chunk_uid"] for c in a] == [c.metadata["chunk_uid"] for c in b]
    assert len({c.metadata["chunk_uid"] for c in a}) == 2
