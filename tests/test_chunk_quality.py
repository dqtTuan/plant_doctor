from langchain_core.documents import Document
from plant_doctor.rag.build_index import chunk_documents_by_type
from plant_doctor.rag.semantic_chunking import cap_chunks, embedded_text, split_markdown_sections


def words(s):
    return len(s.split())


def test_markdown_breadcrumb():
    doc = Document(
        page_content="# Cà chua\n## Mốc sương\n### Triệu chứng\nVết úng nước.",
        metadata={"source": "a.md"},
    )
    [s] = split_markdown_sections(doc)
    assert s.metadata["breadcrumb"] == "Cà chua › Mốc sương › Triệu chứng"
    assert embedded_text(s).startswith("Cà chua › Mốc sương › Triệu chứng\n")


def test_cap_splits_long_chunk_on_sentences():
    doc = Document(page_content=" ".join(["Câu số một dài."] * 50), metadata={"source": "a.txt"})
    out = cap_chunks([doc], max_tokens=20, count_tokens=words)
    assert len(out) > 1
    assert all(words(embedded_text(c)) <= 20 for c in out)


def test_cap_hard_splits_unpunctuated_text():
    doc = Document(page_content="lua " * 300, metadata={"source": "t.pdf", "page": 2})
    out = cap_chunks([doc], max_tokens=50, count_tokens=words)
    assert all(words(c.page_content) <= 50 for c in out)
    assert out[0].metadata["page"] == 2


def test_pipeline_chunks_carry_uid_and_breadcrumb_text():
    doc = Document(
        page_content="# Lúa\n## Đạo ôn\n" + "Vết bệnh hình thoi. " * 40,
        metadata={"source": "a.md", "doc_id": "d1"},
    )
    chunks = chunk_documents_by_type([doc], chunker_type="recursive", max_tokens=40, count_tokens=words)
    assert all(c.page_content.startswith("Lúa › Đạo ôn\n") for c in chunks)
    assert all(words(c.page_content) <= 40 for c in chunks)
    assert len({c.metadata["chunk_uid"] for c in chunks}) == len(chunks)
