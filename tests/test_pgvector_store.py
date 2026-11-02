"""Kiểm thử unit test cho PGVectorStore (với mock connection)."""
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from plant_doctor.rag.pgvector_store import PGVectorStore


@patch("plant_doctor.rag.pgvector_store.psycopg2")
def test_pgvector_store_init_and_search(mock_psycopg2):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    # with conn -> returns mock_conn
    mock_conn.__enter__.return_value = mock_conn
    # with conn.cursor() -> returns mock_cursor
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_psycopg2.connect.return_value = mock_conn

    # Mock kết quả trả về từ DB: [(content, metadata_json, distance)]
    mock_cursor.fetchall.return_value = [
        ("Nội dung phác đồ gỉ sắt cà phê", '{"key": "coffee_leaf_rust", "source": "guide.txt"}', 0.12)
    ]

    store = PGVectorStore(connection_url="postgresql://user:pass@localhost:5432/db")
    store.init_db()

    # Kiểm tra gọi lệnh tạo extension và tạo bảng
    calls = [call[0][0] for call in mock_cursor.execute.call_args_list]
    assert any("CREATE EXTENSION IF NOT EXISTS vector" in c for c in calls)
    assert any("CREATE TABLE IF NOT EXISTS plant_doctor_embeddings" in c for c in calls)

    # Test tìm kiếm
    results = store.similarity_search([0.1] * 384, k=1, filter_key="coffee_leaf_rust")
    assert len(results) == 1
    assert "gỉ sắt" in results[0].page_content
    assert results[0].metadata["key"] == "coffee_leaf_rust"
