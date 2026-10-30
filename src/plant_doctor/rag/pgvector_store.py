"""PostgreSQL Vector Store Module sử dụng extension pgvector.

Quản lý:
1. Tự động khởi tạo extension `vector` và bảng `plant_doctor_embeddings`.
2. Lưu trữ chunks văn bản, metadata JSONB và vector 384 chiều.
3. Tìm kiếm tương đồng Cosine Similarity bằng toán tử `<=>` của pgvector.
"""
import json
import logging
from typing import Any

from langchain_core.documents import Document

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:
    psycopg2 = None
    execute_values = None

logger = logging.getLogger(__name__)


class PGVectorStore:
    def __init__(self, connection_url: str, table_name: str = "plant_doctor_embeddings"):
        self.connection_url = connection_url
        self.table_name = table_name

    def _get_connection(self):
        if psycopg2 is None:
            raise RuntimeError("Thư viện psycopg2 chưa được cài đặt. Hãy chạy: pip install psycopg2-binary pgvector")
        return psycopg2.connect(self.connection_url)

    def init_db(self, vector_dim: int = 384) -> None:
        """Tự động kích hoạt extension vector và tạo bảng lưu trữ."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # 1. Kích hoạt extension pgvector
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                # 2. Tạo bảng lưu trữ vector
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id BIGSERIAL PRIMARY KEY,
                        content TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}',
                        embedding vector({vector_dim}) NOT NULL
                    );
                """)
                # 3. Tạo index IVFFlat hoặc HNSW nếu chưa có (khi bảng có dữ liệu lớn)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_name}_key 
                    ON {self.table_name} (((metadata->>'key')));
                """)
            conn.commit()

    def add_documents(self, docs: list[Document], embeddings: list[list[float]]) -> int:
        """Lưu danh sách Document và vector tương ứng vào PostgreSQL."""
        if not docs or not embeddings:
            return 0
        if len(docs) != len(embeddings):
            raise ValueError("Số lượng documents và embeddings phải bằng nhau.")

        self.init_db(vector_dim=len(embeddings[0]))

        records = [
            (doc.page_content, json.dumps(doc.metadata, ensure_ascii=False), list(emb))
            for doc, emb in zip(docs, embeddings)
        ]

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                query = f"""
                    INSERT INTO {self.table_name} (content, metadata, embedding)
                    VALUES %s
                """
                if execute_values is not None:
                    execute_values(cur, query, records, template="(%s, %s, %s::vector)")
                else:
                    for r in records:
                        cur.execute(f"INSERT INTO {self.table_name} (content, metadata, embedding) VALUES (%s, %s, %s::vector)", r)
            conn.commit()
        return len(records)

    def similarity_search(self, query_vector: list[float], k: int = 2, filter_key: str | None = None) -> list[Document]:
        """Tìm kiếm các document tương đồng nhất dựa trên Cosine Distance (<=>)."""
        vec_str = "[" + ",".join(str(float(x)) for x in query_vector) + "]"
        query = f"""
            SELECT content, metadata, (embedding <=> %s::vector) AS distance
            FROM {self.table_name}
        """
        params = [vec_str]

        if filter_key:
            query += " WHERE metadata->>'key' = %s "
            params.append(filter_key)

        query += f" ORDER BY distance ASC LIMIT {int(k)};"

        docs = []
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
                for row in rows:
                    content, meta_raw, _ = row
                    meta = meta_raw if isinstance(meta_raw, dict) else json.loads(meta_raw)
                    docs.append(Document(page_content=content, metadata=meta))
        return docs
