"""PostgreSQL Vector Store Module sử dụng extension pgvector.

Quản lý:
1. Tự động khởi tạo extension `vector` và bảng `plant_doctor_embeddings`.
2. Lưu trữ chunks văn bản, metadata JSONB, vector và chunk_uid duy nhất.
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
    def __init__(
        self,
        connection_url: str,
        table_name: str = "plant_doctor_embeddings",
        embeddings: Any = None,
    ):
        self.connection_url = connection_url
        self.table_name = table_name
        self.embeddings = embeddings

    def _get_connection(self):
        if psycopg2 is None:
            raise RuntimeError("Thư viện psycopg2 chưa được cài đặt. Hãy chạy: pip install psycopg2-binary pgvector")
        return psycopg2.connect(self.connection_url)

    def init_db(self, vector_dim: int = 384) -> None:
        """Tự động kích hoạt extension vector, tạo bảng lưu trữ và index HNSW."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # 1. Kích hoạt extension pgvector
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                # 2. Tạo bảng lưu trữ vector kèm chunk_uid duy nhất
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id BIGSERIAL PRIMARY KEY,
                        chunk_uid TEXT UNIQUE,
                        content TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}',
                        embedding vector({vector_dim}) NOT NULL
                    );
                """)
                # 3. Tạo index metadata key
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_name}_key
                    ON {self.table_name} (((metadata->>'key')));
                """)
                # 4. Tạo index HNSW cho vector cosine distance
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_name}_hnsw
                    ON {self.table_name} USING hnsw (embedding vector_cosine_ops);
                """)
            conn.commit()

    def add(self, chunks: list[Document]) -> int:
        """Nhúng và lưu danh sách Document vào PostgreSQL."""
        if not chunks:
            return 0
        if self.embeddings is None:
            raise ValueError("Cần cung cấp embeddings cho PGVectorStore để sử dụng phương thức add()")
        doc_texts = [c.page_content for c in chunks]
        vecs = self.embeddings.embed_documents(doc_texts)
        return self.add_documents(chunks, vecs)

    def add_documents(self, docs: list[Document], embeddings: list[list[float]]) -> int:
        """Lưu danh sách Document và vector tương ứng vào PostgreSQL (idempotent qua ON CONFLICT)."""
        if not docs or not embeddings:
            return 0
        if len(docs) != len(embeddings):
            raise ValueError("Số lượng documents và embeddings phải bằng nhau.")

        self.init_db(vector_dim=len(embeddings[0]))

        records = [
            (
                doc.metadata.get("chunk_uid") or f"{doc.metadata.get('source', '')}:{i}",
                doc.page_content,
                json.dumps(doc.metadata, ensure_ascii=False),
                list(emb),
            )
            for i, (doc, emb) in enumerate(zip(docs, embeddings))
        ]

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                query = f"""
                    INSERT INTO {self.table_name} (chunk_uid, content, metadata, embedding)
                    VALUES %s
                    ON CONFLICT (chunk_uid) DO NOTHING
                """
                if execute_values is not None:
                    execute_values(cur, query, records, template="(%s, %s, %s, %s::vector)")
                else:
                    for r in records:
                        cur.execute(
                            f"INSERT INTO {self.table_name} (chunk_uid, content, metadata, embedding) "
                            "VALUES (%s, %s, %s, %s::vector) ON CONFLICT (chunk_uid) DO NOTHING",
                            r,
                        )
            conn.commit()
        return len(records)

    def delete_by_source(self, source: str) -> int:
        """Xóa toàn bộ các bản ghi vector thuộc về file nguồn trong PostgreSQL."""
        count = 0
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self.table_name} WHERE metadata->>'source' = %s RETURNING id;",
                    (source,),
                )
                deleted_rows = cur.fetchall()
                count = len(deleted_rows)
            conn.commit()
        return count

    delete_source = delete_by_source

    def similarity_search_with_score(
        self, query_vector: list[float], k: int = 4
    ) -> list[tuple[Document, float]]:
        """Tìm kiếm document kèm điểm Cosine Similarity (1.0 - distance)."""
        vec_str = "[" + ",".join(str(float(x)) for x in query_vector) + "]"
        query = f"""
            SELECT content, metadata, (embedding <=> %s::vector) AS distance
            FROM {self.table_name}
            ORDER BY distance ASC LIMIT {int(k)};
        """
        results = []
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, [vec_str])
                rows = cur.fetchall()
                for row in rows:
                    content, meta_raw, distance = row
                    meta = meta_raw if isinstance(meta_raw, dict) else json.loads(meta_raw)
                    score = round(1.0 - float(distance), 4)
                    results.append((Document(page_content=content, metadata=meta), score))
        return results

    def similarity_search(
        self, query_vector: list[float], k: int = 2, filter_key: str | None = None
    ) -> list[Document]:
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
