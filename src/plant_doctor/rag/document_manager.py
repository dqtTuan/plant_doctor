"""Trình quản lý tài liệu tri thức người dùng (User Document Ingestion Manager).

Cho phép lưu trữ, trích xuất văn bản từ PDF/MD/TXT, chia đoạn ngữ nghĩa
và cập nhật trực tiếp vào Vector Store mà không cần bất kỳ mock data nào.
"""
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from plant_doctor.config import KB_DIR
from plant_doctor.rag.build_index import chunk_documents_by_type
from plant_doctor.rag.loaders import load_file


class DocumentManager:
    def __init__(self, kb_dir: Path | str = KB_DIR):
        self.kb_dir = Path(kb_dir)
        self.kb_dir.mkdir(parents=True, exist_ok=True)

    def save_uploaded_file(self, file_name: str, file_bytes: bytes) -> Path:
        """Lưu file người dùng vào thư mục tri thức một cách an toàn."""
        clean_name = Path(file_name).name
        dest = self.kb_dir / clean_name
        dest.write_bytes(file_bytes)
        return dest

    def list_user_documents(self) -> list[dict]:
        """Liệt kê toàn bộ các tài liệu thực tế trong kho tri thức."""
        results = []
        for p in sorted(self.kb_dir.iterdir()):
            if p.name.lower().startswith("readme"):
                continue
            if p.is_file() and p.suffix.lower() in (".pdf", ".md", ".txt"):
                stat = p.stat()
                results.append({
                    "name": p.name,
                    "path": str(p),
                    "size_kb": round(stat.st_size / 1024, 2),
                    "modified": stat.st_mtime,
                    "extension": p.suffix.lower(),
                })
        return results

    def delete_document(self, file_name: str, index: Any = None) -> bool:
        """Xóa một tài liệu khỏi kho tri thức và đồng bộ xóa vector nếu có index."""
        clean_name = Path(file_name).name
        target = self.kb_dir / clean_name
        if target.exists() and target.is_file():
            target.unlink()
            if index is not None and hasattr(index, "delete_source"):
                index.delete_source(clean_name)
            return True
        return False

    def load_file_documents(self, file_name: str) -> list[Document]:
        """Trích xuất danh sách Document từ một file cụ thể."""
        target = self.kb_dir / Path(file_name).name
        return load_file(target)

    def ingest_file(
        self,
        file_name: str,
        vector_store: Any,
        embeddings: Any = None,
        chunker_type: str = "semantic",
        max_tokens: int | None = None,
        count_tokens: Any = None,
    ) -> int:
        """Phân đoạn ngữ nghĩa và thêm tài liệu vào Vector Store trực tiếp."""
        clean_name = Path(file_name).name
        raw_docs = self.load_file_documents(clean_name)
        if not raw_docs:
            return 0

        # Xóa vector cũ của file này nếu có để tránh trùng lặp
        if hasattr(vector_store, "delete_source"):
            vector_store.delete_source(clean_name)

        embed_fn = None
        if embeddings and hasattr(embeddings, "embed_documents"):
            embed_fn = embeddings.embed_documents
        elif hasattr(vector_store, "embeddings") and hasattr(vector_store.embeddings, "embed_documents"):
            embed_fn = vector_store.embeddings.embed_documents

        chunks = chunk_documents_by_type(
            raw_docs,
            chunker_type=chunker_type,
            embed_fn=embed_fn,
            max_tokens=max_tokens,
            count_tokens=count_tokens,
        )

        if hasattr(vector_store, "add"):
            return vector_store.add(chunks)

        from plant_doctor.rag.pgvector_store import PGVectorStore

        if isinstance(vector_store, PGVectorStore):
            active_emb = embeddings or getattr(vector_store, "embeddings", None)
            if active_emb is None:
                raise ValueError("Cần cung cấp embeddings cho PGVectorStore")
            doc_texts = [c.page_content for c in chunks]
            vecs = active_emb.embed_documents(doc_texts)
            return vector_store.add_documents(chunks, vecs)
        else:
            # FAISS cũ
            vector_store.add_documents(chunks)
            return len(chunks)
