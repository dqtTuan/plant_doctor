"""Trình quản lý tài liệu tri thức người dùng (User Document Ingestion Manager).

Cho phép lưu trữ, trích xuất văn bản từ PDF/MD/TXT, chia đoạn ngữ nghĩa
và cập nhật trực tiếp vào Vector Store mà không cần bất kỳ mock data nào.
"""
from pathlib import Path
from typing import Any
from langchain_core.documents import Document

from plant_doctor.config import KB_DIR
from plant_doctor.rag.build_index import chunk_documents_by_type


class DocumentManager:
    def __init__(self, kb_dir: Path | str = KB_DIR):
        self.kb_dir = Path(kb_dir)
        self.kb_dir.mkdir(parents=True, exist_ok=True)

    def save_uploaded_file(self, file_name: str, file_bytes: bytes) -> Path:
        """Lưu file người dùng vào thư mục tri thức một cách an toàn."""
        # Ngăn ngừa path traversal
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

    def delete_document(self, file_name: str) -> bool:
        """Xóa một tài liệu khỏi kho tri thức."""
        target = self.kb_dir / Path(file_name).name
        if target.exists() and target.is_file():
            target.unlink()
            return True
        return False

    def load_file_documents(self, file_name: str) -> list[Document]:
        """Trích xuất danh sách Document từ một file cụ thể."""
        target = self.kb_dir / Path(file_name).name
        if not target.exists():
            return []

        suffix = target.suffix.lower()
        docs = []
        if suffix == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(str(target))
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": target.name, "page": i + 1}
                    ))
        elif suffix in (".md", ".txt"):
            text = target.read_text(encoding="utf-8")
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={"source": target.name}
                ))
        return docs

    def ingest_file(self, file_name: str, vector_store: Any, embeddings: Any, chunker_type: str = "semantic") -> int:
        """Phân đoạn ngữ nghĩa và thêm tài liệu vào Vector Store trực tiếp."""
        raw_docs = self.load_file_documents(file_name)
        if not raw_docs:
            return 0

        chunks = chunk_documents_by_type(
            raw_docs,
            chunker_type=chunker_type,
            embed_fn=embeddings.embed_documents if hasattr(embeddings, "embed_documents") else None,
        )

        from plant_doctor.rag.pgvector_store import PGVectorStore
        if isinstance(vector_store, PGVectorStore):
            doc_texts = [c.page_content for c in chunks]
            vecs = embeddings.embed_documents(doc_texts)
            return vector_store.add_documents(chunks, vecs)
        else:
            # FAISS
            vector_store.add_documents(chunks)
            return len(chunks)
