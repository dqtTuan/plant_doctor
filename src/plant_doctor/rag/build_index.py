"""Bước 4: Xây dựng kho tri thức vector cục bộ (FAISS) hoặc PostgreSQL (pgvector).

Hỗ trợ:
- Chunking: Semantic Chunking (ngữ nghĩa) hoặc Recursive (kích thước cố định).
- Vector DB: FAISS (cục bộ) hoặc PostgreSQL (pgvector extension khi có PGVECTOR_URL).

python -m plant_doctor.rag.build_index --chunker-type semantic
"""
import argparse
from pathlib import Path

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from plant_doctor.config import EMBED_DIR, FAISS_DIR, KB_DIR, PGVECTOR_TABLE, PGVECTOR_URL
from plant_doctor.knowledge import entry_to_text, load_kb, title
from plant_doctor.rag.semantic_chunking import split_documents_semantically, tag_disease


def load_documents(kb_dir: Path | str = KB_DIR) -> list[Document]:
    """Nạp tất cả các tài liệu file rời (.pdf, .md, .txt) từ thư mục tri thức."""
    kb_path = Path(kb_dir)
    docs = []
    for path in sorted(kb_path.rglob("*")):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            from pypdf import PdfReader
            for i, page in enumerate(PdfReader(str(path)).pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(Document(page_content=text, metadata={"source": path.name, "page": i + 1}))
        elif suffix in (".md", ".txt"):
            docs.append(Document(page_content=path.read_text(encoding="utf-8"),
                                 metadata={"source": path.name}))
    return docs


def chunk_documents(docs: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200,
                    kb: list | None = None) -> list[Document]:
    """Cắt nhỏ tài liệu bằng RecursiveCharacterTextSplitter và tự động gắn thẻ metadata key."""
    return chunk_documents_by_type(docs, chunker_type="recursive", chunk_size=chunk_size, chunk_overlap=chunk_overlap, kb=kb)


def chunk_documents_by_type(docs: list[Document],
                           chunker_type: str = "semantic",
                           chunk_size: int = 1000,
                           chunk_overlap: int = 200,
                           embed_fn = None,
                           kb: list | None = None) -> list[Document]:
    """Hỗ trợ phân đoạn tài liệu theo Semantic Chunking hoặc Recursive Splitting."""
    if not docs:
        return []

    if chunker_type == "semantic" and embed_fn is not None:
        return split_documents_semantically(docs, embed_fn=embed_fn, kb=kb)

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    if kb:
        for c in chunks:
            tag_disease(c, kb)
    return chunks


def build_vector_store(kb_dir: Path | str = KB_DIR,
                       embed_dir: Path | str = EMBED_DIR,
                       output_dir: Path | str = FAISS_DIR,
                       chunker_type: str = "semantic",
                       chunk_size: int = 1000,
                       chunk_overlap: int = 200,
                       pgvector_url: str = PGVECTOR_URL) -> int:
    """Đóng gói toàn bộ tri thức thành FAISS index cục bộ hoặc PostgreSQL (pgvector)."""
    file_docs = load_documents(kb_dir)
    if not file_docs:
        print(f"Cảnh báo: Không tìm thấy tài liệu nào trong {kb_dir}")

    embed_path = Path(embed_dir)
    if not (embed_path / "config.json").exists() and not (embed_path / "modules.json").exists():
        raise SystemExit(f"Chưa có mô hình embedding tại {embed_path}. "
                         "Hãy chạy `python -m plant_doctor.training.train_embedding`")

    embeddings = HuggingFaceEmbeddings(model_name=str(embed_path),
                                       encode_kwargs={"normalize_embeddings": True})

    kb = load_kb()
    chunks = chunk_documents_by_type(
        file_docs,
        chunker_type=chunker_type,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embed_fn=embeddings.embed_documents,
        kb=kb if kb else None,
    )
    docs = chunks
    print(f"Tổng số tài liệu sẵn sàng index ({chunker_type} chunking): {len(docs)}")

    if pgvector_url:
        print(f"Đang lưu vector vào PostgreSQL (pgvector)...")
        from plant_doctor.rag.pgvector_store import PGVectorStore
        pg_store = PGVectorStore(connection_url=pgvector_url, table_name=PGVECTOR_TABLE)
        doc_texts = [d.page_content for d in docs]
        vecs = embeddings.embed_documents(doc_texts)
        count = pg_store.add_documents(docs, vecs)
        print(f"Đã lưu thành công {count} vector kèm metadata vào PostgreSQL!")
        return count
    else:
        # Lưu vào FAISS cục bộ
        store = FAISS.from_documents(docs, embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        store.save_local(str(out_path))
        print(f"Đã đóng gói {len(docs)} tài liệu vào FAISS tại: {out_path}")
        return len(docs)


def main():
    ap = argparse.ArgumentParser(description="Xây dựng Vector Store (FAISS hoặc PostgreSQL pgvector)")
    ap.add_argument("--kb-dir", default=str(KB_DIR), help="Thư mục chứa tài liệu tri thức")
    ap.add_argument("--embed-dir", default=str(EMBED_DIR), help="Thư mục mô hình embedding")
    ap.add_argument("--out-dir", default=str(FAISS_DIR), help="Thư mục lưu faiss index")
    ap.add_argument("--chunker-type", choices=["semantic", "recursive"], default="semantic",
                    help="Phương pháp chia đoạn: semantic (ngữ nghĩa) hoặc recursive (cố định)")
    ap.add_argument("--chunk-size", type=int, default=1000, help="Kích thước chunk (nếu dùng recursive)")
    ap.add_argument("--chunk-overlap", type=int, default=200, help="Độ gối đầu giữa các chunk")
    ap.add_argument("--pgvector-url", default=PGVECTOR_URL, help="URL kết nối PostgreSQL pgvector (nếu có)")
    args = ap.parse_args()

    build_vector_store(
        kb_dir=args.kb_dir,
        embed_dir=args.embed_dir,
        output_dir=args.out_dir,
        chunker_type=args.chunker_type,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        pgvector_url=args.pgvector_url,
    )


if __name__ == "__main__":
    main()
