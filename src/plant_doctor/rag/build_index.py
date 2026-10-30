"""Bước 4: Xây dựng kho tri thức vector cục bộ (FAISS).

Nguồn tài liệu trong data/knowledge_base/:
  - diseases.json   : mỗi bệnh = 1 tài liệu đầy đủ, gắn metadata key (khớp với nhãn YOLO)
  - *.pdf, *.md, *.txt : tài liệu bạn tự biên soạn -> cắt đoạn theo chunk_size, gối đầu chunk_overlap
Đoạn PDF/văn bản có nhắc tên bệnh sẽ được gắn cùng key để truy xuất có lọc theo chẩn đoán.

python -m plant_doctor.rag.build_index
"""
import argparse
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from plant_doctor.config import EMBED_DIR, FAISS_DIR, KB_DIR
from plant_doctor.knowledge import entry_to_text, load_kb, title


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


def tag_disease(doc: Document, kb: list) -> None:
    """Gắn metadata key cho đoạn tài liệu nếu có nhắc tên cây và tên bệnh."""
    text = doc.page_content.lower()
    for e in kb:
        if e["key"] == "healthy":
            continue
        crop_hit = e["crop_vi"].lower() in text or e["crop_en"].lower() in text
        name_hit = e["name_vi"].lower() in text or e["name_en"].lower() in text
        if crop_hit and name_hit:
            doc.metadata["key"] = e["key"]
            return
    doc.metadata["key"] = ""


def chunk_documents(docs: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200,
                    kb: list | None = None) -> list[Document]:
    """Cắt nhỏ tài liệu bằng RecursiveCharacterTextSplitter và tự động gắn thẻ metadata key."""
    if not docs:
        return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    if kb:
        for c in chunks:
            tag_disease(c, kb)
    return chunks


def build_faiss_index(kb_dir: Path | str = KB_DIR,
                      embed_dir: Path | str = EMBED_DIR,
                      output_dir: Path | str = FAISS_DIR,
                      chunk_size: int = 1000,
                      chunk_overlap: int = 200) -> int:
    """Đóng gói toàn bộ tri thức thành FAISS index cục bộ. Trả về tổng số documents đã index."""
    kb = load_kb()
    docs = [Document(page_content=entry_to_text(e),
                     metadata={"key": e["key"], "source": "diseases.json", "title": title(e)})
            for e in kb]

    file_docs = load_documents(kb_dir)
    chunks = chunk_documents(file_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap, kb=kb)
    docs += chunks
    print(f"{len(kb)} mục từ diseases.json + {len(chunks)} đoạn từ {len(file_docs)} trang/tệp tài liệu")

    embed_path = Path(embed_dir)
    if not (embed_path / "config.json").exists() and not (embed_path / "modules.json").exists():
        raise SystemExit(f"Chưa có mô hình embedding tại {embed_path}. "
                         "Hãy chạy `python -m plant_doctor.training.train_embedding`")

    embeddings = HuggingFaceEmbeddings(model_name=str(embed_path),
                                       encode_kwargs={"normalize_embeddings": True})
    store = FAISS.from_documents(docs, embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(out_path))
    print(f"Đã đóng gói {len(docs)} tài liệu vào FAISS tại: {out_path}")
    return len(docs)


def main():
    ap = argparse.ArgumentParser(description="Xây dựng Vector Store FAISS cục bộ")
    ap.add_argument("--kb-dir", default=str(KB_DIR), help="Thư mục chứa tài liệu tri thức")
    ap.add_argument("--embed-dir", default=str(EMBED_DIR), help="Thư mục mô hình embedding")
    ap.add_argument("--out-dir", default=str(FAISS_DIR), help="Thư mục lưu faiss index")
    ap.add_argument("--chunk-size", type=int, default=1000, help="Kích thước chunk tối đa")
    ap.add_argument("--chunk-overlap", type=int, default=200, help="Độ gối đầu giữa các chunk")
    args = ap.parse_args()

    build_faiss_index(
        kb_dir=args.kb_dir,
        embed_dir=args.embed_dir,
        output_dir=args.out_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )


if __name__ == "__main__":
    main()
