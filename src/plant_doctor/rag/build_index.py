"""Bước 4: Xây dựng kho tri thức vector cục bộ (FAISS).

Nguồn tài liệu trong data/knowledge_base/:
  - diseases.json   : mỗi bệnh = 1 tài liệu đầy đủ, gắn metadata key (khớp với nhãn YOLO)
  - *.pdf, *.md, *.txt : tài liệu bạn tự biên soạn -> cắt đoạn 1000 ký tự, gối đầu 200
Đoạn PDF có nhắc tên bệnh sẽ được gắn cùng key để truy xuất có lọc theo chẩn đoán.

python -m plant_doctor.rag.build_index
"""
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from plant_doctor.config import EMBED_DIR, FAISS_DIR, KB_DIR
from plant_doctor.knowledge import entry_to_text, load_kb, title


def load_files():
    docs = []
    for path in sorted(KB_DIR.rglob("*")):
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


def tag_disease(doc, kb):
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


def main():
    kb = load_kb()
    docs = [Document(page_content=entry_to_text(e),
                     metadata={"key": e["key"], "source": "diseases.json", "title": title(e)})
            for e in kb]

    file_docs = load_files()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(file_docs)
    for c in chunks:
        tag_disease(c, kb)
    docs += chunks
    print(f"{len(kb)} mục từ diseases.json + {len(chunks)} đoạn từ {len(file_docs)} trang/tệp tài liệu")

    if not (EMBED_DIR / "config.json").exists() and not (EMBED_DIR / "modules.json").exists():
        raise SystemExit(f"Chưa có mô hình embedding tại {EMBED_DIR}. "
                         "Hãy chạy `python -m plant_doctor.training.train_embedding`")
    embeddings = HuggingFaceEmbeddings(model_name=str(EMBED_DIR),
                                       encode_kwargs={"normalize_embeddings": True})
    # Vector đã chuẩn hóa l2 + tích vô hướng = độ tương đồng Cosine
    store = FAISS.from_documents(docs, embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    store.save_local(str(FAISS_DIR))
    print(f"Đã đóng gói {len(docs)} tài liệu vào FAISS tại: {FAISS_DIR}")


if __name__ == "__main__":
    main()
