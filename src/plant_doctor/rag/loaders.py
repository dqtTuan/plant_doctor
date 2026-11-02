"""Module trích xuất tài liệu thống nhất (Unified Loaders) và định danh chunk ổn định."""
from collections import defaultdict
import hashlib
from pathlib import Path

from langchain_core.documents import Document

SUPPORTED_SUFFIXES = (".pdf", ".md", ".txt")


def file_doc_id(path: Path) -> str:
    """Tạo mã băm SHA1 16 ký tự đại diện cho nội dung file."""
    return hashlib.sha1(path.read_bytes()).hexdigest()[:16]


def load_file(path: Path | str) -> list[Document]:
    """Trích xuất tài liệu Document từ một file cụ thể (.pdf, .md, .txt)."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return []
    suffix = p.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        return []

    doc_id = file_doc_id(p)
    docs = []
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(p))
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": p.name, "doc_id": doc_id, "page": i + 1},
                    )
                )
    elif suffix in (".md", ".txt"):
        text = p.read_text(encoding="utf-8")
        if text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": p.name, "doc_id": doc_id},
                )
            )
    return docs


def load_directory(kb_dir: Path | str) -> list[Document]:
    """Quét toàn bộ thư mục và trích xuất danh sách Document hợp lệ."""
    kb_path = Path(kb_dir)
    docs = []
    for path in sorted(kb_path.rglob("*")):
        if path.name.lower().startswith("readme"):
            continue
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            docs.extend(load_file(path))
    return docs


def assign_chunk_uids(chunks: list[Document]) -> list[Document]:
    """Gán mã định danh duy nhất ổn định chunk_uid cho từng đoạn tài liệu."""
    group_counts = defaultdict(int)
    for c in chunks:
        doc_id = c.metadata.get("doc_id") or c.metadata.get("source", "")
        page = c.metadata.get("page") or 0
        idx = group_counts[(doc_id, page)]
        group_counts[(doc_id, page)] += 1
        raw_key = f"{doc_id}:{page}:{idx}".encode("utf-8")
        c.metadata["chunk_uid"] = hashlib.sha1(raw_key).hexdigest()[:16]
    return chunks
