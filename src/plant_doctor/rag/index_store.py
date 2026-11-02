"""Trình quản lý vòng đời FAISS Index (lưu trữ, chống trùng, xóa theo tài liệu)."""
import logging
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class FaissIndexManager:
    def __init__(self, index_dir: Path | str, embeddings: Any):
        self.index_dir = Path(index_dir)
        self._embeddings = embeddings
        self._store: FAISS | None = None
        self.version = 0

        self._load_if_exists()

    @property
    def store(self) -> FAISS | None:
        return self._store

    @property
    def embeddings(self) -> Any:
        return self._embeddings

    def _load_if_exists(self) -> None:
        if (self.index_dir / "index.faiss").exists():
            try:
                self._store = FAISS.load_local(
                    str(self.index_dir),
                    self._embeddings,
                    allow_dangerous_deserialization=True,
                    distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
                )
            except Exception as e:
                logger.warning("Không thể nạp FAISS index tại %s: %s", self.index_dir, e)
                self._store = None

    def save(self) -> None:
        if self._store is not None:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            self._store.save_local(str(self.index_dir))

    def all_documents(self) -> list[Document]:
        if self._store is None or not hasattr(self._store, "docstore"):
            return []
        return list(self._store.docstore._dict.values())

    def add(self, chunks: list[Document]) -> int:
        """Thêm danh sách chunk vào FAISS với định danh chunk_uid duy nhất (idempotent)."""
        if not chunks:
            return 0

        existing_ids = set()
        if self._store is not None and hasattr(self._store, "docstore"):
            existing_ids = set(self._store.docstore._dict.keys())

        new_chunks = []
        new_ids = []
        for c in chunks:
            uid = c.metadata.get("chunk_uid") or c.metadata.get("source") or c.page_content[:32]
            if uid not in existing_ids:
                existing_ids.add(uid)
                new_chunks.append(c)
                new_ids.append(uid)

        if not new_chunks:
            return 0

        if self._store is None:
            self._store = FAISS.from_documents(
                new_chunks,
                self._embeddings,
                ids=new_ids,
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
            )
        else:
            self._store.add_documents(new_chunks, ids=new_ids)

        self.save()
        self.version += 1
        return len(new_chunks)

    def delete_source(self, source: str) -> int:
        """Xóa toàn bộ các vector và chunk thuộc về một file nguồn cụ thể."""
        if self._store is None or not hasattr(self._store, "docstore"):
            return 0

        ids_to_delete = []
        for doc_id, doc in list(self._store.docstore._dict.items()):
            if doc.metadata.get("source") == source:
                ids_to_delete.append(doc_id)

        if not ids_to_delete:
            return 0

        try:
            self._store.delete(ids_to_delete)
        except Exception as e:
            logger.warning("Lỗi khi xóa vector FAISS theo ID: %s", e)
            # Xóa thủ công khỏi docstore nếu hàm delete của FAISS gặp trục trặc
            for did in ids_to_delete:
                self._store.docstore._dict.pop(did, None)

        self.save()
        self.version += 1
        return len(ids_to_delete)
