"""Central Service Engine: PlantDoctorPipeline.

Gom toàn bộ logic AI:
- Nhận diện vết bệnh qua YOLOv8
- Ánh xạ nhãn & kho tri thức biểu tượng
- Truy xuất lai qua FAISS và Custom Embedding
- Sinh stream câu trả lời qua mô hình ngôn ngữ Qwen2.5 LoRA
- Cơ chế dự phòng an toàn (Graceful Fallback) khi thiếu model
"""
import logging
import os
from threading import Thread
from typing import Any, Iterator

import numpy as np
from PIL import Image
import torch

from plant_doctor.config import (DEFAULT_MAX_CONTEXT_DOCS, DEFAULT_MAX_NEW_TOKENS,
                                 EMBED_DIR, FAISS_DIR, LLM_MERGED_DIR, PGVECTOR_TABLE,
                                 PGVECTOR_URL, RERANKER_DIR, RETRIEVE_CANDIDATES, YOLO_WEIGHTS)
from plant_doctor.knowledge import entry_to_answer, entry_to_text, lookup
from plant_doctor.prompts import format_context

logger = logging.getLogger(__name__)

# Đảm bảo môi trường chạy offline
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class PlantDoctorPipeline:
    def __init__(self, load_models: bool = True):
        self.yolo = None
        self.vector_store = None
        self.index_manager = None
        self.bm25_index = None
        self.bm25_version = -1
        self.reranker = None
        self.pg_store = None
        self.tokenizer = None
        self.llm = None
        self.embeddings = None

        if load_models:
            self.load_all_models()

    def load_yolo(self) -> Any:
        if not YOLO_WEIGHTS.exists():
            return None
        from ultralytics import YOLO
        self.yolo = YOLO(str(YOLO_WEIGHTS))
        return self.yolo

    def load_vector_store(self) -> Any:
        # 1. Ưu tiên kết nối PostgreSQL (pgvector) nếu có URL
        if PGVECTOR_URL:
            try:
                from plant_doctor.rag.pgvector_store import PGVectorStore
                from langchain_huggingface import HuggingFaceEmbeddings
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=str(EMBED_DIR),
                    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
                self.pg_store = PGVectorStore(
                    connection_url=PGVECTOR_URL,
                    table_name=PGVECTOR_TABLE,
                    embeddings=self.embeddings,
                )
                self.vector_store = self.pg_store
                return self.vector_store
            except Exception as e:
                logger.warning("pgvector unavailable, falling back to FAISS: %s", e)

        # 2. Fallback sang FAISS cục bộ
        if not EMBED_DIR.exists():
            return None
        from langchain_huggingface import HuggingFaceEmbeddings
        from plant_doctor.rag.index_store import FaissIndexManager

        self.embeddings = HuggingFaceEmbeddings(
            model_name=str(EMBED_DIR),
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.index_manager = FaissIndexManager(FAISS_DIR, self.embeddings)
        self.vector_store = self.index_manager.store
        return self.vector_store

    def load_llm(self) -> tuple[Any, Any]:
        if not (LLM_MERGED_DIR / "config.json").exists():
            return None, None
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cuda = torch.cuda.is_available()
        self.tokenizer = AutoTokenizer.from_pretrained(str(LLM_MERGED_DIR))
        model = AutoModelForCausalLM.from_pretrained(
            str(LLM_MERGED_DIR),
            dtype=torch.float16 if cuda else torch.float32,
            device_map="auto" if cuda else None,
        )
        self.llm = model.eval()
        return self.tokenizer, self.llm

    def load_reranker(self) -> Any:
        from plant_doctor.rag.reranker import load_reranker

        self.reranker = load_reranker(RERANKER_DIR)
        return self.reranker

    def load_all_models(self) -> None:
        self.load_yolo()
        self.load_vector_store()
        self.load_reranker()
        self.load_llm()

    def diagnose_image(self, image: Image.Image, conf_thr: float = 0.40) -> tuple[list[tuple[str, float, dict | None]], np.ndarray | None]:
        """Chạy YOLO phát hiện vết bệnh và ánh xạ sang kho tri thức."""
        if self.yolo is None:
            return [], None

        res = self.yolo.predict(image, imgsz=640, conf=conf_thr, verbose=False)[0]
        best = {}
        for b in res.boxes:
            name, conf = self.yolo.names[int(b.cls[0])], float(b.conf[0])
            best[name] = max(conf, best.get(name, 0.0))

        diagnosis = [(n, c, lookup(n)) for n, c in sorted(best.items(), key=lambda x: -x[1])]
        annotated = res.plot()[:, :, ::-1]  # Convert BGR to RGB
        return diagnosis, annotated

    def retrieve_context(self, query: str, keys: list[str] | None = None, max_docs: int = DEFAULT_MAX_CONTEXT_DOCS) -> list:
        """Truy xuất lai (Hybrid BM25 + Dense Search) kèm lọc ngưỡng tin cậy."""
        keys = keys or []
        if self.vector_store is None:
            # Chế độ dự phòng khi thiếu Vector DB
            return [entry_to_text(e) for e in (lookup(k) for k in keys) if e]

        from plant_doctor.rag.query_expansion import label_to_query
        from plant_doctor.rag.retriever import BM25Index, HybridRetriever

        queries = [query]
        for k in keys:
            queries.append(label_to_query(k))
        rerank_query = query if not keys else f"{query}. {label_to_query(keys[0])}"

        # 1. Quản lý chỉ mục BM25 (chỉ khi có index_manager với FAISS)
        bm25 = None
        if self.index_manager is not None:
            if self.bm25_index is None or self.bm25_version != self.index_manager.version:
                docs = self.index_manager.all_documents()
                self.bm25_index = BM25Index(docs)
                self.bm25_version = self.index_manager.version
            bm25 = self.bm25_index

        # 2. Xây dựng dense search callable
        from plant_doctor.rag.pgvector_store import PGVectorStore
        if isinstance(self.vector_store, PGVectorStore):
            def dense_search(q: str, k: int):
                emb = self.embeddings.embed_query(q)
                return self.vector_store.similarity_search_with_score(emb, k=k)
        else:
            def dense_search(q: str, k: int):
                if hasattr(self.vector_store, "similarity_search_with_score"):
                    return self.vector_store.similarity_search_with_score(q, k=k)
                elif hasattr(self.vector_store, "similarity_search"):
                    docs = self.vector_store.similarity_search(q, k=k)
                    return [(d, 1.0) for d in docs]
                return []

        retriever = HybridRetriever(
            dense=dense_search,
            bm25=bm25,
            reranker=self.reranker,
            candidates=RETRIEVE_CANDIDATES,
            top_k=max_docs,
        )
        return retriever.retrieve(queries, rerank_query)

    def stream_answer(self, messages: list[dict], max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> Iterator[str]:
        """Sinh chuỗi câu trả lời streaming từ mô hình ngôn ngữ."""
        if self.llm is None or self.tokenizer is None:
            return

        from transformers import TextIteratorStreamer

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        Thread(target=self.llm.generate, kwargs=kwargs, daemon=True).start()
        yield from streamer

    def fallback_answer(self, diagnosis: list[tuple[str, float, dict | None]], texts: list[str]) -> str:
        """Chế độ dự phòng khi không có LLM: ưu tiên tài liệu thực tế của người dùng."""
        if texts:
            return "Chưa có mô hình ngôn ngữ cục bộ. Trích đoạn tài liệu từ kho tri thức của bạn:\n\n" + format_context(texts)
        entries = [e for _, _, e in diagnosis if e and ("symptoms" in e or "bio_treatment" in e)]
        if entries:
            return "\n\n---\n\n".join(entry_to_answer(e) for e in entries)
        return "Kho tri thức của bạn chưa có tài liệu về bệnh này. Vui lòng nạp thêm tài liệu chuyên ngành (PDF/MD/TXT) vào hệ thống."

    def get_cached_answer(self, question: str, diagnosis: list[tuple[str, float, dict | None]]) -> str | None:
        """Kiểm tra câu hỏi có phải mẫu xuất phác đồ chuẩn không để trả về kết quả ngay lập tức (0.01s)."""
        clean_q = question.strip().lower()
        if clean_q.startswith("mô hình yolov8 vừa phát hiện lá cây bị bệnh:"):
            # Câu hỏi xuất phác đồ tự động
            entries = [e for _, _, e in diagnosis if e]
            if entries:
                return self.fallback_answer(diagnosis, [])
        return None
