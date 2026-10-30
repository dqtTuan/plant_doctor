"""Central Service Engine: PlantDoctorPipeline.

Gom toàn bộ logic AI:
- Nhận diện vết bệnh qua YOLOv8
- Ánh xạ nhãn & kho tri thức biểu tượng
- Truy xuất lai qua FAISS và Custom Embedding
- Sinh stream câu trả lời qua mô hình ngôn ngữ Qwen2.5 LoRA
- Cơ chế dự phòng an toàn (Graceful Fallback) khi thiếu model
"""
import os
from threading import Thread
from typing import Any, Iterator

import numpy as np
import torch
from PIL import Image

from plant_doctor.config import (DEFAULT_MAX_CONTEXT_DOCS, DEFAULT_MAX_NEW_TOKENS,
                                 EMBED_DIR, FAISS_DIR, LLM_MERGED_DIR, YOLO_WEIGHTS)
from plant_doctor.knowledge import entry_to_answer, entry_to_text, is_healthy, lookup
from plant_doctor.prompts import build_messages, format_context

# Đảm bảo môi trường chạy offline
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class PlantDoctorPipeline:
    def __init__(self, load_models: bool = True):
        self.yolo = None
        self.vector_store = None
        self.tokenizer = None
        self.llm = None

        if load_models:
            self.load_all_models()

    def load_yolo(self) -> Any:
        if not YOLO_WEIGHTS.exists():
            return None
        from ultralytics import YOLO
        self.yolo = YOLO(str(YOLO_WEIGHTS))
        return self.yolo

    def load_vector_store(self) -> Any:
        if not FAISS_DIR.exists() or not EMBED_DIR.exists():
            return None
        from langchain_community.vectorstores import FAISS
        from langchain_community.vectorstores.utils import DistanceStrategy
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name=str(EMBED_DIR),
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vector_store = FAISS.load_local(
            str(FAISS_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        )
        return self.vector_store

    def load_llm(self) -> tuple[Any, Any]:
        if not (LLM_MERGED_DIR / "config.json").exists():
            return None, None
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cuda = torch.cuda.is_available()
        self.tokenizer = AutoTokenizer.from_pretrained(str(LLM_MERGED_DIR))
        model = AutoModelForCausalLM.from_pretrained(
            str(LLM_MERGED_DIR),
            torch_dtype=torch.float16 if cuda else torch.float32,
            device_map="auto" if cuda else None,
        )
        self.llm = model.eval()
        return self.tokenizer, self.llm

    def load_all_models(self) -> None:
        self.load_yolo()
        self.load_vector_store()
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

    def retrieve_context(self, query: str, keys: list[str], max_docs: int = DEFAULT_MAX_CONTEXT_DOCS) -> list:
        """Truy xuất lai: lọc theo key YOLO + tìm kiếm tương đồng ngữ nghĩa."""
        if self.vector_store is None:
            # Chế độ dự phòng khi thiếu FAISS
            return [entry_to_text(e) for e in (lookup(k) for k in keys) if e]

        docs, seen = [], set()
        for key in keys:
            docs += self.vector_store.similarity_search(query, k=1, filter={"key": key}, fetch_k=100)
        docs += self.vector_store.similarity_search(query, k=max_docs)

        unique = []
        for d in docs:
            content = d if isinstance(d, str) else d.page_content
            if content not in seen:
                seen.add(content)
                unique.append(d)
        return unique[:max_docs]

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
        """Chế độ dự phòng xuất phác đồ Markdown chuẩn khi không có LLM."""
        entries = [e for _, _, e in diagnosis if e]
        if entries:
            return "\n\n---\n\n".join(entry_to_answer(e) for e in entries)
        return "Chưa có mô hình ngôn ngữ cục bộ. Tài liệu liên quan:\n\n" + format_context(texts)
