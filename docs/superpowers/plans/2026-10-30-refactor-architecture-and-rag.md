# Refactor Architecture & Improve Embedding and Chunking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Chuẩn hóa toàn bộ codebase theo kiến trúc 3 tầng (Layered Architecture), tạo service engine trung tâm `PlantDoctorPipeline`, tinh gọn `streamlit_app.py`, và nâng cấp kỹ thuật chunking/embedding linh hoạt hỗ trợ cả JSON lẫn tài liệu tự do (PDF/MD/TXT).

**Architecture:** Áp dụng mô hình 3 tầng: Tầng Presentation (`app/streamlit_app.py`) giao tiếp thông qua Tầng Service Engine (`src/plant_doctor/pipeline.py`), bên dưới là Tầng Domain & RAG (`knowledge.py`, `rag/`, `training/`). Hệ thống RAG được cải tiến để nạp trực tiếp tài liệu thô tự do kết hợp metadata tagging đệ quy, giảm phụ thuộc vào JSON cứng.

**Tech Stack:** Python 3.10+, PyTorch, Ultralytics YOLOv8, Sentence-Transformers, HuggingFace Transformers, PEFT, LangChain (FAISS, RecursiveCharacterTextSplitter, pypdf), Streamlit, Pytest.

**Spec:** `docs/superpowers/specs/2026-10-30-refactor-architecture-design.md`

## Global Constraints

- Chạy 100% offline: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`.
- Giữ nguyên toàn bộ CLI shortcuts trong `pyproject.toml` (`pd-prepare-dataset`, `pd-train-yolo`, `pd-train-embedding`, `pd-train-llm`, `pd-build-index`).
- Tất cả các bài kiểm tra `pytest` phải chạy trên môi trường CPU không cần nạp GPU hay trọng số mô hình nặng.

## Review Focus

1. **Khả năng Fallback khi thiếu Model**: Khi không có weights YOLO, vector index hoặc merged LLM, pipeline phải tự động suy biến an toàn (graceful fallback) mà không gây crash ứng dụng.
2. **Xung đột mã hóa ký tự (Encoding UTF-8)**: Đảm bảo đọc và ghi các file tri thức tiếng Việt không gặp lỗi mã hóa trên Windows console (cp1252).
3. **Cửa sổ trượt bộ nhớ (Conversation Memory Window)**: `history_pairs` phải bảo toàn đúng thứ tự các cặp `(user, assistant)` gần nhất (k=3) cho multi-turn chat.
4. **Tách biệt giao diện và logic**: `app/streamlit_app.py` không được trực tiếp gọi các API thấp của PyTorch hay LangChain FAISS mà phải gọi qua `PlantDoctorPipeline`.
5. **Tính linh hoạt của Chunking**: Hàm build vector store phải xử lý êm cả khi không có file PDF nào lẫn khi có nhiều file PDF/MD/TXT phức tạp.

---

### Task 1: Chuẩn hóa System Prompt & Prompts Interface

**Files:**
- Modify: `src/plant_doctor/prompts.py`
- Test: `tests/test_knowledge.py`

**Interfaces:**
- Consumes: None
- Produces: `SYSTEM_PROMPT: str`, `format_context(texts: list) -> str`, `build_messages(context: str, question: str, history: list | None = None) -> list[dict]`

- [ ] **Step 1: Viết test cho System Prompt mở rộng và format context**

Bổ sung test trong `tests/test_knowledge.py`:
```python
def test_system_prompt_general_and_farming():
    assert "chuyên gia bệnh học thực vật" in SYSTEM_PROMPT.lower()
    assert "chào hỏi" in SYSTEM_PROMPT.lower() or "đời sống" in SYSTEM_PROMPT.lower()
```

- [ ] **Step 2: Chạy test để xác nhận trạng thái hiện tại**

Run: `python -m pytest tests/test_knowledge.py -k test_system_prompt_general_and_farming`
Expected: PASS hoặc FAIL tùy phiên bản text hiện tại.

- [ ] **Step 3: Cập nhật `src/plant_doctor/prompts.py` với nội dung chuẩn hóa**

Đảm bảo `SYSTEM_PROMPT` phân tách rõ 2 luồng: chuyên môn bệnh cây (bám sát tài liệu) và giao tiếp thông thường (tự nhiên, không từ chối vô lý).

- [ ] **Step 4: Chạy toàn bộ test suite**

Run: `python -m pytest tests/test_knowledge.py`
Expected: PASS (6+ tests)

- [ ] **Step 5: Commit**

```bash
git add src/plant_doctor/prompts.py tests/test_knowledge.py
git commit -m "refactor(prompts): update system prompt to support dual-mode general and agricultural chat"
```

---

### Task 2: Nâng cấp Kỹ thuật Chunking & Đóng gói FAISS Index

**Files:**
- Modify: `src/plant_doctor/rag/build_index.py`
- Test: `tests/test_rag.py`

**Interfaces:**
- Consumes: `knowledge.load_kb()`, `config.KB_DIR`, `config.EMBED_DIR`, `config.FAISS_DIR`
- Produces: 
  - `load_documents(kb_dir: Path) -> list[Document]`
  - `chunk_documents(docs: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200, kb: list | None = None) -> list[Document]`
  - `build_faiss_index(kb_dir: Path, embed_dir: Path, output_dir: Path, chunk_size: int = 1000, chunk_overlap: int = 200) -> int`

- [ ] **Step 1: Viết failing test cho hàm `chunk_documents` và `load_documents`**

Tạo `tests/test_rag.py`:
```python
from langchain_core.documents import Document
from plant_doctor.rag.build_index import chunk_documents

def test_chunk_documents_overlap_and_tagging():
    sample_text = ("Cây cà phê bị bệnh gỉ sắt. " * 50)  # > 1000 chars
    doc = Document(page_content=sample_text, metadata={"source": "test.txt"})
    fake_kb = [{"key": "coffee_leaf_rust", "crop_vi": "Cà phê", "crop_en": "Coffee", "name_vi": "Bệnh gỉ sắt", "name_en": "Rust"}]
    chunks = chunk_documents([doc], chunk_size=300, chunk_overlap=50, kb=fake_kb)
    assert len(chunks) > 1
    assert all(c.metadata.get("key") == "coffee_leaf_rust" for c in chunks)
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `python -m pytest tests/test_rag.py`
Expected: FAIL with ImportError / function not found.

- [ ] **Step 3: Tái cấu trúc `src/plant_doctor/rag/build_index.py`**

Tách các hàm pure function:
- `load_documents`: Đọc an toàn PDF (pypdf), Markdown, Text.
- `chunk_documents`: Dùng `RecursiveCharacterTextSplitter(chunk_size, chunk_overlap)` và chạy `tag_disease`.
- `build_faiss_index`: Nạp Embedding, khởi tạo FAISS với `MAX_INNER_PRODUCT`, lưu ra đĩa và trả về số lượng documents.
- `main()`: Bọc CLI với `argparse` cho phép tùy biến `--chunk-size` và `--chunk-overlap`.

- [ ] **Step 4: Chạy lại test**

Run: `python -m pytest tests/test_rag.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/plant_doctor/rag/build_index.py tests/test_rag.py
git commit -m "feat(rag): modularize document loading, recursive chunking and index building"
```

---

### Task 3: Chuẩn hóa các Modules Training thành Callable Functions

**Files:**
- Modify: `src/plant_doctor/training/train_yolo.py`
- Modify: `src/plant_doctor/training/train_embedding.py`
- Modify: `src/plant_doctor/training/train_llm.py`
- Test: `tests/test_training_interfaces.py`

**Interfaces:**
- Consumes: `config.*`
- Produces:
  - `train_yolo(data_yaml: Path, model_weights: Path, epochs: int, batch: int, imgsz: int, freeze: int) -> dict`
  - `train_custom_embedding(base_model: str, output_dir: Path, epochs: int, batch: int) -> float`
  - `train_custom_llm(base_model: str, adapter_dir: Path, merged_dir: Path, epochs: float, lr: float, batch: int) -> None`

- [ ] **Step 1: Viết test kiểm tra interface và importability của các module training**

Tạo `tests/test_training_interfaces.py`:
```python
from plant_doctor.training.train_yolo import train_yolo
from plant_doctor.training.train_embedding import build_pairs, queries_for
from plant_doctor.training.train_llm import build_samples

def test_training_pure_functions_exist():
    assert callable(train_yolo)
    assert callable(build_pairs)
    assert callable(build_samples)
```

- [ ] **Step 2: Chạy test để xác nhận kết quả**

Run: `python -m pytest tests/test_training_interfaces.py`
Expected: FAIL do hàm `train_yolo` chưa được định nghĩa độc lập (đang nằm trong `main`).

- [ ] **Step 3: Tái cấu trúc 3 file training**

- Trong `train_yolo.py`: Tách logic từ `main()` thành `train_yolo(...)`. `main()` chỉ parse args và gọi `train_yolo(...)`.
- Trong `train_embedding.py`: Tách logic từ `main()` thành `train_custom_embedding(...)`.
- Trong `train_llm.py`: Tách logic từ `main()` thành `train_custom_llm(...)`.

- [ ] **Step 4: Chạy test kiểm tra**

Run: `python -m pytest tests/test_training_interfaces.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/plant_doctor/training/ tests/test_training_interfaces.py
git commit -m "refactor(training): expose pure training functions with cli wrappers"
```

---

### Task 4: Xây dựng Service Engine `PlantDoctorPipeline`

**Files:**
- Create: `src/plant_doctor/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `config.*`, `knowledge.lookup`, `knowledge.entry_to_answer`, `knowledge.entry_to_text`, `prompts.build_messages`, `prompts.format_context`
- Produces: Class `PlantDoctorPipeline`
  - `load_yolo() -> Any`
  - `load_vector_store() -> Any`
  - `load_llm() -> tuple[Any, Any]`
  - `diagnose_image(image: Image.Image, conf_thr: float = 0.40) -> tuple[list[tuple[str, float, dict | None]], np.ndarray]`
  - `retrieve_context(query: str, keys: list[str], max_docs: int = 4) -> list[Document]`
  - `stream_answer(messages: list[dict], max_new_tokens: int = 512) -> Iterator[str]`
  - `fallback_answer(diagnosis: list[tuple[str, float, dict | None]], texts: list[str]) -> str`

- [ ] **Step 1: Viết test cho `PlantDoctorPipeline` (unit test với mock/offline)**

Tạo `tests/test_pipeline.py`:
```python
from unittest.mock import MagicMock
from plant_doctor.pipeline import PlantDoctorPipeline

def test_pipeline_fallback_answer():
    pipeline = PlantDoctorPipeline(load_models=False)
    fake_entry = {
        "name_vi": "Gỉ sắt", "name_en": "Rust", "crop_vi": "Cà phê", "crop_en": "Coffee",
        "pathogen": "Nấm", "symptoms": "Đốm vàng", "causes": "Mưa ẩm",
        "bio_treatment": "Phun đồng", "chemical_note": "Hạn chế", "prevention": "Tỉa cành"
    }
    diagnosis = [("Rust", 0.95, fake_entry)]
    res = pipeline.fallback_answer(diagnosis, [])
    assert "Gỉ sắt" in res
    assert "Phác đồ sinh học" in res

def test_pipeline_retrieve_without_store():
    pipeline = PlantDoctorPipeline(load_models=False)
    docs = pipeline.retrieve_context("cách trị", ["coffee_leaf_rust"])
    assert len(docs) > 0
    assert "coffee" in docs[0].lower() or "cà phê" in docs[0].lower()
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `python -m pytest tests/test_pipeline.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'plant_doctor.pipeline'`

- [ ] **Step 3: Triển khai `src/plant_doctor/pipeline.py`**

Tập trung toàn bộ logic AI:
- `PlantDoctorPipeline.__init__(load_models=True)`
- Quản lý lazy load hoặc nạp trước `yolo`, `vector_store`, `llm` (kèm tokenizer).
- Hàm `diagnose_image`: nhận PIL Image $\rightarrow$ predict YOLO $\rightarrow$ lookup nhãn $\rightarrow$ render bounding box.
- Hàm `retrieve_context`: thực hiện Hybrid Search (lọc theo key YOLO + similarity_search).
- Hàm `stream_answer`: khởi tạo generator chạy qua TextIteratorStreamer trên luồng Thread.
- Hàm `fallback_answer`: định dạng phác đồ chuẩn khi máy không có LLM.

- [ ] **Step 4: Chạy lại test pipeline**

Run: `python -m pytest tests/test_pipeline.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/plant_doctor/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): create central PlantDoctorPipeline service engine"
```

---

### Task 5: Tinh gọn `app/streamlit_app.py` và Kết nối với Pipeline

**Files:**
- Modify: `app/streamlit_app.py`
- Test: Chạy thử app và kiểm thử luồng chat/chẩn đoán

**Interfaces:**
- Consumes: `PlantDoctorPipeline` từ `plant_doctor.pipeline`
- Produces: Giao diện Web mỏng, hoàn toàn không phụ thuộc logic cấp thấp của mô hình.

- [ ] **Step 1: Viết test kiểm tra import của Streamlit App**

Tạo `tests/test_app_import.py`:
```python
def test_streamlit_imports_cleanly():
    import app.streamlit_app as st_app
    assert hasattr(st_app, "get_pipeline")
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `python -m pytest tests/test_app_import.py`
Expected: FAIL

- [ ] **Step 3: Tái cấu trúc `app/streamlit_app.py`**

- Thay thế toàn bộ các hàm `load_yolo`, `load_vector_store`, `load_llm`, `retrieve`, `stream_llm` bằng việc khởi tạo pipeline:
  ```python
  @st.cache_resource(show_spinner="Đang nạp Plant Doctor Engine...")
  def get_pipeline():
      return PlantDoctorPipeline()
  ```
- Rút gọn hàm `answer(question)`:
  ```python
  pipeline = get_pipeline()
  docs = pipeline.retrieve_context(query, keys)
  if pipeline.llm is None:
      text = pipeline.fallback_answer(diag, texts)
  else:
      messages = build_messages(format_context(texts), question, history_pairs())
      text = st.write_stream(pipeline.stream_answer(messages, max_new_tokens))
  ```
- File `streamlit_app.py` giảm từ ~266 dòng xuống dưới 150 dòng, thuần túy là code giao diện Streamlit.

- [ ] **Step 4: Chạy test kiểm tra import**

Run: `python -m pytest tests/test_app_import.py`
Expected: PASS

- [ ] **Step 5: Chạy toàn bộ test suite dự án**

Run: `python -m pytest`
Expected: 100% tests PASS

- [ ] **Step 6: Commit**

```bash
git add app/streamlit_app.py tests/test_app_import.py
git commit -m "refactor(app): slim down streamlit app to thin presentation layer over pipeline"
```

---

### Task 6: Kiểm thử Tích hợp Toàn diện & Cập nhật Tài liệu

**Files:**
- Modify: `README.md`
- Test: Toàn bộ test suite `pytest`

- [ ] **Step 1: Chạy toàn bộ test suite**

Run: `python -m pytest -v`
Expected: PASS toàn bộ các test trong `tests/`

- [ ] **Step 2: Cập nhật tài liệu `README.md`**

Mô tả cấu trúc 3 tầng mới, bổ sung hướng dẫn sử dụng class `PlantDoctorPipeline` trực tiếp từ code Python nếu người dùng muốn tích hợp vào hệ thống khác.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: update architecture documentation and pipeline usage instructions"
```
