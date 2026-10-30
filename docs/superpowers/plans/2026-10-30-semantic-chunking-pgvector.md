# Kế Hoạch Triển Khai Semantic Chunking & Lưu Trữ PostgreSQL (pgvector)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hiện thực hóa chính xác sơ đồ tuần tự: Đọc tài liệu từ `Document Source` $\rightarrow$ Đưa vào hàm `Generate Chunks Function` bằng thuật toán **Semantic Chunking** $\rightarrow$ Gửi chunks sang `Generate Embeddings Function` $\rightarrow$ Lưu trữ toàn bộ Vectors kèm Metadata vào **PostgreSQL Database sử dụng extension `pgvector`** (với URL cấu hình qua `.env`), đồng thời hỗ trợ truy xuất lai trong `PlantDoctorPipeline`.

**Architecture:** Bổ sung module `src/plant_doctor/rag/semantic_chunking.py` cài đặt thuật toán Semantic Chunking thuần túy. Xây dựng module `src/plant_doctor/rag/pgvector_store.py` quản lý schema bảng PostgreSQL (`plant_doctor_embeddings`: id, content, metadata JSONB, embedding vector(384)), tự động chạy `CREATE EXTENSION IF NOT EXISTS vector`, ghi vector và tìm kiếm tương đồng qua toán tử Cosine `<=>`. Tích hợp vào `pipeline.py` và `build_index.py`.

**Tech Stack:** Python 3.10+, PostgreSQL + pgvector, psycopg2-binary, python-dotenv, NumPy, Sentence-Transformers, LangChain Core, Pytest.

**Spec:** Sơ đồ tuần tự RAG: `Document Source` -> `RAG System` -> `Chunk Function (Semantic)` -> `Embedding Function` -> `Document DB (PostgreSQL pgvector with Metadata)`.

## Global Constraints

- Cấu hình URL kết nối thông qua biến môi trường `PGVECTOR_URL` (hỗ trợ file `.env` và `.env.example`).
- Nếu chưa có URL hoặc không kết nối được PostgreSQL thì pipeline vẫn giữ cơ chế fallback an toàn (không crash ứng dụng).
- Các unit tests trong `pytest` chạy độc lập với mock connection để không đòi hỏi database thật khi test.

## Review Focus

1. **Khởi tạo Schema & Extension An toàn**: Kiểm tra lệnh `CREATE EXTENSION IF NOT EXISTS vector;` và tạo bảng tự động mà không gây lỗi khi bảng đã tồn tại.
2. **Xác định ranh giới ngữ nghĩa (Breakpoint Threshold)**: Semantic Chunking dựa trên khoảng cách vector giữa các câu liên tiếp, gom cụm các câu có cùng ngữ nghĩa.
3. **Lưu trữ Metadata JSONB**: Metadata (`key`, `source`, `page`) được lưu dưới dạng JSONB để có thể lọc chính xác (Symbolic Filter) theo nhãn YOLO.
4. **Tìm kiếm Cosine Similarity chuẩn**: Sử dụng toán tử khoảng cách Cosine `<=>` của pgvector kết hợp vector đã chuẩn hóa $L_2$.
5. **Đọc URL từ file `.env`**: Hỗ trợ nạp cấu hình tự động khi ứng dụng khởi chạy.

---

### Task 1: Cấu hình Môi trường & Module Semantic Chunking (`semantic_chunking.py`)

**Files:**
- Create: `src/plant_doctor/rag/semantic_chunking.py`
- Modify: `requirements.txt`
- Modify: `pyproject.toml`
- Create: `.env.example`
- Test: `tests/test_semantic_chunking.py`

**Interfaces:**
- Consumes: `langchain_core.documents.Document`
- Produces: 
  - `split_sentences(text: str) -> list[str]`
  - `semantic_chunking(text: str, embed_fn: Callable[[list[str]], np.ndarray], percentile_threshold: float = 85.0, min_chunk_len: int = 150) -> list[str]`
  - `split_documents_semantically(docs: list[Document], embed_fn: Callable[[list[str]], np.ndarray], percentile_threshold: float = 85.0, kb: list | None = None) -> list[Document]`

- [x] **Step 1: Viết failing test cho `semantic_chunking.py`**
- [x] **Step 2: Chạy test xác nhận fail**
- [x] **Step 3: Triển khai thuật toán Semantic Chunking và cập nhật file cấu hình**
- [x] **Step 4: Chạy test xác nhận pass**
- [x] **Step 5: Commit**

---

### Task 2: Xây dựng Module PostgreSQL pgvector Store (`pgvector_store.py`)

**Files:**
- Create: `src/plant_doctor/rag/pgvector_store.py`
- Modify: `src/plant_doctor/config.py`
- Test: `tests/test_pgvector_store.py`

**Interfaces:**
- Consumes: `config.PGVECTOR_URL`
- Produces: Class `PGVectorStore`
  - `init_db()`
  - `add_documents(docs: list[Document], embeddings: list[list[float]]) -> int`
  - `similarity_search(query_vector: list[float], k: int = 2, filter_key: str | None = None) -> list[Document]`

- [x] **Step 1: Viết failing test cho `PGVectorStore` (với mock connection)**
- [x] **Step 2: Chạy test xác nhận fail**
- [x] **Step 3: Triển khai `src/plant_doctor/rag/pgvector_store.py`**
- [x] **Step 4: Chạy test xác nhận pass**
- [x] **Step 5: Commit**

---

### Task 3: Tích hợp Semantic Chunking & pgvector vào Pipeline Đóng Gói và Truy Xuất

**Files:**
- Modify: `src/plant_doctor/rag/build_index.py`
- Modify: `src/plant_doctor/pipeline.py`
- Test: `tests/test_pipeline.py`
- Test: `tests/test_rag.py`

**Interfaces:**
- Consumes: `semantic_chunking`, `pgvector_store`
- Produces: 
  - `build_index` hỗ trợ lưu trực tiếp vào PostgreSQL pgvector khi có URL.
  - `PlantDoctorPipeline.retrieve_context` tự động ưu tiên truy vấn pgvector nếu đã kết nối.

- [x] **Step 1: Viết failing test cho `pipeline` hỗ trợ pgvector**
- [x] **Step 2: Chạy test xác nhận fail**
- [x] **Step 3: Cập nhật `build_index.py` và `pipeline.py`**
- [x] **Step 4: Chạy test xác nhận pass**
- [x] **Step 5: Commit**

---

### Task 4: Kiểm thử Toàn bộ Hệ thống & Cập nhật Tài liệu

**Files:**
- Modify: `README.md`
- Test: Toàn bộ test suite `pytest`

- [x] **Step 1: Chạy toàn bộ test suite dự án**
- [x] **Step 2: Cập nhật tài liệu hướng dẫn PostgreSQL pgvector trong `README.md`**
- [x] **Step 3: Commit**
