# Kế Hoạch Loại Bỏ Hoàn Toàn diseases.json & Chuyển Sang Unstructured RAG

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Loại bỏ hoàn toàn file `data/knowledge_base/diseases.json` khỏi codebase, chuyển đổi tri thức sang tài liệu Markdown/PDF phi cấu trúc, nâng cấp hệ thống sang kiến trúc **Unstructured RAG thuần túy** với Query Expansion tự động từ nhãn YOLO.

**Architecture:** 
1. Chuyển đổi toàn bộ tri thức hiện có sang tài liệu Markdown phi cấu trúc `data/knowledge_base/cam_nang_benh_cay_trong.md` và xóa vĩnh viễn `diseases.json`.
2. Tái cấu trúc `src/plant_doctor/knowledge.py` và tạo `src/plant_doctor/rag/query_expansion.py`: phân tích nhãn YOLO tự động (`parse_label`) và sinh câu truy vấn ngữ nghĩa phong phú (`label_to_query`) mà không phụ thuộc vào bất kỳ file JSON nào.
3. Cập nhật `build_index.py`, `train_embedding.py`, `train_llm.py` để đọc trực tiếp các file tài liệu (`.md`, `.txt`, `.pdf`).
4. Cập nhật `pipeline.py` sang **Open Semantic Retrieval**: tìm kiếm tương đồng vector trực tiếp không cần lọc cứng `filter={'key': ...}` và truyền tri thức vào Prompt cho LLM sinh phác đồ.

**Tech Stack:** Python 3.10+, LangChain Core / Community, FAISS, PostgreSQL + pgvector, Sentence-Transformers, PyPDF, Pytest.

**Spec:** Loại bỏ 100% sự hiện diện của `diseases.json`.

## Global Constraints

- Xóa bỏ file `data/knowledge_base/diseases.json` và cấu hình `KB_JSON`.
- Hệ thống hoạt động 100% dựa trên tài liệu văn bản tự do (`.md`, `.txt`, `.pdf`).
- Không làm giảm tốc độ suy luận: giữ nguyên cơ chế Context Pruning (top-2), streaming token, và phân tích nhãn nhanh $O(1)$.
- Đảm bảo toàn bộ test suite (`pytest`) đều PASS mà không có bất kỳ test nào phụ thuộc vào `diseases.json`.

## Review Focus

1. **Xóa `diseases.json` không gây lỗi CI/CD**: Xóa step `Validate diseases.json` trong `.github/workflows/ci.yml`.
2. **Nhận diện nhãn mới ngoài 30 bệnh**: `parse_label` phân tích chuẩn xác nhãn YOLO bất kỳ (ví dụ `Rice___Blast`, `Coffee___Rust`) thành thông tin hiển thị đẹp mắt.
3. **Open Semantic Retrieval**: Tìm kiếm tương đồng ngữ nghĩa trên PostgreSQL `pgvector` hoặc FAISS mà không cần điều kiện lọc `filter_key`.
4. **Đào tạo mô hình không cần JSON**: `train_embedding.py` và `train_llm.py` trích xuất mẫu học trực tiếp từ tài liệu Markdown hoặc file `jsonl`.
5. **Giao diện Web không bị vỡ**: Giao diện `streamlit_app.py` hiển thị nhãn chẩn đoán và tài liệu tham khảo nguồn `.md`/`.pdf` mượt mà.

---

### Task 1: Chuyển Đổi Tri Thức Sang Markdown & Xóa Bỏ Vĩnh Viễn `diseases.json`

**Files:**
- Create: `data/knowledge_base/cam_nang_benh_cay_trong.md`
- Delete: `data/knowledge_base/diseases.json`
- Modify: `src/plant_doctor/config.py`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Produces: `data/knowledge_base/cam_nang_benh_cay_trong.md` (chứa toàn bộ thông tin bệnh học, triệu chứng, điều trị sinh học và phòng ngừa)
- Removes: `KB_JSON` trong `config.py`

- [x] **Step 1: Viết script sinh `cam_nang_benh_cay_trong.md` từ `diseases.json`**
- [x] **Step 2: Chạy script tạo file Markdown và kiểm tra nội dung**
- [x] **Step 3: Xóa `data/knowledge_base/diseases.json` và xóa `KB_JSON` trong `src/plant_doctor/config.py`**
- [x] **Step 4: Cập nhật `.github/workflows/ci.yml` loại bỏ bước kiểm tra JSON**
- [x] **Step 5: Commit**

```bash
git add data/knowledge_base/cam_nang_benh_cay_trong.md src/plant_doctor/config.py .github/workflows/ci.yml
git rm data/knowledge_base/diseases.json
git commit -m "refactor(kb): migrate structured diseases.json to unstructured markdown and remove json"
```

---

### Task 2: Module Phân Tích Nhãn YOLO & Query Expansion (`query_expansion.py` & `knowledge.py`)

**Files:**
- Create: `src/plant_doctor/rag/query_expansion.py`
- Modify: `src/plant_doctor/knowledge.py`
- Test: `tests/test_query_expansion.py`
- Modify: `tests/test_knowledge.py`

**Interfaces:**
- Produces:
  - `parse_label(class_name: str) -> dict`: Tách nhãn `Crop___Disease` thành `{crop_en, crop_vi, disease_en, disease_vi, is_healthy, key, display_name}`.
  - `label_to_query(class_name: str, aspect: str | None = None) -> str`: Tạo câu truy vấn giàu ngữ nghĩa để tìm kiếm trong Vector DB.
  - `lookup(class_name: str) -> dict`: Tự động phân tích nhãn không phụ thuộc JSON.

- [x] **Step 1: Viết failing test cho `query_expansion.py` và `test_knowledge.py` (không còn `load_kb`)**
- [x] **Step 2: Chạy test xác nhận fail**
- [x] **Step 3: Triển khai `query_expansion.py` và cập nhật `knowledge.py`**
- [x] **Step 4: Chạy test xác nhận pass**
- [x] **Step 5: Commit**

```bash
git add src/plant_doctor/rag/query_expansion.py src/plant_doctor/knowledge.py tests/test_query_expansion.py tests/test_knowledge.py
git commit -m "feat(rag): add rule-based label parser and semantic query expansion"
```

---

### Task 3: Cập Nhật Lập Chỉ Mục & Huấn Luyện Từ Tài Liệu Tự Do (`build_index.py`, `train_embedding.py`, `train_llm.py`)

**Files:**
- Modify: `src/plant_doctor/rag/build_index.py`
- Modify: `src/plant_doctor/training/train_embedding.py`
- Modify: `src/plant_doctor/training/train_llm.py`
- Modify: `tests/test_rag.py`
- Modify: `tests/test_training_interfaces.py`

**Interfaces:**
- Modifies:
  - `build_vector_store`: Nạp trực tiếp tất cả tài liệu `.md`, `.txt`, `.pdf` từ `data/knowledge_base/`, chia chunk theo ngữ nghĩa (`semantic_chunking`), không gán cứng metadata JSON.
  - `train_embedding.py` & `train_llm.py`: Trích xuất mẫu huấn luyện từ tài liệu `.md` hoặc `embedding_pairs.jsonl`.

- [x] **Step 1: Viết test cho việc index tài liệu thuần túy không có JSON**
- [x] **Step 2: Chạy test xác nhận fail**
- [x] **Step 3: Triển khai cập nhật cho `build_index.py`, `train_embedding.py`, `train_llm.py`**
- [x] **Step 4: Chạy test xác nhận pass**
- [x] **Step 5: Commit**

```bash
git add src/plant_doctor/rag/build_index.py src/plant_doctor/training/train_embedding.py src/plant_doctor/training/train_llm.py tests/
git commit -m "feat(rag): adapt build_index and training modules for pure unstructured documents"
```

---

### Task 4: Chuyển Đổi Pipeline Sang Open Semantic Retrieval & Cập Nhật Streamlit App

**Files:**
- Modify: `src/plant_doctor/pipeline.py`
- Modify: `app/streamlit_app.py`
- Modify: `tests/test_pipeline.py`

**Interfaces:**
- Modifies:
  - `PlantDoctorPipeline.retrieve_context(query, keys=None, max_docs=2)`: Chạy Open Semantic Search trên Vector Store (PostgreSQL `pgvector` hoặc FAISS). Nếu có nhãn YOLO, dùng `label_to_query(cls)` làm truy vấn mở rộng.
  - `PlantDoctorPipeline.diagnose_image`: Dùng `parse_label` hiển thị nhãn đẹp.
  - Loại bỏ phụ thuộc vào `diseases.json` trong `get_cached_answer` (chuyển sang LLM streaming dựa trên RAG từ Markdown).

- [x] **Step 1: Viết failing test cho `pipeline.py`**
- [x] **Step 2: Chạy test xác nhận fail**
- [x] **Step 3: Triển khai cập nhật `pipeline.py` và `streamlit_app.py`**
- [x] **Step 4: Chạy test xác nhận pass**
- [x] **Step 5: Commit**

```bash
git add src/plant_doctor/pipeline.py app/streamlit_app.py tests/test_pipeline.py
git commit -m "feat(pipeline): implement open semantic retrieval without metadata filter"
```

---

### Task 5: Kiểm Thử Toàn Diện Hệ Thống & Cập Nhật Tài Liệu (`README.md`)

**Files:**
- Modify: `README.md`
- Test: Toàn bộ test suite `pytest`

- [x] **Step 1: Chạy toàn bộ test suite dự án `pytest`**
- [x] **Step 2: Cập nhật `README.md` giải thích kiến trúc Unstructured RAG (loại bỏ JSON)**
- [x] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: update documentation for pure unstructured RAG architecture"
```
